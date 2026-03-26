"""Embodied-R1 model wrapper for spatial reasoning and pointing."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from vlm_mppi.config import ModelConfig

logger = logging.getLogger(__name__)


class Ability(str, Enum):
    """Embodied-R1 pointing abilities."""

    REG = "REG"  # Referring Expression Grounding — locate a named object
    RRG = "RRG"  # Region Referring Grounding — find a target placement region
    OFG = "OFG"  # Object Functional Grounding — find a functional grasp point
    VTG = "VTG"  # Visual Trace Generation — generate a motion trajectory


# Official prompt templates from Yuan et al., ICLR 2026.
# Source: https://github.com/pickxiguapi/Embodied-R1/blob/main/inference_example.py
# Output format: <think>...</think><answer><point>[[x1,y1],...]</point></answer>
# Coordinates are in pixel space (not normalised).
PROMPTS: dict[Ability, str] = {
    Ability.REG: (
        "Provide one or more points coordinate of objects region {instruction}. "
        "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        "The answer consists only of several coordinate points, with the overall format being: "
        "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    ),
    Ability.OFG: (
        "Please provide the 2D points coordinate of the region this sentence describes: {instruction}. "
        "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        "The answer consists only of several coordinate points, with the overall format being: "
        "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    ),
    Ability.RRG: (
        "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
        "Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved. "
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        "The answer consists only of several coordinate points, with the overall format being: "
        "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    ),
    Ability.VTG: (
        "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
        "Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. "
        "You must provide the points in the order of the trajectory, and the number of points must be 8. "
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        "The answer consists only of several coordinate points, with the overall format being: "
        "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ..., [x8, y8]]</point></answer>."
    ),
}


@dataclass
class PointingResult:
    """Parsed pointing output from Embodied-R1."""

    ability: Ability
    reasoning: str = ""
    points_px: list[tuple[float, float]] = field(default_factory=list)
    raw_output: str = ""

    @property
    def has_points(self) -> bool:
        return len(self.points_px) > 0

    @property
    def n_points(self) -> int:
        return len(self.points_px)


class EmbodiedR1:
    """Wrapper around the Embodied-R1 3B VLM for robotic spatial reasoning.

    Usage::

        model = EmbodiedR1.load()
        result = model.point("scene.png", "pick up the red cup", Ability.OFG)
        print(result.points_px)  # [(312.5, 201.3)]
    """

    def __init__(self, model, processor, config: ModelConfig):
        self._model = model
        self._processor = processor
        self._config = config

    @classmethod
    def load(cls, config: Optional[ModelConfig] = None) -> "EmbodiedR1":
        """Load model from HuggingFace. Downloads weights on first call (~6 GB)."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        config = config or ModelConfig()
        logger.info("Loading %s ...", config.model_id)

        attn = "flash_attention_2" if config.flash_attn2 else None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            local_files_only=config.local_files_only,
            attn_implementation=attn,
        )
        processor = AutoProcessor.from_pretrained(
            config.model_id,
            local_files_only=config.local_files_only,
        )

        if config.torch_compile:
            logger.info("Compiling model with torch.compile (first inference will be slow)...")
            model = torch.compile(model, mode="reduce-overhead")

        logger.info("Model loaded on %s", next(model.parameters()).device)
        return cls(model, processor, config)

    def point(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        ability: Ability = Ability.VTG,
    ) -> PointingResult:
        """Run a pointing query.

        Args:
            image: file path or PIL image.
            instruction: natural language task description.
            ability: which pointing ability to invoke.

        Returns:
            Parsed PointingResult with pixel-space coordinates.
        """
        pil_image = self._load_pil(image)
        prompt_text = PROMPTS[ability].format(instruction=instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                repetition_penalty=self._config.repetition_penalty,
                do_sample=False,
            )

        # Strip prompt tokens
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = self._processor.decode(generated, skip_special_tokens=True)

        return _parse_output(raw_output, ability)

    def point_all(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        abilities: list[Ability] | None = None,
    ) -> dict[Ability, PointingResult]:
        """Run multiple pointing abilities on the same image.

        Args:
            image: file path or PIL image.
            instruction: task instruction.
            abilities: which abilities to run (default: OFG + RRG + VTG).

        Returns:
            Dict mapping each ability to its PointingResult.
        """
        if abilities is None:
            abilities = [Ability.OFG, Ability.RRG, Ability.VTG]

        results = {}
        for ability in abilities:
            logger.info("Running %s: %s", ability.value, instruction)
            results[ability] = self.point(image, instruction, ability)
            logger.info("  → %d points", results[ability].n_points)

        return results

    # ── helpers ───────────────────────────────────────────

    @staticmethod
    def _load_pil(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(image).convert("RGB")


# ── output parsing ────────────────────────────────────────────


def _parse_output(raw: str, ability: Ability) -> PointingResult:
    """Extract pixel coordinates from Embodied-R1's text output.

    Official format (Yuan et al., ICLR 2026):
        <think> reasoning </think><answer><point>[[x1,y1],[x2,y2],...]</point></answer>

    Coordinates are in pixel space (not normalised to [0,1000]).
    """
    result = PointingResult(ability=ability, raw_output=raw)

    # Extract reasoning from <think> block
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        result.reasoning = think_match.group(1).strip()

    # Extract coordinate list from <answer><point>[[...]]</point></answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else raw

    point_match = re.search(r"<point>(.*?)</point>", answer_text, re.DOTALL)
    if point_match:
        coords_str = point_match.group(1).strip()
        try:
            coords = ast.literal_eval(coords_str)
            if isinstance(coords, list):
                for pt in coords:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        result.points_px.append((float(pt[0]), float(pt[1])))
        except (ValueError, SyntaxError):
            # Fallback: extract all [x, y] pairs with regex
            for x_str, y_str in re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", coords_str):
                result.points_px.append((float(x_str), float(y_str)))

    return result
