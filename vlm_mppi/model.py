"""Embodied-R1 model wrapper for spatial reasoning and pointing."""

from __future__ import annotations

import logging
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


# Prompt templates per ability. These follow the conventions from the
# Embodied-R1 paper (Yuan et al., ICLR 2026).
PROMPTS: dict[Ability, str] = {
    Ability.REG: (
        "In the given image, the task is: '{instruction}'. "
        "Please think step by step, then point to the referred object."
    ),
    Ability.RRG: (
        "In the given image, the task is: '{instruction}'. "
        "Please think step by step, reason about the spatial layout, "
        "and point to the target placement region."
    ),
    Ability.OFG: (
        "In the given image, the task is: '{instruction}'. "
        "Please think step by step, and point to the functional part "
        "of the object that should be grasped or interacted with."
    ),
    Ability.VTG: (
        "In the given image, the task is: '{instruction}'. "
        "Please think step by step, reason about the manipulation process, "
        "and generate a visual trace (sequence of points) showing the motion "
        "of the target object from the current position to the goal."
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

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        )
        processor = AutoProcessor.from_pretrained(config.model_id)

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
        from qwen_vl_utils import process_vision_info

        image_uri = self._resolve_image(image)
        prompt_text = PROMPTS[ability].format(instruction=instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._config.max_new_tokens
            )

        # Strip prompt tokens
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        raw_output = self._processor.decode(generated, skip_special_tokens=True)

        # Get image dimensions for coordinate conversion
        pil_image = self._load_pil(image)
        w, h = pil_image.size

        return _parse_output(raw_output, ability, w, h)

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
    def _resolve_image(image: str | Path | Image.Image) -> str:
        if isinstance(image, Image.Image):
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp.name)
            return f"file://{tmp.name}"
        return f"file://{Path(image).resolve()}"

    @staticmethod
    def _load_pil(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(image)


# ── output parsing ────────────────────────────────────────────


def _parse_output(raw: str, ability: Ability, img_w: int, img_h: int) -> PointingResult:
    """Extract pixel coordinates from Embodied-R1's text output.

    Coordinates follow Qwen2.5-VL convention: normalized to [0, 1000].
    """
    import re

    result = PointingResult(ability=ability, raw_output=raw)

    # Reasoning: everything before the first coordinate pattern
    split = re.split(r"<point>|\(\d{1,4}\s*,", raw, maxsplit=1)
    if split:
        result.reasoning = split[0].strip()

    # <point>x, y</point> format (primary)
    matches = re.findall(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", raw)

    # (x, y) fallback — common in VTG traces
    if not matches:
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", raw)

    # Convert normalized [0, 1000] → pixel coordinates
    for x_str, y_str in matches:
        u = int(x_str) / 1000.0 * img_w
        v = int(y_str) / 1000.0 * img_h
        result.points_px.append((u, v))

    return result
