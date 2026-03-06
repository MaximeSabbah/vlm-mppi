"""
VLM Planner: wraps Qwen VLMs (Qwen2.5-VL / Qwen3-VL / Qwen3.5) for robotic task planning.

Supports two modes:
  1. Local inference via transformers (default)
  2. Remote inference via vLLM OpenAI-compatible server (faster, production)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .config import VLMConfig

logger = logging.getLogger(__name__)


class VLMPlanner:
    """High-level VLM planner that converts scene images + instructions into task plans."""

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._model = None
        self._processor = None
        self._vllm_client = None

        if self.config.use_vllm:
            self._init_vllm_client()
        else:
            self._init_local_model()

    # ── Model loading ────────────────────────────────────────────

    def _init_local_model(self):
        """Load the model locally with transformers.

        Uses AutoModelForImageTextToText so the same code works for
        Qwen2.5-VL, Qwen3-VL, Qwen3.5, and future Qwen VLM families.
        Requires transformers >= 4.57.0 for Qwen3-VL / Qwen3.5.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        logger.info("Loading %s (this may download weights on first run)...", self.config.model_id)

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)

        logger.info("Model loaded successfully.")

    def _init_vllm_client(self):
        """Connect to a running vLLM server."""
        from openai import OpenAI

        self._vllm_client = OpenAI(
            base_url=self.config.vllm_base_url,
            api_key="unused",  # vLLM doesn't require a key
        )
        logger.info("Connected to vLLM server at %s", self.config.vllm_base_url)

    # ── Inference ────────────────────────────────────────────────

    def query(self, image: Image.Image, prompt: str) -> str:
        """
        Send an image + text prompt to the VLM and return the raw response string.

        Args:
            image: PIL Image of the scene (RGB).
            prompt: Full text prompt including instructions.

        Returns:
            Raw string response from the VLM.
        """
        if self.config.use_vllm:
            return self._query_vllm(image, prompt)
        return self._query_local(image, prompt)

    def _query_local(self, image: Image.Image, prompt: str) -> str:
        """Run inference locally with transformers."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
        return self._processor.decode(generated_ids, skip_special_tokens=True)

    def _query_vllm(self, image: Image.Image, prompt: str) -> str:
        """Run inference via vLLM OpenAI-compatible API."""
        import base64
        import io

        # Encode image as base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self._vllm_client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    # ── Structured output ────────────────────────────────────────

    def plan(
        self,
        image: Image.Image,
        instruction: str,
        keypoints_3d: dict[str, list[float]],
        execution_history: Optional[list[dict]] = None,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate a structured task plan from a scene image and instruction.

        Args:
            image: PIL Image of the scene with keypoints overlaid.
            instruction: Natural language task instruction.
            keypoints_3d: Dict mapping keypoint labels to [x, y, z] positions (meters).
            execution_history: List of previous (plan, observation) pairs for replanning.
            prompt_template: Custom prompt. If None, uses the default keypoint prompt.
            max_retries: Number of retries if JSON parsing fails.

        Returns:
            Parsed dict with keys: task_description, interaction_object,
            grasp_required, target_keypoints, safety_margin_m, done.
        """
        if prompt_template is None:
            prompt_template = self._default_keypoint_prompt()

        # Build keypoint description
        kp_lines = []
        for label, pos in keypoints_3d.items():
            kp_lines.append(f"  Keypoint {label}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        kp_desc = "\n".join(kp_lines)

        # Build history description
        history_desc = "None"
        if execution_history:
            history_parts = []
            for i, entry in enumerate(execution_history):
                history_parts.append(f"  Step {i+1}: {entry.get('task_description', 'N/A')}")
            history_desc = "\n".join(history_parts)

        prompt = prompt_template.format(
            instruction=instruction,
            keypoints=kp_desc,
            history=history_desc,
        )

        # Query with retries
        for attempt in range(max_retries):
            raw = self.query(image, prompt)
            try:
                return self._parse_json_response(raw)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Attempt %d/%d: failed to parse VLM output: %s", attempt + 1, max_retries, e)
                if attempt == max_retries - 1:
                    raise ValueError(f"VLM did not return valid JSON after {max_retries} attempts. Last response:\n{raw}")

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Extract and parse JSON from VLM response, stripping markdown fences."""
        clean = raw.strip()
        # Remove markdown code fences if present
        if clean.startswith("```"):
            # Find the end of the opening fence line
            first_newline = clean.index("\n")
            clean = clean[first_newline + 1 :]
        if clean.endswith("```"):
            clean = clean[: clean.rfind("```")]
        clean = clean.strip()
        return json.loads(clean)

    @staticmethod
    def _default_keypoint_prompt() -> str:
        return """You are a robotic manipulation planner for a Franka Panda arm operating in a shared workspace with a human. You see a scene with numbered keypoints overlaid on objects.

Current keypoint positions (meters, X=forward, Y=left, Z=up):
{keypoints}

Execution history:
{history}

Task instruction: "{instruction}"

Determine WHERE each relevant keypoint should move to accomplish the NEXT step of the task. Output ONLY valid JSON (no markdown fences, no explanation):
{{
    "task_description": "what this step does",
    "interaction_object": "object name to manipulate",
    "grasp_required": true or false,
    "target_keypoints": {{
        "1": [x, y, z],
        "2": [x, y, z]
    }},
    "safety_margin_m": 0.05,
    "done": false
}}

Rules:
- Only include keypoints that need to move in target_keypoints.
- Express targets relative to other keypoints when possible.
- If a human is visible, increase safety_margin_m.
- Set "done": true only when the full task is complete.
- Push large objects, grasp small ones."""
