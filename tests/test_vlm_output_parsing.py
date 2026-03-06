"""Tests for VLM output parsing and plan conversion."""

import json
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.mppi_interface import vlm_plan_to_se3_goal, extract_safety_margin


class TestJsonParsing:
    """Test that various VLM response formats are correctly parsed."""

    def test_clean_json(self):
        raw = '{"task_description": "push box", "done": false}'
        result = VLMPlanner._parse_json_response(raw)
        assert result["task_description"] == "push box"
        assert result["done"] is False

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"task_description": "push box", "done": false}\n```'
        result = VLMPlanner._parse_json_response(raw)
        assert result["task_description"] == "push box"

    def test_json_with_generic_fences(self):
        raw = '```\n{"done": true}\n```'
        result = VLMPlanner._parse_json_response(raw)
        assert result["done"] is True

    def test_json_with_whitespace(self):
        raw = '\n  {"done": false, "target_keypoints": {"1": [0.5, 0.1, 0.0]}}  \n'
        result = VLMPlanner._parse_json_response(raw)
        assert result["target_keypoints"]["1"] == [0.5, 0.1, 0.0]

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            VLMPlanner._parse_json_response("This is not JSON at all")


class TestPlanConversion:
    """Test conversion of VLM plans to MPPI goals."""

    def test_single_keypoint_goal(self):
        plan = {"target_keypoints": {"1": [0.5, 0.1, 0.2]}}
        goal = vlm_plan_to_se3_goal(plan)
        assert goal.shape == (7,)
        np.testing.assert_allclose(goal[:3], [0.5, 0.1, 0.2])
        np.testing.assert_allclose(goal[3:], [1, 0, 0, 0])  # identity quat

    def test_multiple_keypoints_centroid(self):
        plan = {
            "target_keypoints": {
                "1": [0.4, 0.0, 0.0],
                "2": [0.6, 0.0, 0.0],
            }
        }
        goal = vlm_plan_to_se3_goal(plan)
        np.testing.assert_allclose(goal[:3], [0.5, 0.0, 0.0])

    def test_empty_keypoints_raises(self):
        with pytest.raises(ValueError):
            vlm_plan_to_se3_goal({"target_keypoints": {}})

    def test_safety_margin_default(self):
        assert extract_safety_margin({}) == 0.05

    def test_safety_margin_custom(self):
        assert extract_safety_margin({"safety_margin_m": 0.15}) == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
