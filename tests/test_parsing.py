"""Tests for Embodied-R1 output parsing — no GPU required."""

import pytest

from vlm_mppi.model import Ability, _parse_output


class TestParseOutput:
    """Test coordinate extraction from model text output."""

    def test_point_tag_format(self):
        raw = "I see a red cup on the table. <point>320, 450</point>"
        result = _parse_output(raw, Ability.OFG, img_w=640, img_h=480)
        assert result.n_points == 1
        u, v = result.points_px[0]
        assert abs(u - 0.320 * 640) < 0.1
        assert abs(v - 0.450 * 480) < 0.1

    def test_parenthesis_format(self):
        raw = "The trace goes through (100, 200) then (300, 400) then (500, 600)"
        result = _parse_output(raw, Ability.VTG, img_w=1000, img_h=1000)
        assert result.n_points == 3
        assert abs(result.points_px[0][0] - 100.0) < 0.1
        assert abs(result.points_px[2][1] - 600.0) < 0.1

    def test_no_points(self):
        raw = "I cannot determine the grasp point from this image."
        result = _parse_output(raw, Ability.OFG, img_w=640, img_h=480)
        assert result.n_points == 0
        assert not result.has_points

    def test_reasoning_extracted(self):
        raw = "The mug handle faces right. I should grasp from the side. <point>500, 300</point>"
        result = _parse_output(raw, Ability.OFG, img_w=640, img_h=480)
        assert "handle faces right" in result.reasoning
        assert result.has_points

    def test_multiple_point_tags(self):
        raw = "<point>100, 200</point> then <point>300, 400</point>"
        result = _parse_output(raw, Ability.VTG, img_w=1000, img_h=1000)
        assert result.n_points == 2

    def test_coordinate_scaling(self):
        """Coordinates in [0,1000] should scale to image dimensions."""
        raw = "<point>500, 500</point>"
        result = _parse_output(raw, Ability.OFG, img_w=1920, img_h=1080)
        u, v = result.points_px[0]
        assert abs(u - 960.0) < 0.1  # 500/1000 * 1920
        assert abs(v - 540.0) < 0.1  # 500/1000 * 1080


class TestPointingResult:
    def test_properties(self):
        from vlm_mppi.model import PointingResult

        r = PointingResult(ability=Ability.OFG, points_px=[(1.0, 2.0), (3.0, 4.0)])
        assert r.has_points
        assert r.n_points == 2

    def test_empty(self):
        from vlm_mppi.model import PointingResult

        r = PointingResult(ability=Ability.REG)
        assert not r.has_points
        assert r.n_points == 0
