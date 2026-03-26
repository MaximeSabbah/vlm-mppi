"""Tests for Embodied-R1 output parsing — no GPU required."""

import pytest

from vlm_mppi.model import Ability, PointingResult, _parse_output


class TestParseOutput:
    """Validate coordinate extraction from the official output format:
    <think> reasoning </think><answer><point>[[x1,y1], [x2,y2], ...]</point></answer>
    Coordinates are in pixel space (not normalised).
    """

    def test_canonical_format(self):
        raw = (
            "<think>The bolt head is at the top of the tool.</think>"
            "<answer><point>[[320, 240]]</point></answer>"
        )
        result = _parse_output(raw, Ability.OFG)
        assert result.n_points == 1
        assert result.points_px[0] == (320.0, 240.0)

    def test_multi_point(self):
        raw = (
            "<think>Trajectory from block to target.</think>"
            "<answer><point>[[100, 200], [150, 180], [200, 160]]</point></answer>"
        )
        result = _parse_output(raw, Ability.VTG)
        assert result.n_points == 3
        assert result.points_px[0] == (100.0, 200.0)
        assert result.points_px[2] == (200.0, 160.0)

    def test_reasoning_extracted(self):
        raw = (
            "<think>The mug handle faces right so grasp from the side.</think>"
            "<answer><point>[[500, 300]]</point></answer>"
        )
        result = _parse_output(raw, Ability.OFG)
        assert "handle faces right" in result.reasoning
        assert result.has_points

    def test_no_points(self):
        raw = "<think>I cannot determine the grasp point.</think><answer></answer>"
        result = _parse_output(raw, Ability.OFG)
        assert result.n_points == 0
        assert not result.has_points

    def test_raw_output_preserved(self):
        raw = "<think>ok</think><answer><point>[[10, 20]]</point></answer>"
        result = _parse_output(raw, Ability.REG)
        assert result.raw_output == raw

    def test_vtg_eight_points(self):
        coords = ", ".join(f"[{i*10}, {i*5}]" for i in range(8))
        raw = f"<think>plan</think><answer><point>[{coords}]</point></answer>"
        result = _parse_output(raw, Ability.VTG)
        assert result.n_points == 8
        assert result.points_px[7] == (70.0, 35.0)


class TestPointingResult:
    def test_properties(self):
        r = PointingResult(ability=Ability.OFG, points_px=[(1.0, 2.0), (3.0, 4.0)])
        assert r.has_points
        assert r.n_points == 2

    def test_empty(self):
        r = PointingResult(ability=Ability.REG)
        assert not r.has_points
        assert r.n_points == 0
