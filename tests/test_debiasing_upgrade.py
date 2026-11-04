"""
Tests for Feature 3: Raw vs Debiased Comparison Lab & Correct Ratio Definitions
"""

import pytest
from backend.app.services.debiasing import von_neumann_debias


class TestDebiasingRatios:
    def test_ratios_for_alternating_stream(self):
        # "0110" -> 4 input bits, 2 complete pairs ("01" and "10")
        # Both pairs are retained -> 2 output bits ("01")
        # output_to_input_ratio = 2 / 4 = 0.50
        # pair_retention_ratio = 2 / 2 = 1.00
        res = von_neumann_debias("0110")
        assert res["input_bits"] == 4
        assert res["total_complete_pairs"] == 2
        assert res["output_bits"] == 2
        assert res["output_to_input_ratio"] == 0.50
        assert res["pair_retention_ratio"] == 1.00
        assert res["discarded_bits"] == 2
        assert res["extraction_efficiency"] == 0.50

    def test_ratios_for_all_four_pairs(self):
        # "00011011" -> 8 bits, 4 pairs:
        # 00 (discard), 01 (out 0), 10 (out 1), 11 (discard)
        # output_bits = 2 ("01")
        # output_to_input_ratio = 2 / 8 = 0.25 (exactly 25%)
        # pair_retention_ratio = 2 / 4 = 0.50 (exactly 50%)
        res = von_neumann_debias("00011011")
        assert res["input_bits"] == 8
        assert res["total_complete_pairs"] == 4
        assert res["output_bits"] == 2
        assert res["output_to_input_ratio"] == 0.25
        assert res["pair_retention_ratio"] == 0.50
        assert res["discarded_bits"] == 6

    def test_min_entropy_present_in_input_and_output_stats(self):
        res = von_neumann_debias("0110" * 50)
        assert "min_entropy_per_bit" in res["input_stats"]
        assert "min_entropy_per_bit" in res["output_stats"]
        assert res["output_stats"]["min_entropy_per_bit"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_stream_ratios(self):
        res = von_neumann_debias("")
        assert res["output_to_input_ratio"] == 0.0
        assert res["pair_retention_ratio"] == 0.0
        assert res["discarded_bits"] == 0

