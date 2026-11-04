"""
Tests for the Von Neumann Debiasing Service.
All tests use deterministic synthetic inputs.
"""

import pytest
from backend.app.services.debiasing import von_neumann_debias, ALGORITHM_NOTE


class TestVonNeumannPairMappings:
    """Test the core pair-to-bit mapping rules."""

    def test_pair_01_maps_to_0(self):
        result = von_neumann_debias("01")
        assert result["output_bitstream"] == "0"
        assert result["count_01"] == 1

    def test_pair_10_maps_to_1(self):
        result = von_neumann_debias("10")
        assert result["output_bitstream"] == "1"
        assert result["count_10"] == 1

    def test_pair_00_discarded(self):
        result = von_neumann_debias("00")
        assert result["output_bitstream"] == ""
        assert result["count_00"] == 1
        assert result["output_bits"] == 0

    def test_pair_11_discarded(self):
        result = von_neumann_debias("11")
        assert result["output_bitstream"] == ""
        assert result["count_11"] == 1
        assert result["output_bits"] == 0

    def test_all_four_pairs(self):
        result = von_neumann_debias("00011011")
        assert result["count_00"] == 1
        assert result["count_01"] == 1
        assert result["count_10"] == 1
        assert result["count_11"] == 1
        assert result["output_bits"] == 2
        assert result["output_bitstream"] == "01"


class TestPairCounts:
    def test_total_complete_pairs_even(self):
        result = von_neumann_debias("0101")
        assert result["total_complete_pairs"] == 2

    def test_total_complete_pairs_odd(self):
        result = von_neumann_debias("010")  # 3 bits → 1 pair + 1 discarded
        assert result["total_complete_pairs"] == 1

    def test_retained_plus_discarded_equals_total_pairs(self):
        bits = "001011001101"
        result = von_neumann_debias(bits)
        assert result["retained_pairs"] + result["discarded_pairs"] == result["total_complete_pairs"]


class TestEdgeCases:
    def test_empty_input(self):
        result = von_neumann_debias("")
        assert result["input_bits"] == 0
        assert result["output_bits"] == 0
        assert result["output_bitstream"] == ""
        assert any("empty" in w.lower() for w in result["warnings"])

    def test_single_bit_discarded(self):
        result = von_neumann_debias("0")
        assert result["output_bits"] == 0
        assert result["total_complete_pairs"] == 0

    def test_odd_length_warning(self):
        result = von_neumann_debias("011")
        assert any("unpaired" in w.lower() or "odd" in w.lower() for w in result["warnings"])

    def test_odd_length_uses_complete_pairs_only(self):
        result = von_neumann_debias("011")  # only pair is "01"
        assert result["total_complete_pairs"] == 1
        assert result["output_bitstream"] == "0"

    def test_all_equal_pairs_00_no_output(self):
        result = von_neumann_debias("0000")
        assert result["output_bits"] == 0
        assert result["count_00"] == 2

    def test_all_equal_pairs_11_no_output(self):
        result = von_neumann_debias("1111")
        assert result["output_bits"] == 0
        assert result["count_11"] == 2

    def test_all_equal_pairs_warning(self):
        result = von_neumann_debias("000000")
        assert any("equal" in w.lower() or "all pairs" in w.lower() or "no output" in w.lower()
                   for w in result["warnings"])

    def test_invalid_characters_raise_value_error(self):
        with pytest.raises(ValueError, match="invalid"):
            von_neumann_debias("01201")

    def test_invalid_character_letters(self):
        with pytest.raises(ValueError):
            von_neumann_debias("AB01")


class TestDeterminism:
    def test_same_input_produces_same_output(self):
        bits = "01100110010110010011"
        r1 = von_neumann_debias(bits)
        r2 = von_neumann_debias(bits)
        assert r1["output_bitstream"] == r2["output_bitstream"]

    def test_pair_order_preserved(self):
        """01 and 10 alternate → output should be 0,1,0,1"""
        bits = "01" * 4 + "10" * 4  # 01010101 10101010
        result = von_neumann_debias(bits)
        # 4 '01' pairs → 4 zeros, 4 '10' pairs → 4 ones
        assert result["output_bitstream"] == "0000" + "1111"


class TestExtractionEfficiency:
    def test_perfect_alternating_efficiency(self):
        """010101… all pairs are 01 or 10 → efficiency = 0.5"""
        bits = "01" * 100
        result = von_neumann_debias(bits)
        assert result["extraction_efficiency"] == pytest.approx(0.5, abs=0.01)

    def test_zero_efficiency_all_equal(self):
        result = von_neumann_debias("0000000000")
        assert result["extraction_efficiency"] == 0.0

    def test_efficiency_between_0_and_0_5(self):
        """Efficiency can never exceed 0.5 for i.i.d. bits."""
        bits = "0110011001101010010111" * 10
        result = von_neumann_debias(bits)
        assert 0.0 <= result["extraction_efficiency"] <= 0.5 + 0.01  # tolerance for small streams


class TestAlgorithmNote:
    def test_algorithm_note_present(self):
        result = von_neumann_debias("0110")
        assert result["algorithm_note"] == ALGORITHM_NOTE

    def test_algorithm_note_mentions_iid(self):
        assert "i.i.d" in ALGORITHM_NOTE.lower() or "independent" in ALGORITHM_NOTE.lower()

    def test_algorithm_note_warns_against_randomness_claim(self):
        note_lower = ALGORITHM_NOTE.lower()
        assert "does not" in note_lower or "not automatically" in note_lower


class TestLargeInput:
    def test_large_input_completes(self):
        import os
        # Generate a pseudo-random binary string for testing
        import random
        rng = random.Random(42)
        bits = "".join(rng.choice("01") for _ in range(10000))
        result = von_neumann_debias(bits)
        assert result["input_bits"] == 10000
        assert result["output_bits"] > 0
        assert result["output_bits"] < result["input_bits"]
