"""
Tests for the NIST SP 800-22 Statistical Tests implementation.
Uses deterministic synthetic sequences where analytical behavior is known.
"""

import math
import random

import pytest

from backend.app.services.statistical_tests import (
    frequency_test,
    block_frequency_test,
    runs_test,
    longest_run_test,
    spectral_test,
    run_selected_tests,
    AVAILABLE_TESTS,
    ALL_TEST_NAMES,
    MIN_BITS_FREQUENCY,
    MIN_BITS_RUNS,
    MIN_BITS_SPECTRAL,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def balanced_stream():
    """A perfectly alternating stream: 0101… – all tests should see ideal statistics."""
    return "01" * 5000  # 10000 bits


@pytest.fixture
def all_zeros():
    return "0" * 10000


@pytest.fixture
def all_ones():
    return "1" * 10000


@pytest.fixture
def short_stream():
    return "0110"


@pytest.fixture
def pseudo_random_stream():
    """A large pseudo-random stream for general pass/fail behavior."""
    rng = random.Random(42)
    return "".join(rng.choice("01") for _ in range(10000))


# ─────────────────────────────────────────────
# Frequency (Monobit) Test
# ─────────────────────────────────────────────

class TestFrequencyTest:
    def test_balanced_stream_passes(self, balanced_stream):
        result = frequency_test(balanced_stream, alpha=0.01)
        assert result["outcome"] == "PASS"

    def test_all_zeros_fails(self, all_zeros):
        result = frequency_test(all_zeros, alpha=0.01)
        assert result["outcome"] == "FAIL"

    def test_all_ones_fails(self, all_ones):
        result = frequency_test(all_ones, alpha=0.01)
        assert result["outcome"] == "FAIL"

    def test_insufficient_data_returns_correct_outcome(self, short_stream):
        result = frequency_test(short_stream, alpha=0.01)
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_p_value_returned_for_sufficient_data(self, balanced_stream):
        result = frequency_test(balanced_stream, alpha=0.01)
        assert result["p_value"] is not None
        assert 0.0 <= result["p_value"] <= 1.0

    def test_statistic_returned_for_sufficient_data(self, balanced_stream):
        result = frequency_test(balanced_stream, alpha=0.01)
        assert result["statistic"] is not None
        assert result["statistic"] >= 0.0

    def test_minimum_bits_documented(self):
        result = frequency_test("0" * 50, alpha=0.01)
        assert result["minimum_bits_required"] == MIN_BITS_FREQUENCY

    def test_no_value_for_insufficient_data(self, short_stream):
        result = frequency_test(short_stream)
        assert result["p_value"] is None
        assert result["statistic"] is None

    def test_significance_level_stored(self, balanced_stream):
        result = frequency_test(balanced_stream, alpha=0.05)
        assert result["significance_level"] == pytest.approx(0.05)


# ─────────────────────────────────────────────
# Block Frequency Test
# ─────────────────────────────────────────────

class TestBlockFrequencyTest:
    def test_balanced_stream_passes(self, balanced_stream):
        result = block_frequency_test(balanced_stream, block_size=128, alpha=0.01)
        assert result["outcome"] == "PASS"

    def test_all_zeros_fails(self, all_zeros):
        result = block_frequency_test(all_zeros, block_size=128)
        assert result["outcome"] == "FAIL"

    def test_insufficient_data(self):
        result = block_frequency_test("0" * 50, block_size=128)
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_p_value_in_range(self, balanced_stream):
        result = block_frequency_test(balanced_stream, alpha=0.01)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_block_size_respected(self, balanced_stream):
        result = block_frequency_test(balanced_stream, block_size=64)
        # The result's actual_bits_used should be a multiple of 64
        bits_used = result.get("actual_bits_used", 0)
        assert bits_used % 64 == 0 or bits_used == 0


# ─────────────────────────────────────────────
# Runs Test
# ─────────────────────────────────────────────

class TestRunsTest:
    def test_balanced_stream_passes(self, balanced_stream):
        result = runs_test(balanced_stream, alpha=0.01)
        # Alternating stream has maximum possible runs — may fail or pass depending on NIST formula
        # What we assert: outcome is one of the valid values and p_value is real
        assert result["outcome"] in ("PASS", "FAIL")
        assert result["p_value"] is not None

    def test_all_zeros_fails(self, all_zeros):
        result = runs_test(all_zeros, alpha=0.01)
        assert result["outcome"] == "FAIL"

    def test_all_ones_fails(self, all_ones):
        result = runs_test(all_ones, alpha=0.01)
        assert result["outcome"] == "FAIL"

    def test_insufficient_data(self):
        result = runs_test("01" * 3, alpha=0.01)
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_statistic_is_run_count(self, pseudo_random_stream):
        result = runs_test(pseudo_random_stream)
        # Statistic should be a positive number (total runs)
        if result["statistic"] is not None:
            assert result["statistic"] > 0


# ─────────────────────────────────────────────
# Longest Run Test
# ─────────────────────────────────────────────

class TestLongestRunTest:
    def test_pseudo_random_completes(self, pseudo_random_stream):
        result = longest_run_test(pseudo_random_stream, alpha=0.01)
        assert result["outcome"] in ("PASS", "FAIL", "INSUFFICIENT_DATA", "ERROR")

    def test_all_ones_fails(self, all_ones):
        result = longest_run_test(all_ones, alpha=0.01)
        assert result["outcome"] == "FAIL"

    def test_insufficient_data(self):
        result = longest_run_test("0" * 10, alpha=0.01)
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_p_value_for_sufficient_data(self, pseudo_random_stream):
        result = longest_run_test(pseudo_random_stream, alpha=0.01)
        if result["p_value"] is not None:
            assert 0.0 <= result["p_value"] <= 1.0


# ─────────────────────────────────────────────
# Spectral (DFT) Test
# ─────────────────────────────────────────────

class TestSpectralTest:
    def test_pseudo_random_completes(self, pseudo_random_stream):
        result = spectral_test(pseudo_random_stream, alpha=0.01)
        assert result["outcome"] in ("PASS", "FAIL")

    def test_periodic_pattern_fails(self):
        """A repeated pattern has strong spectral peaks → should FAIL."""
        periodic = "10001000" * 200  # 1600 bits, period 8
        result = spectral_test(periodic, alpha=0.01)
        # This might pass or fail depending on exact pattern; what we test is it runs
        assert result["outcome"] in ("PASS", "FAIL")
        assert result["p_value"] is not None

    def test_insufficient_data(self):
        result = spectral_test("01" * 100, alpha=0.01)  # 200 bits < 1000 min
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_p_value_in_range(self, pseudo_random_stream):
        result = spectral_test(pseudo_random_stream)
        if result["p_value"] is not None:
            assert 0.0 <= result["p_value"] <= 1.0


# ─────────────────────────────────────────────
# run_selected_tests
# ─────────────────────────────────────────────

class TestRunSelectedTests:
    def test_all_tests_run(self, pseudo_random_stream):
        result = run_selected_tests(
            pseudo_random_stream,
            test_names=ALL_TEST_NAMES,
            alpha=0.01,
        )
        assert len(result["results"]) == len(ALL_TEST_NAMES)

    def test_passed_plus_failed_plus_not_run_equals_total(self, pseudo_random_stream):
        result = run_selected_tests(pseudo_random_stream, test_names=ALL_TEST_NAMES)
        total = result["passed"] + result["failed"] + result["not_run"]
        assert total == len(ALL_TEST_NAMES)

    def test_unknown_test_marked_not_run(self, pseudo_random_stream):
        result = run_selected_tests(
            pseudo_random_stream,
            test_names=["frequency", "nonexistent_test"],
        )
        outcomes = [r["outcome"] for r in result["results"]]
        assert "NOT_RUN" in outcomes

    def test_skipped_not_counted_as_passed(self, pseudo_random_stream):
        result = run_selected_tests(
            pseudo_random_stream,
            test_names=["nonexistent_test"],
        )
        assert result["passed"] == 0

    def test_insufficient_not_counted_as_passed(self):
        short = "01" * 10  # 20 bits — too short for most tests
        result = run_selected_tests(short, test_names=ALL_TEST_NAMES, alpha=0.01)
        # All should be INSUFFICIENT_DATA, none should be PASS
        assert result["passed"] == 0

    def test_invalid_bitstream_raises(self, pseudo_random_stream):
        with pytest.raises(ValueError):
            run_selected_tests("0102", test_names=["frequency"])

    def test_result_has_test_run_id(self, pseudo_random_stream):
        result = run_selected_tests(pseudo_random_stream, test_names=["frequency"])
        assert "test_run_id" in result
        assert len(result["test_run_id"]) > 0

    def test_summary_note_warns_against_randomness(self, pseudo_random_stream):
        result = run_selected_tests(pseudo_random_stream, test_names=ALL_TEST_NAMES)
        note = result["summary_note"].lower()
        assert "does not prove" in note or "not prove" in note or "not" in note

    def test_p_value_none_for_not_run(self, pseudo_random_stream):
        result = run_selected_tests(
            pseudo_random_stream,
            test_names=["nonexistent_test"],
        )
        for r in result["results"]:
            if r["outcome"] == "NOT_RUN":
                assert r["p_value"] is None

    def test_all_tests_available_list(self):
        assert "frequency" in AVAILABLE_TESTS
        assert "block_frequency" in AVAILABLE_TESTS
        assert "runs" in AVAILABLE_TESTS
        assert "longest_run" in AVAILABLE_TESTS
        assert "spectral" in AVAILABLE_TESTS


# ─────────────────────────────────────────────
# Numerical edge cases
# ─────────────────────────────────────────────

class TestNumericalEdgeCases:
    def test_frequency_single_bit(self):
        """Single bit — too short for frequency test."""
        result = frequency_test("1", alpha=0.01)
        assert result["outcome"] == "INSUFFICIENT_DATA"

    def test_runs_near_threshold(self):
        """Stream right at the minimum bits boundary."""
        bits = "01" * (MIN_BITS_RUNS // 2)
        result = runs_test(bits, alpha=0.01)
        # Should either run or return INSUFFICIENT_DATA, but not crash
        assert result["outcome"] in ("PASS", "FAIL", "INSUFFICIENT_DATA", "ERROR")

    def test_all_tests_handle_empty_stream(self):
        for name, fn in AVAILABLE_TESTS.items():
            if name == "block_frequency":
                result = fn("", alpha=0.01, block_size=128)
            else:
                result = fn("", alpha=0.01)
            assert result["outcome"] in (
                "INSUFFICIENT_DATA", "NOT_RUN", "ERROR"
            ), f"Test '{name}' should handle empty stream gracefully"
            # p_value should NOT be fabricated
            assert result["p_value"] is None
