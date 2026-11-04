"""
Tests for Feature 1: Quantum Entropy Quality Analyzer
"""

import math
import pytest
from backend.app.services.entropy_analyzer import (
    compute_shannon_entropy,
    compute_min_entropy,
    compute_entropy_profile,
    window_entropy_analysis,
    compare_entropy_profiles,
)


class TestEntropyCalculations:
    def test_ideal_fair_stream_shannon_entropy(self):
        # 50/50 balance -> H = 1.0
        h = compute_shannon_entropy(0.5, 0.5)
        assert h == pytest.approx(1.0, abs=1e-5)

    def test_ideal_fair_stream_min_entropy(self):
        # 50/50 balance -> max(0.5, 0.5)=0.5 -> H_min = -log2(0.5) = 1.0
        h_min = compute_min_entropy(0.5, 0.5)
        assert h_min == pytest.approx(1.0, abs=1e-5)

    def test_completely_biased_stream_min_entropy(self):
        # p0=1.0, p1=0.0 -> max(1.0, 0.0)=1.0 -> H_min = -log2(1.0) = 0.0
        assert compute_min_entropy(1.0, 0.0) == 0.0
        assert compute_min_entropy(0.0, 1.0) == 0.0

    def test_completely_biased_stream_shannon_entropy(self):
        assert compute_shannon_entropy(1.0, 0.0) == 0.0
        assert compute_shannon_entropy(0.0, 1.0) == 0.0

    def test_empty_input_profile(self):
        profile = compute_entropy_profile("")
        assert profile["total_bits"] == 0
        assert profile["shannon_entropy_per_bit"] == 0.0
        assert profile["min_entropy_per_bit"] == 0.0
        assert profile["bias_deviation"] == 0.5

    def test_min_entropy_is_always_less_or_equal_to_shannon(self):
        # By Rényi entropy ordering, H_min <= H_Shannon always
        for p1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            p0 = 1.0 - p1
            h_shannon = compute_shannon_entropy(p0, p1)
            h_min = compute_min_entropy(p0, p1)
            assert h_min <= h_shannon + 1e-6

    def test_profile_on_actual_bitstream(self):
        stream = "01010101" * 100  # 800 bits, 400 zeros, 400 ones
        profile = compute_entropy_profile(stream)
        assert profile["total_bits"] == 800
        assert profile["zeros"] == 400
        assert profile["ones"] == 400
        assert profile["proportion_ones"] == 0.5
        assert profile["bias_deviation"] == 0.0
        assert profile["shannon_entropy_per_bit"] == pytest.approx(1.0, abs=1e-5)
        assert profile["min_entropy_per_bit"] == pytest.approx(1.0, abs=1e-5)


class TestWindowEntropyAnalysis:
    def test_window_analysis_small_stream(self):
        # Stream shorter than window_size
        stream = "0101"
        res = window_entropy_analysis(stream, window_size=64)
        assert res["num_windows"] == 0
        assert "warning" in res

    def test_window_analysis_stationary_stream(self):
        # Perfectly uniform alternating stream across 1024 bits
        stream = "01" * 512
        res = window_entropy_analysis(stream, window_size=128, step_size=128)
        assert res["num_windows"] == 8
        assert res["summary"]["mean_shannon"] == pytest.approx(1.0, abs=1e-3)
        assert res["summary"]["mean_min_entropy"] == pytest.approx(1.0, abs=1e-3)
        assert res["summary"]["std_shannon"] == pytest.approx(0.0, abs=1e-3)

    def test_window_analysis_detects_drift(self):
        # First half is balanced (0101...), second half is biased (111100...)
        first_half = "01" * 256          # 512 bits, p1 = 0.5
        second_half = "1110" * 128       # 512 bits, p1 = 0.75
        drift_stream = first_half + second_half

        res = window_entropy_analysis(drift_stream, window_size=128, step_size=128)
        assert res["num_windows"] == 8
        # First window (balanced)
        assert res["windows"][0]["min_entropy"] > res["windows"][7]["min_entropy"]
        # Standard deviation should be non-zero due to drift
        assert res["summary"]["std_min_entropy"] > 0.05


class TestCompareEntropyProfiles:
    def test_comparison_shows_gain_after_debiasing(self):
        # Biased raw stream: 75% ones
        raw_stream = "1110" * 250   # 1000 bits
        # Debiased stream from 1110: 11 (discard), 10 (output 1) -> all ones?
        # Let's test with a realistic raw vs balanced debiased:
        debiased_stream = "0110" * 100
        comp = compare_entropy_profiles(raw_stream, debiased_stream)
        assert comp["debiased"]["shannon_entropy_per_bit"] > comp["raw"]["shannon_entropy_per_bit"]
        assert comp["shannon_entropy_gain"] > 0.0
        assert comp["min_entropy_gain"] > 0.0

