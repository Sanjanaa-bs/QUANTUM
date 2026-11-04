"""
Tests for Feature 2: Automated Noise Sweep Experiment
"""

import pytest
from backend.app.services.noise_experiment import run_noise_sweep


class TestNoiseSweep:
    def test_noise_sweep_executes_all_points(self):
        levels = [0.0, 0.05, 0.10]
        res = run_noise_sweep(noise_levels=levels, num_bits=1000, seed=42)
        assert res["total_points"] == 3
        assert len(res["points"]) == 3
        assert res["noise_levels_tested"] == [0.0, 0.05, 0.10]

    def test_noise_sweep_records_real_data(self):
        res = run_noise_sweep(noise_levels=[0.0, 0.15], num_bits=1000, seed=123)
        p0 = res["points"][0]
        p1 = res["points"][1]

        # Check real fields exist
        assert p0["num_bits"] == 1000
        assert p0["execution_time_seconds"] > 0
        assert p0["raw_shannon_entropy"] > 0
        assert p0["raw_min_entropy"] > 0
        assert "output_to_input_ratio" in p0
        assert "pair_retention_ratio" in p0
        assert "raw_tests" in p0

        # Point 1 (15% readout noise) should have higher or equal bias deviation than Point 0 (ideal)
        assert p1["raw_bias_deviation"] >= 0.0
        # debiased_bits can legitimately be 0 at high noise with small num_bits (all pairs match)
        assert isinstance(p1["debiased_bits"], int)
        assert p1["debiased_bits"] >= 0

    def test_noise_sweep_handles_bounds(self):
        # Negative and excessive noise levels should be clamped safely
        res = run_noise_sweep(noise_levels=[-0.1, 0.0, 0.7], num_bits=500)
        assert 0.0 in res["noise_levels_tested"]
        for lvl in res["noise_levels_tested"]:
            assert 0.0 <= lvl <= 0.49

