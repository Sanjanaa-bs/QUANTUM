"""
Tests for Feature 7: Experiment Benchmarking
"""

import pytest
from backend.app.services.benchmarker import run_pipeline_benchmark, _compute_trial_statistics


class TestBenchmarking:
    def test_trial_statistics_computation(self):
        timings = [0.10, 0.12, 0.11, 0.09, 0.15]
        stats = _compute_trial_statistics(timings, bit_count=1000)
        assert stats["trials_recorded"] == 5
        assert stats["median_seconds"] == pytest.approx(0.11, abs=1e-3)
        assert stats["min_seconds"] == pytest.approx(0.09, abs=1e-3)
        assert stats["max_seconds"] == pytest.approx(0.15, abs=1e-3)
        assert stats["throughput_bits_per_second"] > 0

    def test_pipeline_benchmark_execution(self):
        res = run_pipeline_benchmark(num_bits=1000, trials=3, qubit_configs=[1, 2])
        assert res["num_bits_tested"] == 1000
        assert res["trials_executed"] == 3
        assert "1_qubit" in res["qubit_scaling"]
        assert "2_qubit" in res["qubit_scaling"]
        assert "debiasing" in res["pipeline_stages"]
        assert "statistical_testing_5tests" in res["pipeline_stages"]
        assert "pdf_export" in res["pipeline_stages"]
        assert "disclaimer" in res

