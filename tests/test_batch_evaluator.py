"""
Tests for Feature 4: Batch Statistical Evaluation
"""

import pytest
from backend.app.services.batch_evaluator import run_batch_evaluation


class TestBatchEvaluation:
    def test_batch_runs_requested_sequences(self):
        res = run_batch_evaluation(
            num_sequences=6,
            bits_per_sequence=1000,
            source_type="ideal_simulator",
            tests=["frequency", "runs"],
            seed=42,
        )
        assert res["num_sequences"] == 6
        assert len(res["sequences"]) == 6
        assert "confidence_interval" in res
        assert "frequency" in res["test_summaries"]
        assert "runs" in res["test_summaries"]

    def test_batch_rejection_rates_and_ci(self):
        res = run_batch_evaluation(
            num_sequences=8,
            bits_per_sequence=1000,
            tests=["frequency"],
            alpha=0.01,
            seed=100,
        )
        freq_summary = res["test_summaries"]["frequency"]
        assert freq_summary["evaluated_sequences"] == 8
        assert freq_summary["observed_pass_rate"] + freq_summary["observed_rejection_rate"] == pytest.approx(1.0, abs=1e-5)
        # Check confidence interval structure
        ci = res["confidence_interval"]
        assert ci["lower_bound"] <= ci["upper_bound"]
        assert ci["expected_pass_rate"] == 0.99

    def test_batch_p_value_bins_sum_to_evaluated(self):
        res = run_batch_evaluation(
            num_sequences=10,
            bits_per_sequence=1000,
            tests=["frequency"],
            seed=7,
        )
        bins = res["test_summaries"]["frequency"]["p_value_histogram_bins"]
        assert len(bins) == 10
        assert sum(bins) == res["test_summaries"]["frequency"]["evaluated_sequences"]

    def test_batch_bounds_clamping(self):
        # Sequences below 5 are clamped to 5
        res = run_batch_evaluation(num_sequences=2, bits_per_sequence=50)
        assert res["num_sequences"] == 5
        assert res["bits_per_sequence"] == 100

