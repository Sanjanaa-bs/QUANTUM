"""
Batch Statistical Evaluation Service
====================================
Generates multiple independent quantum simulator sequences and evaluates
their empirical p-value distributions against the theoretical uniform distribution
stipulated by NIST SP 800-22 Section 4.2.

Features:
  - Independent bitstream generation across K sequences.
  - Per-sequence evaluation of selected NIST SP 800-22 tests.
  - Calculation of observed rejection proportions (alpha-hat).
  - Confidence interval computation for passing proportions: (1-alpha) ± 3*sqrt(alpha*(1-alpha)/K).
  - p-value uniformity histogram and chi-square goodness-of-fit test.
  - Bounded concurrency and limits to prevent resource exhaustion.
  - Scientific explanation of multiple testing considerations.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import special as sp_special

from backend.app.services.quantum_engine import run_ideal_simulation, run_noisy_simulation
from backend.app.services.statistical_tests import (
    ALL_TEST_NAMES,
    run_selected_tests,
)

logger = logging.getLogger(__name__)

MULTIPLE_TESTING_NOTE = (
    "In statistical hypothesis testing, rejecting H0 with p < α is expected to occur "
    "by random chance with frequency α even when the generator is completely ideal. "
    "When running M tests across K sequences, the family-wise error rate is approximately "
    "1 - (1-α)^(M*K). Therefore, isolated test failures do NOT indicate non-randomness; "
    "evaluating the overall rejection rate and p-value uniformity across a large batch is "
    "the authoritative approach."
)


def run_batch_evaluation(
    num_sequences: int = 20,
    bits_per_sequence: int = 2000,
    source_type: str = "ideal_simulator",
    tests: Optional[List[str]] = None,
    alpha: float = 0.01,
    seed: Optional[int] = None,
    num_qubits: int = 1,
) -> Dict[str, Any]:
    """
    Run batch evaluation over multiple independent sequences.

    Args:
        num_sequences: Number of independent bitstreams to generate (5 to 100).
        bits_per_sequence: Length of each stream in bits (100 to 20,000).
        source_type: "ideal_simulator" or "noisy_simulator".
        tests: List of NIST test names to run.
        alpha: Significance level (default 0.01).
        seed: Optional base seed.
        num_qubits: Qubit register size.

    Returns:
        Structured batch result including sequence logs, rejection proportions,
        p-value distribution histograms, and uniformity tests.
    """
    # Safety limits
    num_sequences = max(5, min(100, int(num_sequences)))
    bits_per_sequence = max(100, min(20000, int(bits_per_sequence)))
    if tests is None or len(tests) == 0:
        tests = ["frequency", "block_frequency", "runs", "longest_run", "spectral"]

    t_start = time.perf_counter()
    sequence_results: List[Dict[str, Any]] = []

    # Track p-values and outcomes per test
    test_p_values: Dict[str, List[float]] = {t: [] for t in tests}
    test_outcomes: Dict[str, Dict[str, int]] = {
        t: {"PASS": 0, "FAIL": 0, "INSUFFICIENT_DATA": 0, "NOT_RUN": 0}
        for t in tests
    }

    for seq_idx in range(num_sequences):
        seq_seed = (seed + seq_idx * 1009) if seed is not None else None

        if source_type == "noisy_simulator":
            sim = run_noisy_simulation(
                num_bits=bits_per_sequence,
                num_qubits=num_qubits,
                prob_meas0_prep1=0.08,
                prob_meas1_prep0=0.01,
                seed=seq_seed,
            )
        else:
            sim = run_ideal_simulation(
                num_bits=bits_per_sequence,
                num_qubits=num_qubits,
                seed=seq_seed,
            )

        bitstream = sim["bitstream"]
        test_run = run_selected_tests(bitstream, tests, alpha=alpha)

        seq_summary: Dict[str, Any] = {
            "sequence_index": seq_idx,
            "bits": len(bitstream),
            "proportion_ones": sim["stats"]["proportion_ones"],
            "shannon_entropy": sim["stats"]["shannon_entropy_per_bit"],
            "min_entropy": sim["stats"]["min_entropy_per_bit"],
            "passed": test_run["passed"],
            "failed": test_run["failed"],
            "not_run": test_run["not_run"],
            "tests": {},
        }

        for r in test_run["results"]:
            t_name = r["test_name"]
            # Find matching key in tests
            matched_key = None
            for k in tests:
                if k.lower() in t_name.lower().replace(" ", "_"):
                    matched_key = k
                    break
            if not matched_key:
                matched_key = t_name

            if matched_key not in test_p_values:
                test_p_values[matched_key] = []
                test_outcomes[matched_key] = {"PASS": 0, "FAIL": 0, "INSUFFICIENT_DATA": 0, "NOT_RUN": 0}

            outcome = r["outcome"]
            if outcome in test_outcomes[matched_key]:
                test_outcomes[matched_key][outcome] += 1

            if r.get("p_value") is not None and outcome in ("PASS", "FAIL"):
                test_p_values[matched_key].append(float(r["p_value"]))

            seq_summary["tests"][t_name] = {
                "outcome": outcome,
                "p_value": r.get("p_value"),
                "statistic": r.get("statistic"),
            }

        sequence_results.append(seq_summary)

    total_time = time.perf_counter() - t_start

    # NIST SP 800-22 Section 4.2.1: Pass proportion confidence interval
    # (1 - alpha) ± 3 * sqrt(alpha * (1 - alpha) / K)
    expected_pass_rate = 1.0 - alpha
    margin = 3.0 * math.sqrt((alpha * (1.0 - alpha)) / num_sequences)
    ci_lower = max(0.0, round(expected_pass_rate - margin, 4))
    ci_upper = min(1.0, round(expected_pass_rate + margin, 4))

    # Compile aggregated stats per test
    test_summaries: Dict[str, Any] = {}
    for t_key in test_p_values.keys():
        outcomes = test_outcomes[t_key]
        p_vals = test_p_values[t_key]
        n_eval = outcomes["PASS"] + outcomes["FAIL"]
        rejection_rate = round(outcomes["FAIL"] / n_eval, 4) if n_eval > 0 else 0.0
        pass_rate = round(outcomes["PASS"] / n_eval, 4) if n_eval > 0 else 0.0

        # p-value histogram (10 equal bins in [0, 1])
        bins = [0] * 10
        for pv in p_vals:
            b_idx = min(9, int(pv * 10))
            bins[b_idx] += 1

        # Goodness-of-fit uniformity chi-square test (NIST Section 4.2.2)
        if len(p_vals) >= 10:
            expected_per_bin = len(p_vals) / 10.0
            chi2_stat = sum(((b - expected_per_bin) ** 2) / expected_per_bin for b in bins)
            # 9 degrees of freedom
            p_uniformity = float(sp_special.gammaincc(9.0 / 2.0, chi2_stat / 2.0))
            uniform = p_uniformity >= 0.0001
        else:
            chi2_stat = None
            p_uniformity = None
            uniform = None

        test_summaries[t_key] = {
            "evaluated_sequences": n_eval,
            "pass_count": outcomes["PASS"],
            "fail_count": outcomes["FAIL"],
            "insufficient_count": outcomes["INSUFFICIENT_DATA"],
            "observed_pass_rate": pass_rate,
            "observed_rejection_rate": rejection_rate,
            "within_confidence_interval": (ci_lower <= pass_rate <= ci_upper) if n_eval > 0 else False,
            "p_value_histogram_bins": bins,
            "uniformity_chi2_statistic": round(chi2_stat, 4) if chi2_stat is not None else None,
            "uniformity_p_value": round(p_uniformity, 6) if p_uniformity is not None else None,
            "p_values_uniform": uniform,
        }

    return {
        "num_sequences": num_sequences,
        "bits_per_sequence": bits_per_sequence,
        "source_type": source_type,
        "num_qubits": num_qubits,
        "significance_level": alpha,
        "is_seeded": seed is not None,
        "base_seed": seed,
        "total_execution_time_seconds": round(total_time, 4),
        "confidence_interval": {
            "expected_pass_rate": round(expected_pass_rate, 4),
            "lower_bound": ci_lower,
            "upper_bound": ci_upper,
            "confidence_level": "99.7% (3-sigma)",
        },
        "test_summaries": test_summaries,
        "sequences": sequence_results,
        "multiple_testing_note": MULTIPLE_TESTING_NOTE,
    }

