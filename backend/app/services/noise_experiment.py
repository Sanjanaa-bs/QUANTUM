"""
Noise Impact Experiment Service
===============================
Performs automated noise sweeps using Qiskit Aer simulation with
configurable readout-error probabilities.

For each noise level:
  - Runs actual quantum simulation with asymmetric readout error:
    P(meas 0 | prep 1) = noise_level (energy relaxation T1 decay |1⟩ → |0⟩),
    P(meas 1 | prep 0) = round(noise_level * asymmetry_ratio, 4) (cryogenic thermal excitation |0⟩ → |1⟩).
  - Computes observed bit bias, Shannon entropy, empirical min-entropy.
  - Applies Von Neumann debiasing to the actual generated bitstream.
  - Measures extraction efficiency (output_to_input_ratio & pair_retention_ratio).
  - Evaluates NIST statistical test performance (Frequency & Runs tests).
  - Records execution time and noise parameters.

Scientific Integrity Notice:
  Simulation uses classical pseudo-random sampling with an artificial readout-error model.
  It does not recreate environmental noise, T1/T2 decoherence, or crosstalk of physical QPUs.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from backend.app.services.debiasing import von_neumann_debias
from backend.app.services.quantum_engine import run_ideal_simulation, run_noisy_simulation
from backend.app.services.statistical_tests import frequency_test, runs_test

logger = logging.getLogger(__name__)


def run_noise_sweep(
    noise_levels: Optional[List[float]] = None,
    num_bits: int = 5000,
    seed: Optional[int] = None,
    alpha: float = 0.01,
    asymmetry_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Execute an automated noise sweep across specified noise levels.

    Args:
        noise_levels: List of readout error probabilities (default: [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]).
        num_bits: Number of bits per simulation run (between 500 and 50,000).
        seed: Optional base seed for reproducibility.
        alpha: Significance level for statistical tests.
        asymmetry_ratio: Ratio P(meas 1 | prep 0) / P(meas 0 | prep 1), reflecting physical
            thermal excitation relative to T1 relaxation (default 0.15).

    Returns:
        Structured sweep result containing points for each noise level,
        comparison metrics, and execution metadata.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]

    # Clean and sort noise levels
    clean_levels: List[float] = []
    for lvl in noise_levels:
        val = max(0.0, min(0.49, float(lvl)))
        if val not in clean_levels:
            clean_levels.append(val)
    clean_levels.sort()

    num_bits = max(500, min(50000, int(num_bits)))
    asymmetry_ratio = max(0.0, min(1.0, float(asymmetry_ratio)))
    t_start = time.perf_counter()
    points: List[Dict[str, Any]] = []

    for idx, lvl in enumerate(clean_levels):
        run_seed = (seed + idx * 101) if seed is not None else None
        t0 = time.perf_counter()

        if lvl == 0.0:
            sim_result = run_ideal_simulation(num_bits=num_bits, seed=run_seed)
        else:
            sim_result = run_noisy_simulation(
                num_bits=num_bits,
                prob_meas0_prep1=lvl,
                prob_meas1_prep0=round(lvl * asymmetry_ratio, 4),
                seed=run_seed,
            )

        sim_time = time.perf_counter() - t0
        bitstream = sim_result["bitstream"]
        raw_stats = sim_result["stats"]

        # Apply Von Neumann debiasing to the actual bits
        debias_res = von_neumann_debias(bitstream)

        # Run NIST Frequency and Runs tests on the raw stream
        freq_raw = frequency_test(bitstream, alpha=alpha)
        runs_raw = runs_test(bitstream, alpha=alpha)

        # Run NIST Frequency and Runs tests on the debiased stream (if long enough)
        deb_bits = debias_res["output_bitstream"]
        freq_deb = frequency_test(deb_bits, alpha=alpha) if len(deb_bits) >= 100 else None
        runs_deb = runs_test(deb_bits, alpha=alpha) if len(deb_bits) >= 100 else None

        points.append({
            "noise_level": round(lvl, 4),
            "noise_percentage": round(lvl * 100, 2),
            "execution_time_seconds": round(sim_time, 4),
            "num_bits": num_bits,
            "raw_zeros": raw_stats["zeros"],
            "raw_ones": raw_stats["ones"],
            "raw_proportion_ones": raw_stats["proportion_ones"],
            "raw_bias_deviation": raw_stats["bias_deviation"],
            "raw_shannon_entropy": raw_stats["shannon_entropy_per_bit"],
            "raw_min_entropy": raw_stats["min_entropy_per_bit"],
            # Debiasing metrics
            "debiased_bits": debias_res["output_bits"],
            "discarded_bits": debias_res["discarded_bits"],
            "output_to_input_ratio": debias_res["output_to_input_ratio"],
            "pair_retention_ratio": debias_res["pair_retention_ratio"],
            "extraction_efficiency": debias_res["extraction_efficiency"],
            "debiased_proportion_ones": debias_res["output_stats"]["proportion_ones"] if debias_res["output_stats"] else 0.0,
            "debiased_bias_deviation": debias_res["output_stats"]["bias_deviation"] if debias_res["output_stats"] else 0.5,
            "debiased_shannon_entropy": debias_res["output_stats"]["shannon_entropy_per_bit"] if debias_res["output_stats"] else 0.0,
            "debiased_min_entropy": debias_res["output_stats"]["min_entropy_per_bit"] if debias_res["output_stats"] else 0.0,
            # Pair counts
            "pair_counts": {
                "00": debias_res["count_00"],
                "01": debias_res["count_01"],
                "10": debias_res["count_10"],
                "11": debias_res["count_11"],
            },
            # Statistical test outcomes
            "raw_tests": {
                "frequency": {
                    "outcome": freq_raw["outcome"],
                    "p_value": freq_raw["p_value"],
                    "statistic": freq_raw["statistic"],
                },
                "runs": {
                    "outcome": runs_raw["outcome"],
                    "p_value": runs_raw["p_value"],
                    "statistic": runs_raw["statistic"],
                },
            },
            "debiased_tests": {
                "frequency": {
                    "outcome": freq_deb["outcome"] if freq_deb else "INSUFFICIENT_DATA",
                    "p_value": freq_deb["p_value"] if freq_deb else None,
                },
                "runs": {
                    "outcome": runs_deb["outcome"] if runs_deb else "INSUFFICIENT_DATA",
                    "p_value": runs_deb["p_value"] if runs_deb else None,
                },
            },
        })

    total_time = time.perf_counter() - t_start

    return {
        "sweep_type": "readout_error_sweep",
        "num_bits_per_point": num_bits,
        "is_seeded": seed is not None,
        "base_seed": seed,
        "significance_level": alpha,
        "total_sweep_time_seconds": round(total_time, 4),
        "total_points": len(points),
        "noise_levels_tested": clean_levels,
        "points": points,
        "scientific_summary": (
            "Higher readout error increases bit bias away from 0.5 due to physical asymmetric qubit relaxation "
            "(P(0|1) > P(1|0)), reducing both Shannon entropy and min-entropy in the raw stream. While Von Neumann "
            "debiasing successfully restores the output proportion to ~0.5 and entropy to ~1.0, it suffers a steep "
            "decline in extraction efficiency as more pairs match (predominantly 00) and must be discarded."
        ),
    }

