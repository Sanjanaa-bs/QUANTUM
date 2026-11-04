"""
Experiment Benchmarking Service
===============================
Measures computational performance, latency, and throughput across the QRNG pipeline:
  1. Quantum Circuit Generation (1, 2, 4, and 8 qubit configurations)
  2. Von Neumann Debiasing Extractor
  3. NIST SP 800-22 Statistical Testing (5 tests)
  4. PDF Report Generation (ReportLab)

Calculates median, mean, standard deviation, min, max, and IQR across repeated trials.
Computes generation and processing throughput in bits per second.

Scientific Integrity Notice:
  Benchmarks reflect classical CPU execution, Python runtime overhead, and Qiskit C++ simulator
  performance. They do NOT measure quantum physical gate times and do NOT establish any form
  of quantum advantage or quantum supremacy.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.services.debiasing import von_neumann_debias
from backend.app.services.quantum_engine import run_ideal_simulation
from backend.app.services.statistical_tests import run_selected_tests
from backend.app.services.report_service import export_pdf

logger = logging.getLogger(__name__)

BENCHMARK_DISCLAIMER = (
    "Benchmark measurements represent local classical host CPU performance, Python "
    "interpreter execution, and Qiskit Aer simulation overhead. They do NOT reflect "
    "physical quantum coherent gate times or establish quantum advantage. "
    "Hardware queue latency is fundamentally separated from simulator execution."
)


def _compute_trial_statistics(values: List[float], bit_count: int) -> Dict[str, Any]:
    """Compute summary statistics (median, mean, std, min, max, iqr, throughput)."""
    if not values:
        return {
            "median_seconds": 0.0,
            "mean_seconds": 0.0,
            "std_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "iqr_seconds": 0.0,
            "throughput_bits_per_second": 0.0,
            "trials_recorded": 0,
        }

    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)

    throughput = round(bit_count / med, 2) if med > 0 else 0.0

    return {
        "median_seconds": round(med, 5),
        "mean_seconds": round(mean, 5),
        "std_seconds": round(std, 5),
        "min_seconds": round(min_v, 5),
        "max_seconds": round(max_v, 5),
        "iqr_seconds": round(iqr, 5),
        "throughput_bits_per_second": throughput,
        "trials_recorded": len(values),
    }


def run_pipeline_benchmark(
    num_bits: int = 5000,
    trials: int = 5,
    qubit_configs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Execute repeated benchmark trials across the QRNG pipeline.

    Args:
        num_bits: Number of bits per trial (500 to 20,000).
        trials: Number of repeated executions (3 to 10).
        qubit_configs: List of qubit counts to benchmark (default [1, 2, 4, 8]).

    Returns:
        Dictionary with per-operation statistics, qubit scaling benchmarks,
        and throughput metrics.
    """
    num_bits = max(500, min(20000, int(num_bits)))
    trials = max(3, min(10, int(trials)))
    if qubit_configs is None:
        qubit_configs = [1, 2, 4, 8]

    t_start = time.perf_counter()

    # Track trial timings
    gen_times_by_qubit: Dict[int, List[float]] = {q: [] for q in qubit_configs}
    debias_times: List[float] = []
    test_times: List[float] = []
    pdf_times: List[float] = []

    last_bitstream = ""

    for trial_idx in range(trials):
        # 1. Multi-qubit generation benchmarks
        for q in qubit_configs:
            t0 = time.perf_counter()
            sim = run_ideal_simulation(num_bits=num_bits, num_qubits=q, seed=42 + trial_idx)
            elapsed = time.perf_counter() - t0
            gen_times_by_qubit[q].append(elapsed)
            if q == 1:
                last_bitstream = sim["bitstream"]

        # 2. Debiasing benchmark
        t0 = time.perf_counter()
        deb_res = von_neumann_debias(last_bitstream)
        debias_times.append(time.perf_counter() - t0)

        # 3. Statistical tests benchmark (5 tests)
        t0 = time.perf_counter()
        _ = run_selected_tests(last_bitstream, ["frequency", "block_frequency", "runs", "longest_run", "spectral"])
        test_times.append(time.perf_counter() - t0)

        # 4. PDF report benchmark
        t0 = time.perf_counter()
        # Create a lightweight dummy structure for PDF generation
        from unittest.mock import Mock
        from datetime import datetime
        mock_exp = Mock(
            experiment_id="bench-dummy",
            created_at=datetime.utcnow(),
            source_type="ideal_simulator",
            backend_name="AerSimulator",
            requested_bits=num_bits,
            actual_bits=num_bits,
            seed_used=42,
            is_seeded=True,
            circuit_description="Benchmark test circuit",
            noise_params=None,
            execution_time_seconds=0.05,
            status="completed",
            measurement_counts={"0": num_bits // 2, "1": num_bits - (num_bits // 2)},
            bitstream=last_bitstream[:100],
            zeros=num_bits // 2,
            ones=num_bits - (num_bits // 2),
            proportion_ones=0.5,
            shannon_entropy=1.0,
            min_entropy=1.0,
            bias_deviation=0.0,
            num_qubits=1,
            source_classification_note="Benchmark mock",
            warnings=[],
        )
        # Explicitly set spec attributes that use format strings so Mock doesn't intercept __format__
        mock_exp.min_entropy = 1.0
        mock_exp.num_qubits = 1
        _ = export_pdf(mock_exp)
        pdf_times.append(time.perf_counter() - t0)

    total_elapsed = time.perf_counter() - t_start

    # Compile summary per qubit config
    qubit_benchmarks: Dict[str, Any] = {}
    for q in qubit_configs:
        qubit_benchmarks[f"{q}_qubit"] = {
            "num_qubits": q,
            **_compute_trial_statistics(gen_times_by_qubit[q], num_bits),
        }

    pipeline_summary = {
        "generation_1qubit": qubit_benchmarks.get("1_qubit"),
        "debiasing": _compute_trial_statistics(debias_times, num_bits),
        "statistical_testing_5tests": _compute_trial_statistics(test_times, num_bits),
        "pdf_export": _compute_trial_statistics(pdf_times, num_bits),
    }

    return {
        "num_bits_tested": num_bits,
        "trials_executed": trials,
        "total_benchmark_time_seconds": round(total_elapsed, 4),
        "qubit_scaling": qubit_benchmarks,
        "pipeline_stages": pipeline_summary,
        "disclaimer": BENCHMARK_DISCLAIMER,
    }

