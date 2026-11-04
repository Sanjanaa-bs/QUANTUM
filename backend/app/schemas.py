"""
Pydantic schemas for API request/response validation.
All fields are strictly typed and documented.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────


class SourceType(str, Enum):
    IDEAL_SIMULATOR = "ideal_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    HARDWARE = "hardware"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TestOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ERROR = "ERROR"


# ─────────────────────────────────────────────
# Noise Model Configuration
# ─────────────────────────────────────────────


class ReadoutErrorConfig(BaseModel):
    """Configuration for a simple readout-error noise model."""
    prob_meas0_prep1: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of measuring 0 when qubit is in |1⟩ (bit-flip on 1)"
    )
    prob_meas1_prep0: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of measuring 1 when qubit is in |0⟩ (bit-flip on 0)"
    )

    model_config = {"json_schema_extra": {
        "example": {"prob_meas0_prep1": 0.05, "prob_meas1_prep0": 0.05}
    }}


# ─────────────────────────────────────────────
# Generation Request
# ─────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for generating quantum random bits."""
    source_type: SourceType = Field(
        default=SourceType.IDEAL_SIMULATOR,
        description=(
            "Source mode: ideal_simulator uses classical pseudo-random sampling "
            "of a quantum circuit; noisy_simulator adds a configurable readout-error "
            "model; hardware requires IBM Quantum credentials."
        )
    )
    num_bits: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Number of bits to generate (100–100,000)"
    )
    num_qubits: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of qubits in parallel Hadamard register (1, 2, 4, or 8)"
    )
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Optional reproducible seed for simulator experiments. "
            "When set, the run is labeled as seeded."
        )
    )
    noise_config: Optional[ReadoutErrorConfig] = Field(
        default=None,
        description="Readout-error noise configuration (used only for noisy_simulator)"
    )

    @field_validator("noise_config")
    @classmethod
    def noise_only_for_noisy(cls, v, info):
        values = info.data
        if v is not None and values.get("source_type") != SourceType.NOISY_SIMULATOR:
            raise ValueError(
                "noise_config may only be set when source_type is 'noisy_simulator'."
            )
        return v


# ─────────────────────────────────────────────
# Generation Response
# ─────────────────────────────────────────────


class BitstreamStats(BaseModel):
    """Descriptive statistics for a binary sequence."""
    total_bits: int
    zeros: int
    ones: int
    proportion_ones: float = Field(description="Observed fraction of 1s")
    proportion_zeros: float = Field(description="Observed fraction of 0s")
    bias_deviation: float = Field(
        description="Absolute deviation of proportion_ones from 0.5"
    )
    shannon_entropy_per_bit: float = Field(
        description=(
            "Shannon entropy estimate in bits per symbol. "
            "This is a descriptive statistic; it does not certify cryptographic entropy."
        )
    )
    min_entropy_per_bit: float = Field(
        default=0.0,
        description=(
            "Empirical min-entropy estimate H_min = -log2(max(p0, p1)). "
            "Worst-case uncertainty metric governing randomness extraction."
        )
    )


class GenerateResponse(BaseModel):
    """Response from the generate endpoint."""
    experiment_id: str
    source_type: SourceType
    backend_name: str
    requested_bits: int
    actual_bits: int
    num_qubits: int = Field(default=1, description="Number of qubits in circuit")
    shots_executed: Optional[int] = Field(default=None, description="Number of circuit shots executed")
    bitstream_preview: str = Field(
        description="First 200 characters of the bitstream for display"
    )
    bitstream_length: int
    stats: BitstreamStats
    measurement_counts: Dict[str, int] = Field(
        description=(
            "Aggregated measurement counts from the quantum circuit. "
            "Note: aggregated counts do not preserve chronological shot order."
        )
    )
    circuit_description: str
    noise_params: Optional[Dict[str, Any]] = None
    seed_used: Optional[int] = None
    is_seeded: bool
    execution_time_seconds: float
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
    source_classification_note: str = Field(
        description="Explicit note clarifying the source type"
    )


# ─────────────────────────────────────────────
# Debiasing
# ─────────────────────────────────────────────


class DebiasRequest(BaseModel):
    """Request to apply von Neumann debiasing to a saved experiment."""
    use_debiased_input: bool = Field(
        default=False,
        description="If True, attempt to debias an already-debiased stream"
    )


class PairCounts(BaseModel):
    total_complete_pairs: int
    count_00: int
    count_01: int
    count_10: int
    count_11: int
    retained_pairs: int
    discarded_pairs: int
    discarded_percentage: float
    discarded_bits: int = Field(default=0, description="Total discarded bits")


class DebiasResponse(BaseModel):
    experiment_id: str
    debias_id: str
    input_bits: int
    output_bits: int
    pair_counts: PairCounts
    input_stats: BitstreamStats
    output_stats: Optional[BitstreamStats] = None
    output_bitstream_preview: str
    output_to_input_ratio: float = Field(
        default=0.0,
        description="Output bits / Input bits ratio (theoretically ~25% for balanced independent bits)"
    )
    pair_retention_ratio: float = Field(
        default=0.0,
        description="Output bits / Complete pairs ratio (theoretically ~50% for balanced independent bits)"
    )
    discarded_bits: int = Field(default=0, description="Total discarded bits")
    extraction_efficiency: float = Field(
        description="Output bits / Input bits ratio (alias for output_to_input_ratio)"
    )
    algorithm: str = "Von Neumann Extractor"
    algorithm_note: str
    warnings: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────


class TestRequest(BaseModel):
    """Request to run statistical tests."""
    tests: List[str] = Field(
        default=["frequency", "block_frequency", "runs", "longest_run", "spectral"],
        description="List of test names to run"
    )
    use_debiased: bool = Field(
        default=False,
        description="If True, run tests on the debiased output stream"
    )
    block_size: int = Field(
        default=128,
        ge=8,
        le=1000,
        description="Block size for block frequency test"
    )
    significance_level: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Significance level alpha for hypothesis testing"
    )


class SingleTestResult(BaseModel):
    test_name: str
    test_purpose: str
    outcome: TestOutcome
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    significance_level: float
    minimum_bits_required: int
    actual_bits_used: int
    interpretation: str
    warning: Optional[str] = None


class TestRunResponse(BaseModel):
    experiment_id: str
    test_run_id: str
    tested_on: str = Field(description="'original' or 'debiased'")
    bits_analyzed: int
    significance_level: float
    results: List[SingleTestResult]
    passed: int
    failed: int
    not_run: int
    summary_note: str


# ─────────────────────────────────────────────
# Experiment Record
# ─────────────────────────────────────────────


class ExperimentRecord(BaseModel):
    experiment_id: str
    created_at: str
    source_type: str
    backend_name: str
    requested_bits: int
    actual_bits: int
    seed_used: Optional[int]
    is_seeded: bool
    execution_time_seconds: float
    status: str
    has_debiased: bool
    has_test_results: bool
    warnings: List[str] = Field(default_factory=list)


class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentRecord]
    total: int
    page: int
    page_size: int


# ─────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────


class CompareRequest(BaseModel):
    experiment_ids: List[str] = Field(
        min_length=2,
        max_length=5,
        description="2–5 experiment IDs to compare"
    )


class CompareEntry(BaseModel):
    experiment_id: str
    source_type: str
    backend_name: str
    num_qubits: Optional[int] = 1
    actual_bits: int
    proportion_ones: float
    bias_deviation: float
    shannon_entropy: float
    min_entropy: Optional[float] = None
    execution_time_seconds: float
    has_test_results: bool
    nist_passed: Optional[int] = None
    nist_failed: Optional[int] = None


class CompareResponse(BaseModel):
    entries: List[CompareEntry]
    warnings: List[str] = Field(default_factory=list)
    comparison_note: str


# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────


class HardwareJobRequest(BaseModel):
    num_bits: int = Field(ge=100, le=8192, description="Shots for hardware job (max 8192)")
    backend_name: str = Field(description="IBM Quantum backend name")


class HardwareJobStatus(BaseModel):
    job_id: str
    backend_name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result_experiment_id: Optional[str] = None
    error_message: Optional[str] = None


# ─────────────────────────────────────────────
# Noise Impact & Batch Evaluation Schemas
# ─────────────────────────────────────────────


class NoiseSweepRequest(BaseModel):
    noise_levels: List[float] = Field(
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.15],
        description="List of readout-error probabilities to sweep (0.0 to 0.49)"
    )
    num_bits: int = Field(default=5000, ge=500, le=50000, description="Bits per noise point")
    seed: Optional[int] = Field(default=None, description="Optional reproducible base seed")
    significance_level: float = Field(default=0.01, ge=0.001, le=0.1)


class NoiseSweepResponse(BaseModel):
    sweep_type: str
    num_bits_per_point: int
    is_seeded: bool
    base_seed: Optional[int] = None
    significance_level: float
    total_sweep_time_seconds: float
    total_points: int
    noise_levels_tested: List[float]
    points: List[Dict[str, Any]]
    scientific_summary: str


class BatchEvaluationRequest(BaseModel):
    num_sequences: int = Field(default=20, ge=5, le=100, description="Number of independent sequences")
    bits_per_sequence: int = Field(default=2000, ge=100, le=20000, description="Bits per sequence")
    source_type: str = Field(default="ideal_simulator", description="'ideal_simulator' or 'noisy_simulator'")
    num_qubits: int = Field(default=1, ge=1, le=8, description="Qubit register size")
    tests: List[str] = Field(
        default=["frequency", "block_frequency", "runs", "longest_run", "spectral"],
        description="List of NIST test names to run"
    )
    significance_level: float = Field(default=0.01, ge=0.001, le=0.1)
    seed: Optional[int] = Field(default=None, description="Optional base seed")


class BatchEvaluationResponse(BaseModel):
    num_sequences: int
    bits_per_sequence: int
    source_type: str
    num_qubits: int
    significance_level: float
    is_seeded: bool
    base_seed: Optional[int] = None
    total_execution_time_seconds: float
    confidence_interval: Dict[str, Any]
    test_summaries: Dict[str, Any]
    sequences: List[Dict[str, Any]]
    multiple_testing_note: str


class BenchmarkRequest(BaseModel):
    num_bits: int = Field(default=5000, ge=500, le=20000, description="Bit length per trial")
    trials: int = Field(default=5, ge=3, le=10, description="Repetitions for median/std calculation")
    qubit_configs: List[int] = Field(default=[1, 2, 4, 8], description="Qubit counts to evaluate")


class BenchmarkResponse(BaseModel):
    num_bits_tested: int
    trials_executed: int
    total_benchmark_time_seconds: float
    qubit_scaling: Dict[str, Any]
    pipeline_stages: Dict[str, Any]
    disclaimer: str


class EntropyAnalysisRequest(BaseModel):
    window_size: int = Field(default=256, ge=16, le=4096, description="Sliding window size in bits")
    step_size: Optional[int] = Field(default=None, description="Step size (None for non-overlapping)")


class EntropyAnalysisResponse(BaseModel):
    experiment_id: str
    profile: Dict[str, Any]
    window_analysis: Dict[str, Any]
    debiased_comparison: Optional[Dict[str, Any]] = None
    explanations: Dict[str, str]


class CircuitVisualizerResponse(BaseModel):
    num_qubits: int
    total_basis_states: int
    theoretical_prob_per_state: float
    circuit_ascii: str
    stages: List[Dict[str, Any]]
    explanation: str


# ─────────────────────────────────────────────
# Generic responses
# ─────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    hardware_configured: bool
    database_ok: bool


class InfoResponse(BaseModel):
    title: str
    version: str
    description: str
    supported_source_types: List[str]
    hardware_configured: bool
    max_bits: int
    min_bits: int
    default_bits: int
    nist_tests_implemented: List[str]
    disclaimer: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
