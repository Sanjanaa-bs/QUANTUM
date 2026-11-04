"""
SQLAlchemy ORM models for the QRNG application.
Uses SQLite for local storage. No secrets are stored here.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """Stores one QRNG experiment (generation run)."""
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    source_type = Column(String, nullable=False)      # SourceType enum value
    backend_name = Column(String, nullable=False)
    requested_bits = Column(Integer, nullable=False)
    actual_bits = Column(Integer, nullable=False)
    seed_used = Column(Integer, nullable=True)
    is_seeded = Column(Boolean, default=False)
    circuit_description = Column(Text, nullable=True)
    noise_params = Column(JSON, nullable=True)         # dict or None
    execution_time_seconds = Column(Float, nullable=True)
    status = Column(String, default="completed")       # ExperimentStatus enum value
    error_message = Column(Text, nullable=True)

    # Measurement data
    measurement_counts = Column(JSON, nullable=True)  # e.g. {"0": 5012, "1": 4988}
    bitstream = Column(Text, nullable=True)            # raw bit string "0101..."

    # Descriptive statistics (denormalized for fast retrieval)
    zeros = Column(Integer, nullable=True)
    ones = Column(Integer, nullable=True)
    proportion_ones = Column(Float, nullable=True)
    shannon_entropy = Column(Float, nullable=True)
    min_entropy = Column(Float, nullable=True)
    bias_deviation = Column(Float, nullable=True)
    num_qubits = Column(Integer, default=1, nullable=True)

    # Source classification
    source_classification_note = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)  # list of strings

    # Hardware job (only populated for real hardware experiments)
    hardware_job_id = Column(String, nullable=True)

    # Foreign fields set after debiasing / test runs
    has_debiased = Column(Boolean, default=False)
    has_test_results = Column(Boolean, default=False)


class DebiasingRecord(Base):
    """Stores the result of von Neumann debiasing applied to an experiment."""
    __tablename__ = "debiasing_records"

    debias_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Pair counts
    input_bits = Column(Integer, nullable=False)
    total_complete_pairs = Column(Integer, nullable=False)
    count_00 = Column(Integer, nullable=False)
    count_01 = Column(Integer, nullable=False)
    count_10 = Column(Integer, nullable=False)
    count_11 = Column(Integer, nullable=False)
    retained_pairs = Column(Integer, nullable=False)
    discarded_pairs = Column(Integer, nullable=False)
    discarded_bits = Column(Integer, nullable=True)

    # Output
    output_bits = Column(Integer, nullable=False)
    output_bitstream = Column(Text, nullable=True)
    output_zeros = Column(Integer, nullable=True)
    output_ones = Column(Integer, nullable=True)
    output_proportion_ones = Column(Float, nullable=True)
    output_shannon_entropy = Column(Float, nullable=True)
    output_min_entropy = Column(Float, nullable=True)
    output_to_input_ratio = Column(Float, nullable=True)
    pair_retention_ratio = Column(Float, nullable=True)
    extraction_efficiency = Column(Float, nullable=True)


class StatisticalTestRun(Base):
    """Stores one batch of statistical test results for an experiment."""
    __tablename__ = "statistical_test_runs"

    test_run_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    tested_on = Column(String, nullable=False)   # "original" or "debiased"
    bits_analyzed = Column(Integer, nullable=False)
    significance_level = Column(Float, nullable=False)
    results_json = Column(JSON, nullable=True)    # list of SingleTestResult dicts
    passed = Column(Integer, nullable=True)
    failed = Column(Integer, nullable=True)
    not_run = Column(Integer, nullable=True)
    implementation_version = Column(String, default="1.0.0")


class HardwareJob(Base):
    """Tracks IBM Quantum hardware job submissions (optional feature)."""
    __tablename__ = "hardware_jobs"

    job_id = Column(String, primary_key=True)
    backend_name = Column(String, nullable=False)
    status = Column(String, default="submitted")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_experiment_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    requested_bits = Column(Integer, nullable=True)
    # NOTE: IBM_QUANTUM_TOKEN is NEVER stored here.
