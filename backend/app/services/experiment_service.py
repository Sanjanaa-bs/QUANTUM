"""
Experiment Service
==================
Manages CRUD operations for experiment records in the SQLite database.
All database operations use parameterized ORM queries (no raw SQL).
No secrets are stored in the database.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.database.models import (
    Experiment, DebiasingRecord, StatisticalTestRun, HardwareJob
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Experiment CRUD
# ─────────────────────────────────────────────

def save_experiment(db: Session, result: Dict[str, Any]) -> str:
    """
    Persist a generation result to the experiments table.
    Returns the experiment_id.
    """
    stats = result.get("stats", {})
    exp = Experiment(
        experiment_id=result["experiment_id"],
        created_at=datetime.utcnow(),
        source_type=result["source_type"],
        backend_name=result["backend_name"],
        requested_bits=result["requested_bits"],
        actual_bits=result["actual_bits"],
        seed_used=result.get("seed_used"),
        is_seeded=result.get("is_seeded", False),
        circuit_description=result.get("circuit_description"),
        noise_params=result.get("noise_params"),
        execution_time_seconds=result.get("execution_time_seconds"),
        status="completed",
        measurement_counts=result.get("measurement_counts"),
        bitstream=result.get("bitstream"),
        zeros=stats.get("zeros"),
        ones=stats.get("ones"),
        proportion_ones=stats.get("proportion_ones"),
        shannon_entropy=stats.get("shannon_entropy_per_bit"),
        min_entropy=stats.get("min_entropy_per_bit"),
        bias_deviation=stats.get("bias_deviation"),
        num_qubits=result.get("num_qubits", 1),
        source_classification_note=result.get("source_classification_note"),
        warnings=result.get("warnings", []),
        has_debiased=False,
        has_test_results=False,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    logger.info("Saved experiment %s (%s)", exp.experiment_id, exp.source_type)
    return exp.experiment_id


def get_experiment(db: Session, experiment_id: str) -> Optional[Experiment]:
    """Retrieve a single experiment by ID."""
    return db.query(Experiment).filter(
        Experiment.experiment_id == experiment_id
    ).first()


def list_experiments(
    db: Session, page: int = 1, page_size: int = 20
) -> Dict[str, Any]:
    """Paginated list of experiments, newest first."""
    offset = (page - 1) * page_size
    total = db.query(Experiment).count()
    records = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {"experiments": records, "total": total, "page": page, "page_size": page_size}


def delete_experiment(db: Session, experiment_id: str) -> bool:
    """Delete an experiment and its associated records."""
    exp = get_experiment(db, experiment_id)
    if not exp:
        return False
    db.query(DebiasingRecord).filter(
        DebiasingRecord.experiment_id == experiment_id
    ).delete()
    db.query(StatisticalTestRun).filter(
        StatisticalTestRun.experiment_id == experiment_id
    ).delete()
    db.delete(exp)
    db.commit()
    return True


# ─────────────────────────────────────────────
# Debiasing CRUD
# ─────────────────────────────────────────────

def save_debias_result(
    db: Session, experiment_id: str, debias_result: Dict[str, Any]
) -> str:
    """Persist debiasing result and mark parent experiment."""
    record = DebiasingRecord(
        debias_id=debias_result["debias_id"],
        experiment_id=experiment_id,
        created_at=datetime.utcnow(),
        input_bits=debias_result["input_bits"],
        total_complete_pairs=debias_result["total_complete_pairs"],
        count_00=debias_result["count_00"],
        count_01=debias_result["count_01"],
        count_10=debias_result["count_10"],
        count_11=debias_result["count_11"],
        retained_pairs=debias_result["retained_pairs"],
        discarded_pairs=debias_result["discarded_pairs"],
        discarded_bits=debias_result.get("discarded_bits"),
        output_bits=debias_result["output_bits"],
        output_bitstream=debias_result["output_bitstream"],
        output_zeros=debias_result.get("output_stats", {}).get("zeros"),
        output_ones=debias_result.get("output_stats", {}).get("ones"),
        output_proportion_ones=debias_result.get("output_stats", {}).get("proportion_ones"),
        output_shannon_entropy=debias_result.get("output_stats", {}).get("shannon_entropy_per_bit"),
        output_min_entropy=debias_result.get("output_stats", {}).get("min_entropy_per_bit"),
        output_to_input_ratio=debias_result.get("output_to_input_ratio"),
        pair_retention_ratio=debias_result.get("pair_retention_ratio"),
        extraction_efficiency=debias_result["extraction_efficiency"],
    )
    db.add(record)
    # Mark parent
    exp = get_experiment(db, experiment_id)
    if exp:
        exp.has_debiased = True
    db.commit()
    return record.debias_id


def get_debias_result(db: Session, experiment_id: str) -> Optional[DebiasingRecord]:
    """Get the most recent debiasing record for an experiment."""
    return (
        db.query(DebiasingRecord)
        .filter(DebiasingRecord.experiment_id == experiment_id)
        .order_by(DebiasingRecord.created_at.desc())
        .first()
    )


# ─────────────────────────────────────────────
# Statistical Tests CRUD
# ─────────────────────────────────────────────

def save_test_run(
    db: Session,
    experiment_id: str,
    test_run_result: Dict[str, Any],
    tested_on: str,
) -> str:
    """Persist a test run and mark parent experiment."""
    run = StatisticalTestRun(
        test_run_id=test_run_result["test_run_id"],
        experiment_id=experiment_id,
        created_at=datetime.utcnow(),
        tested_on=tested_on,
        bits_analyzed=test_run_result["bits_analyzed"],
        significance_level=test_run_result["significance_level"],
        results_json=test_run_result["results"],
        passed=test_run_result["passed"],
        failed=test_run_result["failed"],
        not_run=test_run_result["not_run"],
    )
    db.add(run)
    exp = get_experiment(db, experiment_id)
    if exp:
        exp.has_test_results = True
    db.commit()
    return run.test_run_id


def get_test_runs(db: Session, experiment_id: str) -> List[StatisticalTestRun]:
    """Get all test runs for an experiment."""
    return (
        db.query(StatisticalTestRun)
        .filter(StatisticalTestRun.experiment_id == experiment_id)
        .order_by(StatisticalTestRun.created_at.desc())
        .all()
    )


# ─────────────────────────────────────────────
# Hardware Job CRUD
# ─────────────────────────────────────────────

def save_hardware_job(db: Session, job_data: Dict[str, Any]) -> str:
    """Persist a submitted hardware job record."""
    job = HardwareJob(
        job_id=job_data["job_id"],
        backend_name=job_data["backend_name"],
        status=job_data.get("status", "submitted"),
        created_at=datetime.utcnow(),
        requested_bits=job_data.get("requested_bits"),
        # NOTE: IBM_QUANTUM_TOKEN is NEVER stored here.
    )
    db.add(job)
    db.commit()
    return job.job_id


def update_hardware_job(
    db: Session, job_id: str, status: str,
    result_experiment_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update a hardware job's status."""
    job = db.query(HardwareJob).filter(HardwareJob.job_id == job_id).first()
    if job:
        job.status = status
        if result_experiment_id:
            job.result_experiment_id = result_experiment_id
        if error_message:
            job.error_message = error_message
        if status in ("DONE", "ERROR", "CANCELLED"):
            job.completed_at = datetime.utcnow()
        db.commit()


def get_hardware_job(db: Session, job_id: str) -> Optional[HardwareJob]:
    """Get a hardware job record."""
    return db.query(HardwareJob).filter(HardwareJob.job_id == job_id).first()
