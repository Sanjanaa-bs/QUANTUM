"""
FastAPI REST API routes for the QRNG application.
All endpoints use Pydantic validation and return consistent HTTP status codes.
No secrets are returned in any response.
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.session import get_db
from backend.app.schemas import (
    GenerateRequest, GenerateResponse, BitstreamStats,
    DebiasRequest, DebiasResponse, PairCounts,
    TestRequest, TestRunResponse, SingleTestResult,
    ExperimentRecord, ExperimentListResponse,
    CompareRequest, CompareResponse, CompareEntry,
    HardwareJobRequest, HardwareJobStatus,
    HealthResponse, InfoResponse, ErrorResponse,
    SourceType, ExperimentStatus, TestOutcome,
    NoiseSweepRequest, NoiseSweepResponse,
    BatchEvaluationRequest, BatchEvaluationResponse,
    BenchmarkRequest, BenchmarkResponse,
    EntropyAnalysisRequest, EntropyAnalysisResponse,
    CircuitVisualizerResponse,
)
from backend.app.services import (
    quantum_engine, hardware_engine, debiasing, statistical_tests,
    entropy_analyzer, noise_experiment, batch_evaluator, benchmarker
)
from backend.app.services import experiment_service as exp_svc
from backend.app.services import report_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────
# Health & Info
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Application health check. Does NOT submit jobs or reveal secrets."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=settings.APP_VERSION,
        hardware_configured=settings.hardware_configured,
        database_ok=db_ok,
    )


@router.get("/info", response_model=InfoResponse, tags=["System"])
def get_info():
    """Return supported features and source modes."""
    return InfoResponse(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        supported_source_types=[st.value for st in SourceType],
        hardware_configured=settings.hardware_configured,
        max_bits=settings.MAX_BITS,
        min_bits=settings.MIN_BITS,
        default_bits=settings.DEFAULT_BITS,
        nist_tests_implemented=statistical_tests.ALL_TEST_NAMES,
        disclaimer=(
            "Simulator modes use classical computation. "
            "Passing NIST tests does NOT prove randomness or cryptographic security. "
            "This application is NOT certified as a cryptographically secure RNG."
        ),
    )


# ─────────────────────────────────────────────
# Experiments — Generate
# ─────────────────────────────────────────────

@router.post(
    "/experiments/generate",
    response_model=GenerateResponse,
    status_code=201,
    tags=["Experiments"],
)
def generate_bits(body: GenerateRequest, db: Session = Depends(get_db)):
    """
    Run a quantum bit generation experiment (simulator or hardware).
    For hardware, use POST /hardware/jobs instead.
    """
    if body.source_type == SourceType.HARDWARE:
        raise HTTPException(
            status_code=400,
            detail=(
                "Hardware experiments must be submitted via POST /api/v1/hardware/jobs. "
                "This endpoint is for simulator experiments only."
            ),
        )

    if not quantum_engine.is_available():
        raise HTTPException(
            status_code=503,
            detail="Qiskit / Qiskit-Aer is not installed. Cannot run simulation.",
        )

    try:
        if body.source_type == SourceType.IDEAL_SIMULATOR:
            result = quantum_engine.run_ideal_simulation(
                num_bits=body.num_bits,
                num_qubits=body.num_qubits,
                seed=body.seed,
            )
        else:  # NOISY_SIMULATOR
            nc = body.noise_config or {}
            p01 = nc.prob_meas0_prep1 if hasattr(nc, "prob_meas0_prep1") else 0.05
            p10 = nc.prob_meas1_prep0 if hasattr(nc, "prob_meas1_prep0") else 0.05
            result = quantum_engine.run_noisy_simulation(
                num_bits=body.num_bits,
                num_qubits=body.num_qubits,
                prob_meas0_prep1=p01,
                prob_meas1_prep0=p10,
                seed=body.seed,
            )
    except Exception as exc:
        logger.exception("Simulation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    # Persist
    try:
        exp_svc.save_experiment(db, result)
    except Exception as exc:
        logger.exception("Failed to persist experiment: %s", exc)
        # Return result even if persistence failed
        result["warnings"].append("Warning: experiment could not be saved to database.")

    stats = result["stats"]
    return GenerateResponse(
        experiment_id=result["experiment_id"],
        source_type=SourceType(result["source_type"]),
        backend_name=result["backend_name"],
        requested_bits=result["requested_bits"],
        actual_bits=result["actual_bits"],
        num_qubits=result.get("num_qubits", 1),
        shots_executed=result.get("shots_executed", body.num_bits),
        bitstream_preview=result["bitstream"][:200],
        bitstream_length=len(result["bitstream"]),
        stats=BitstreamStats(**stats),
        measurement_counts=result["measurement_counts"],
        circuit_description=result["circuit_description"],
        noise_params=result.get("noise_params"),
        seed_used=result.get("seed_used"),
        is_seeded=result.get("is_seeded", False),
        execution_time_seconds=result["execution_time_seconds"],
        timestamp=result.get("timestamp", __import__("datetime").datetime.utcnow().isoformat()),
        warnings=result.get("warnings", []),
        source_classification_note=result["source_classification_note"],
    )


# ─────────────────────────────────────────────
# Experiments — CRUD
# ─────────────────────────────────────────────

@router.get(
    "/experiments",
    response_model=ExperimentListResponse,
    tags=["Experiments"],
)
def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List all saved experiments with pagination, newest first."""
    data = exp_svc.list_experiments(db, page=page, page_size=page_size)
    records = [
        ExperimentRecord(
            experiment_id=e.experiment_id,
            created_at=e.created_at.isoformat() if e.created_at else "",
            source_type=e.source_type,
            backend_name=e.backend_name,
            requested_bits=e.requested_bits,
            actual_bits=e.actual_bits,
            seed_used=e.seed_used,
            is_seeded=e.is_seeded,
            execution_time_seconds=e.execution_time_seconds or 0.0,
            status=e.status,
            has_debiased=e.has_debiased,
            has_test_results=e.has_test_results,
            warnings=e.warnings or [],
        )
        for e in data["experiments"]
    ]
    return ExperimentListResponse(
        experiments=records,
        total=data["total"],
        page=data["page"],
        page_size=data["page_size"],
    )


@router.get(
    "/experiments/{experiment_id}",
    tags=["Experiments"],
)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieve full experiment details including bitstream preview."""
    exp = exp_svc.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")
    return {
        "experiment_id": exp.experiment_id,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "source_type": exp.source_type,
        "backend_name": exp.backend_name,
        "requested_bits": exp.requested_bits,
        "actual_bits": exp.actual_bits,
        "seed_used": exp.seed_used,
        "is_seeded": exp.is_seeded,
        "circuit_description": exp.circuit_description,
        "noise_params": exp.noise_params,
        "execution_time_seconds": exp.execution_time_seconds,
        "status": exp.status,
        "measurement_counts": exp.measurement_counts,
        "bitstream": exp.bitstream,
        "bitstream_preview": (exp.bitstream or "")[:200],
        "bitstream_length": len(exp.bitstream) if exp.bitstream else 0,
        "zeros": exp.zeros,
        "ones": exp.ones,
        "proportion_ones": exp.proportion_ones,
        "shannon_entropy": exp.shannon_entropy,
        "min_entropy": getattr(exp, "min_entropy", None),
        "num_qubits": getattr(exp, "num_qubits", 1),
        "bias_deviation": exp.bias_deviation,
        "source_classification_note": exp.source_classification_note,
        "warnings": exp.warnings or [],
        "has_debiased": exp.has_debiased,
        "has_test_results": exp.has_test_results,
        "hardware_job_id": exp.hardware_job_id,
    }


# ─────────────────────────────────────────────
# Debiasing
# ─────────────────────────────────────────────

@router.post(
    "/experiments/{experiment_id}/debias",
    response_model=DebiasResponse,
    status_code=201,
    tags=["Debiasing"],
)
def debias_experiment(
    experiment_id: str,
    body: DebiasRequest,
    db: Session = Depends(get_db),
):
    """Apply von Neumann debiasing to a saved experiment's bitstream."""
    exp = exp_svc.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")
    if not exp.bitstream:
        raise HTTPException(
            status_code=422, detail="Experiment has no stored bitstream to debias."
        )

    try:
        result = debiasing.von_neumann_debias(exp.bitstream)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Debiasing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Debiasing error: {exc}")

    # Persist
    debias_id = exp_svc.save_debias_result(db, experiment_id, result)

    pc = PairCounts(
        total_complete_pairs=result["total_complete_pairs"],
        count_00=result["count_00"],
        count_01=result["count_01"],
        count_10=result["count_10"],
        count_11=result["count_11"],
        retained_pairs=result["retained_pairs"],
        discarded_pairs=result["discarded_pairs"],
        discarded_percentage=result["discarded_percentage"],
        discarded_bits=result.get("discarded_bits", 0),
    )

    from backend.app.schemas import BitstreamStats as BS
    input_stats = BS(**result["input_stats"])
    output_stats = None
    if result["output_stats"] and result["output_bits"] > 0:
        output_stats = BS(**result["output_stats"])

    return DebiasResponse(
        experiment_id=experiment_id,
        debias_id=result["debias_id"],
        input_bits=result["input_bits"],
        output_bits=result["output_bits"],
        pair_counts=pc,
        input_stats=input_stats,
        output_stats=output_stats,
        output_bitstream_preview=(result["output_bitstream"] or "")[:200],
        output_to_input_ratio=result.get("output_to_input_ratio", result["extraction_efficiency"]),
        pair_retention_ratio=result.get("pair_retention_ratio", 0.0),
        discarded_bits=result.get("discarded_bits", 0),
        extraction_efficiency=result["extraction_efficiency"],
        algorithm_note=debiasing.ALGORITHM_NOTE,
        warnings=result["warnings"],
    )


# ─────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────

@router.post(
    "/experiments/{experiment_id}/tests",
    response_model=TestRunResponse,
    status_code=201,
    tags=["Statistical Tests"],
)
def run_tests(
    experiment_id: str,
    body: TestRequest,
    db: Session = Depends(get_db),
):
    """Run selected NIST SP 800-22 statistical tests on a saved experiment."""
    exp = exp_svc.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")

    if body.use_debiased:
        debias_rec = exp_svc.get_debias_result(db, experiment_id)
        if not debias_rec or not debias_rec.output_bitstream:
            raise HTTPException(
                status_code=422,
                detail="No debiased bitstream available. Run debiasing first."
            )
        bitstream = debias_rec.output_bitstream
        tested_on = "debiased"
    else:
        if not exp.bitstream:
            raise HTTPException(status_code=422, detail="No bitstream available for testing.")
        bitstream = exp.bitstream
        tested_on = "original"

    try:
        result = statistical_tests.run_selected_tests(
            bitstream=bitstream,
            test_names=body.tests,
            alpha=body.significance_level,
            block_size=body.block_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Statistical test error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Test error: {exc}")

    # Persist
    exp_svc.save_test_run(db, experiment_id, result, tested_on)

    test_results = [
        SingleTestResult(
            test_name=r["test_name"],
            test_purpose=r["test_purpose"],
            outcome=TestOutcome(r["outcome"]),
            statistic=r.get("statistic"),
            p_value=r.get("p_value"),
            significance_level=r["significance_level"],
            minimum_bits_required=r["minimum_bits_required"],
            actual_bits_used=r["actual_bits_used"],
            interpretation=r["interpretation"],
            warning=r.get("warning"),
        )
        for r in result["results"]
    ]

    return TestRunResponse(
        experiment_id=experiment_id,
        test_run_id=result["test_run_id"],
        tested_on=tested_on,
        bits_analyzed=result["bits_analyzed"],
        significance_level=result["significance_level"],
        results=test_results,
        passed=result["passed"],
        failed=result["failed"],
        not_run=result["not_run"],
        summary_note=result["summary_note"],
    )


# ─────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────

@router.post(
    "/experiments/compare",
    response_model=CompareResponse,
    tags=["Experiments"],
)
def compare_experiments(body: CompareRequest, db: Session = Depends(get_db)):
    """Compare 2–5 experiments across key metrics."""
    entries = []
    warnings = []

    bit_counts = []
    for eid in body.experiment_ids:
        exp = exp_svc.get_experiment(db, eid)
        if not exp:
            warnings.append(f"Experiment {eid!r} not found and was skipped.")
            continue

        # Get test summary if available
        test_runs = exp_svc.get_test_runs(db, eid)
        nist_passed = nist_failed = None
        if test_runs:
            latest = test_runs[0]
            nist_passed = latest.passed
            nist_failed = latest.failed

        bit_counts.append(exp.actual_bits)
        entries.append(CompareEntry(
            experiment_id=exp.experiment_id,
            source_type=exp.source_type,
            backend_name=exp.backend_name,
            num_qubits=getattr(exp, "num_qubits", 1) or 1,
            actual_bits=exp.actual_bits,
            proportion_ones=exp.proportion_ones or 0.0,
            bias_deviation=exp.bias_deviation or 0.0,
            shannon_entropy=exp.shannon_entropy or 0.0,
            min_entropy=getattr(exp, "min_entropy", None),
            execution_time_seconds=exp.execution_time_seconds or 0.0,
            has_test_results=exp.has_test_results,
            nist_passed=nist_passed,
            nist_failed=nist_failed,
        ))

    if len(set(bit_counts)) > 1:
        warnings.append(
            "Experiments have different bit counts. Direct metric comparison "
            "may not be statistically meaningful."
        )

    sources = {e.source_type for e in entries}
    if "hardware" in sources and len(sources) > 1:
        warnings.append(
            "Comparing hardware and simulator results. Note that different sources "
            "have fundamentally different randomness properties."
        )

    return CompareResponse(
        entries=entries,
        warnings=warnings,
        comparison_note=(
            "Comparison shows observed metrics only. Different sample sizes, "
            "source types, and noise configurations make direct statistical "
            "comparison potentially misleading. Missing data is NOT fabricated."
        ),
    )


# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────

@router.get(
    "/experiments/{experiment_id}/export",
    tags=["Exports"],
)
def export_experiment(
    experiment_id: str,
    format: str = Query("json", description="Export format: txt, csv, json, pdf"),
    db: Session = Depends(get_db),
):
    """Export experiment data in the requested format."""
    exp = exp_svc.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")

    debias_rec = exp_svc.get_debias_result(db, experiment_id)
    test_runs = exp_svc.get_test_runs(db, experiment_id)

    fmt = format.lower()
    if fmt == "txt":
        if not exp.bitstream:
            raise HTTPException(status_code=422, detail="No bitstream available.")
        data = report_service.export_txt(exp.bitstream)
        return Response(
            content=data,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="qrng_{experiment_id[:8]}.txt"'},
        )
    elif fmt == "json":
        data = report_service.export_json(exp, debias_rec, test_runs)
        return Response(
            content=data,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="qrng_{experiment_id[:8]}.json"'},
        )
    elif fmt == "csv":
        data = report_service.export_csv(exp, debias_rec, test_runs)
        return Response(
            content=data,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="qrng_{experiment_id[:8]}.csv"'},
        )
    elif fmt == "pdf":
        try:
            data = report_service.export_pdf(exp, debias_rec, test_runs)
            return Response(
                content=data,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="qrng_{experiment_id[:8]}.pdf"'},
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {format!r}. Choose: txt, csv, json, pdf"
        )


# ─────────────────────────────────────────────
# Hardware (Optional)
# ─────────────────────────────────────────────

@router.get("/hardware/backends", tags=["Hardware (Optional)"])
def list_hardware_backends():
    """
    List available IBM Quantum backends.
    Only works when IBM_QUANTUM_TOKEN is configured.
    """
    if not hardware_engine.is_hardware_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "IBM Quantum is not configured. "
                "Set IBM_QUANTUM_TOKEN in your .env file to enable hardware access."
            ),
        )
    try:
        backends = hardware_engine.list_available_backends()
        return {"backends": backends, "note": "Real IBM Quantum backends."}
    except hardware_engine.HardwareJobError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post(
    "/hardware/jobs",
    status_code=202,
    tags=["Hardware (Optional)"],
)
def submit_hardware_job(body: HardwareJobRequest, db: Session = Depends(get_db)):
    """
    Submit an explicitly requested hardware experiment to IBM Quantum.
    Requires IBM_QUANTUM_TOKEN to be configured.
    NEVER silently falls back to simulation.
    """
    if not hardware_engine.is_hardware_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "IBM Quantum is not configured. "
                "Set IBM_QUANTUM_TOKEN in your .env file."
            ),
        )
    try:
        job_data = hardware_engine.submit_hardware_job(body.num_bits, body.backend_name)
    except hardware_engine.HardwareNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except hardware_engine.HardwareJobError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Persist job record (token is NOT stored)
    exp_svc.save_hardware_job(db, job_data)
    return job_data


@router.get("/hardware/jobs/{job_id}", tags=["Hardware (Optional)"])
def get_hardware_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the status and results of a submitted hardware job.
    Returns genuine hardware results only.
    NEVER fabricates or substitutes simulation output.
    """
    job_record = exp_svc.get_hardware_job(db, job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    if not hardware_engine.is_hardware_configured():
        raise HTTPException(
            status_code=503,
            detail="IBM Quantum is not configured. Cannot retrieve job results.",
        )
    try:
        status_data = hardware_engine.get_hardware_job_result(
            job_id, job_record.backend_name, settings.HARDWARE_JOB_TIMEOUT_SECONDS
        )
    except hardware_engine.HardwareJobError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # If job completed, save as experiment
    if status_data.get("status") == "DONE" and status_data.get("result"):
        hw_result = status_data["result"]
        hw_result["experiment_id"] = str(uuid.uuid4())
        hw_result["source_type"] = "hardware"
        hw_result["backend_name"] = job_record.backend_name
        hw_result["requested_bits"] = job_record.requested_bits or 0
        hw_result["actual_bits"] = len(hw_result.get("bitstream", ""))
        hw_result["is_seeded"] = False
        hw_result["execution_time_seconds"] = 0.0
        hw_result["hardware_job_id"] = job_id

        # Save the hardware result as a proper experiment
        from backend.app.database.models import Experiment
        hw_exp = Experiment(
            experiment_id=hw_result["experiment_id"],
            source_type="hardware",
            backend_name=job_record.backend_name,
            requested_bits=hw_result["requested_bits"],
            actual_bits=hw_result["actual_bits"],
            bitstream=hw_result.get("bitstream"),
            measurement_counts=hw_result.get("measurement_counts"),
            zeros=hw_result["stats"].get("zeros"),
            ones=hw_result["stats"].get("ones"),
            proportion_ones=hw_result["stats"].get("proportion_ones"),
            shannon_entropy=hw_result["stats"].get("shannon_entropy_per_bit"),
            bias_deviation=hw_result["stats"].get("bias_deviation"),
            source_classification_note=hw_result.get("source_classification_note"),
            warnings=hw_result.get("warnings", []),
            hardware_job_id=job_id,
            status="completed",
        )
        from backend.app.database.session import SessionLocal
        with SessionLocal() as _db:
            _db.add(hw_exp)
            exp_svc.update_hardware_job(
                _db, job_id, "DONE", result_experiment_id=hw_result["experiment_id"]
            )
            _db.commit()

    return status_data


# ─────────────────────────────────────────────
# Feature 1: Quantum Entropy Quality Analyzer
# ─────────────────────────────────────────────

@router.post(
    "/experiments/{experiment_id}/entropy-analysis",
    response_model=EntropyAnalysisResponse,
    tags=["Entropy Analysis"],
)
def analyze_experiment_entropy(
    experiment_id: str,
    body: EntropyAnalysisRequest = EntropyAnalysisRequest(),
    db: Session = Depends(get_db),
):
    """
    Perform deep information-theoretic entropy and sliding window analysis
    on an experiment's bitstream.
    """
    exp = exp_svc.get_experiment(db, experiment_id)
    if not exp or not exp.bitstream:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id!r} not found or has no bitstream."
        )

    profile = entropy_analyzer.compute_entropy_profile(exp.bitstream)
    win_analysis = entropy_analyzer.window_entropy_analysis(
        exp.bitstream, window_size=body.window_size, step_size=body.step_size
    )

    # Check if debiasing was performed
    deb_record = exp_svc.get_debias_result(db, experiment_id)
    deb_comp = None
    if deb_record and deb_record.output_bitstream:
        deb_comp = entropy_analyzer.compare_entropy_profiles(
            exp.bitstream, deb_record.output_bitstream
        )

    return EntropyAnalysisResponse(
        experiment_id=experiment_id,
        profile=profile,
        window_analysis=win_analysis,
        debiased_comparison=deb_comp,
        explanations=entropy_analyzer.ENTROPY_EXPLANATIONS,
    )


# ─────────────────────────────────────────────
# Feature 2: Automated Noise Sweep
# ─────────────────────────────────────────────

@router.post(
    "/experiments/noise-sweep",
    response_model=NoiseSweepResponse,
    tags=["Noise Experiment"],
)
def run_noise_sweep_endpoint(body: NoiseSweepRequest):
    """
    Execute an automated noise sweep across specified readout-error levels.
    """
    if not quantum_engine.is_available():
        raise HTTPException(status_code=503, detail="Qiskit Aer is not installed.")

    try:
        res = noise_experiment.run_noise_sweep(
            noise_levels=body.noise_levels,
            num_bits=body.num_bits,
            seed=body.seed,
            alpha=body.significance_level,
            asymmetry_ratio=body.asymmetry_ratio,
        )
        return NoiseSweepResponse(**res)
    except Exception as exc:
        logger.exception("Noise sweep failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Noise sweep error: {exc}")


# ─────────────────────────────────────────────
# Feature 4: Batch Statistical Evaluation
# ─────────────────────────────────────────────

@router.post(
    "/experiments/batch-evaluate",
    response_model=BatchEvaluationResponse,
    tags=["Batch Evaluation"],
)
def run_batch_evaluation_endpoint(body: BatchEvaluationRequest):
    """
    Run NIST SP 800-22 evaluation over multiple independent sequences.
    """
    if not quantum_engine.is_available():
        raise HTTPException(status_code=503, detail="Qiskit Aer is not installed.")

    try:
        res = batch_evaluator.run_batch_evaluation(
            num_sequences=body.num_sequences,
            bits_per_sequence=body.bits_per_sequence,
            source_type=body.source_type,
            tests=body.tests,
            alpha=body.significance_level,
            seed=body.seed,
            num_qubits=body.num_qubits,
        )
        return BatchEvaluationResponse(**res)
    except Exception as exc:
        logger.exception("Batch evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {exc}")


# ─────────────────────────────────────────────
# Feature 7: System Benchmarking
# ─────────────────────────────────────────────

@router.post(
    "/benchmarks/run",
    response_model=BenchmarkResponse,
    tags=["Benchmarking"],
)
def run_system_benchmark_endpoint(body: BenchmarkRequest):
    """
    Execute repeated benchmark trials across the QRNG pipeline.
    """
    if not quantum_engine.is_available():
        raise HTTPException(status_code=503, detail="Qiskit Aer is not installed.")

    try:
        res = benchmarker.run_pipeline_benchmark(
            num_bits=body.num_bits,
            trials=body.trials,
            qubit_configs=body.qubit_configs,
        )
        return BenchmarkResponse(**res)
    except Exception as exc:
        logger.exception("Benchmark failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Benchmark error: {exc}")


# ─────────────────────────────────────────────
# Feature 6: Interactive Circuit Visualization
# ─────────────────────────────────────────────

@router.get(
    "/circuits/visualize",
    response_model=CircuitVisualizerResponse,
    tags=["Circuits"],
)
def get_circuit_visualization(num_qubits: int = Query(1, ge=1, le=8)):
    """
    Retrieve stage-by-stage quantum circuit state, mathematical description,
    and ASCII diagram.
    """
    data = quantum_engine.get_circuit_visualization_data(num_qubits=num_qubits)
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])
    return CircuitVisualizerResponse(**data)


# ─────────────────────────────────────────────
# Feature 8: Hardware Calibration & Status
# ─────────────────────────────────────────────

@router.get(
    "/hardware/status",
    tags=["Hardware"],
)
def get_hardware_status_endpoint():
    """Check IBM Quantum credentials and backend operational status."""
    return hardware_engine.get_hardware_status()


@router.get(
    "/hardware/calibration",
    tags=["Hardware"],
)
def get_hardware_calibration_endpoint(backend_name: Optional[str] = None):
    """Retrieve genuine calibration data from IBM Quantum backend if available."""
    return hardware_engine.get_hardware_calibration(backend_name=backend_name)

