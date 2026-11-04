"""
Hardware Engine Service — Optional IBM Quantum Integration
===========================================================

Requirements:
- IBM_QUANTUM_TOKEN must be set in the environment (never hardcoded).
- Never silently falls back to simulation when hardware is requested.
- Never fabricates a completed job or hardware output.
- Hardware access is explicitly initiated by the user only.
- Credentials are never stored in database, logs, or reports.

This module gracefully reports its unavailability when credentials or
the qiskit-ibm-runtime package are absent.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional IBM Runtime import ───────────────────────────────────────────────
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    logger.info(
        "qiskit-ibm-runtime not installed. Hardware execution is disabled."
    )


class HardwareNotConfiguredError(Exception):
    """Raised when IBM Quantum credentials are absent or invalid."""


class HardwareJobError(Exception):
    """Raised when a hardware job fails or times out."""


def is_hardware_configured() -> bool:
    """
    Returns True only if IBM_QUANTUM_TOKEN is set in the environment
    AND qiskit-ibm-runtime is installed.
    Reads from environment — never from hardcoded values.
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    return IBM_RUNTIME_AVAILABLE and bool(token)


def list_available_backends() -> List[Dict[str, Any]]:
    """
    Return a list of available IBM Quantum backends for the configured account.
    Raises HardwareNotConfiguredError if credentials are missing.
    """
    if not is_hardware_configured():
        raise HardwareNotConfiguredError(
            "IBM Quantum is not configured. Set IBM_QUANTUM_TOKEN in your .env file."
        )
    token = os.environ["IBM_QUANTUM_TOKEN"]
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        backends = service.backends(operational=True, simulator=False)
        return [
            {
                "name": b.name,
                "num_qubits": b.num_qubits,
                "status": "operational",
                "pending_jobs": getattr(b.status(), "pending_jobs", "unknown"),
            }
            for b in backends
        ]
    except Exception as exc:
        logger.error("Failed to list backends: %s", exc)
        raise HardwareJobError(f"Could not retrieve backend list: {exc}") from exc


def submit_hardware_job(
    num_bits: int,
    backend_name: str,
) -> Dict[str, Any]:
    """
    Submit a Hadamard-circuit job to a real IBM Quantum backend.

    Returns job metadata including the actual IBM job ID.
    NEVER fabricates a job ID or result.
    NEVER silently falls back to simulation.

    Args:
        num_bits: Number of shots (measurements).
        backend_name: IBM Quantum backend name.

    Returns:
        dict with job_id, backend_name, status, created_at.
    """
    if not is_hardware_configured():
        raise HardwareNotConfiguredError(
            "IBM Quantum credentials are not configured. "
            "Set IBM_QUANTUM_TOKEN in your .env file and try again."
        )

    # Import here to avoid NameError if not installed
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    token = os.environ["IBM_QUANTUM_TOKEN"]
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")

    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        backend = service.backend(backend_name)
    except Exception as exc:
        raise HardwareJobError(
            f"Could not connect to backend '{backend_name}': {exc}"
        ) from exc

    # Build the QRNG circuit
    qc = QuantumCircuit(1, 1, name="QRNG_Hardware")
    qc.h(0)
    qc.measure(0, 0)

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_circuit = pm.run(qc)
    except Exception:
        # Fallback transpile
        from qiskit import transpile
        isa_circuit = transpile(qc, backend=backend)

    try:
        sampler = Sampler(backend)
        job = sampler.run([isa_circuit], shots=num_bits)
        job_id = job.job_id()
        logger.info("Submitted hardware job %s to %s", job_id, backend_name)
    except Exception as exc:
        raise HardwareJobError(
            f"Failed to submit job to '{backend_name}': {exc}"
        ) from exc

    return {
        "job_id": job_id,
        "backend_name": backend_name,
        "status": "submitted",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_bits": num_bits,
    }


def get_hardware_job_result(
    job_id: str,
    backend_name: str,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """
    Retrieve the result of a previously submitted IBM Quantum job.

    Returns ONLY genuine hardware results.
    NEVER substitutes simulation output for a missing hardware result.
    """
    if not is_hardware_configured():
        raise HardwareNotConfiguredError(
            "IBM Quantum credentials are not configured."
        )

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ["IBM_QUANTUM_TOKEN"]
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")

    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        job = service.job(job_id)
    except Exception as exc:
        raise HardwareJobError(f"Could not retrieve job '{job_id}': {exc}") from exc

    status = job.status()
    status_str = str(status.name) if hasattr(status, "name") else str(status)

    if status_str in ("QUEUED", "RUNNING", "PENDING"):
        return {
            "job_id": job_id,
            "backend_name": backend_name,
            "status": status_str,
            "result": None,
            "message": f"Job is still {status_str}. Check back later.",
        }

    if status_str == "ERROR" or status_str == "CANCELLED":
        return {
            "job_id": job_id,
            "backend_name": backend_name,
            "status": status_str,
            "result": None,
            "message": f"Job ended with status {status_str}.",
        }

    # DONE — extract genuine results
    try:
        result = job.result()
        # SamplerV2 result structure
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
    except Exception as exc:
        raise HardwareJobError(
            f"Could not parse results for job '{job_id}': {exc}"
        ) from exc

    from backend.app.services.quantum_engine import (
        _counts_to_bitstream, _compute_stats, SOURCE_NOTES
    )
    normalized = {k.strip(): v for k, v in counts.items()}
    bitstream = _counts_to_bitstream(normalized)
    stats = _compute_stats(bitstream)

    return {
        "job_id": job_id,
        "backend_name": backend_name,
        "status": "DONE",
        "result": {
            "source_type": "hardware",
            "bitstream": bitstream,
            "measurement_counts": normalized,
            "stats": stats,
            "source_classification_note": SOURCE_NOTES["hardware"],
            "warnings": [
                "Hardware results reflect actual qubit measurements. "
                "Device errors, decoherence, and calibration affect outputs.",
                "Aggregated counts do not preserve chronological shot order.",
            ],
        },
    }


def get_hardware_status() -> Dict[str, Any]:
    """
    Get detailed hardware connectivity status and available backends.
    Does NOT reveal secrets or credentials.
    """
    configured = is_hardware_configured()
    if not configured:
        return {
            "configured": False,
            "runtime_installed": IBM_RUNTIME_AVAILABLE,
            "token_present": bool(os.environ.get("IBM_QUANTUM_TOKEN")),
            "message": (
                "IBM Quantum is not configured. To enable real hardware execution, "
                "set IBM_QUANTUM_TOKEN in your .env file. Classical simulation is fully functional."
            ),
            "backends": [],
        }

    try:
        backends = list_available_backends()
        return {
            "configured": True,
            "runtime_installed": True,
            "token_present": True,
            "message": f"Connected to IBM Quantum. Found {len(backends)} operational backend(s).",
            "backends": backends,
        }
    except Exception as exc:
        return {
            "configured": True,
            "runtime_installed": True,
            "token_present": True,
            "message": f"IBM Quantum configured but connection failed: {exc}",
            "backends": [],
        }


def get_hardware_calibration(backend_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch genuine device calibration and error metrics from IBM Quantum if available.
    Never fabricates calibration metrics.
    """
    if not is_hardware_configured():
        return {
            "configured": False,
            "message": "IBM Quantum is not configured. Real calibration metrics are unavailable.",
            "calibration_data": None,
        }

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ["IBM_QUANTUM_TOKEN"]
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)

    if not backend_name:
        backends = service.backends(operational=True, simulator=False)
        if not backends:
            return {"configured": True, "message": "No operational physical backends found.", "calibration_data": None}
        target_backend = backends[0]
    else:
        target_backend = service.backend(backend_name)

    props = target_backend.properties()
    if not props:
        return {
            "configured": True,
            "backend_name": target_backend.name,
            "message": "Backend properties / calibration table not returned by IBM API.",
            "calibration_data": None,
        }

    # Extract qubit 0 properties if available
    qubit_data = {}
    try:
        t1 = props.t1(0)
        t2 = props.t2(0)
        readout_err = props.readout_error(0)
        qubit_data = {
            "qubit_index": 0,
            "t1_us": round(t1 * 1e6, 2) if t1 else None,
            "t2_us": round(t2 * 1e6, 2) if t2 else None,
            "readout_error": round(readout_err, 5) if readout_err else None,
            "last_calibrated": str(props.last_update_date) if hasattr(props, "last_update_date") else None,
        }
    except Exception as exc:
        logger.warning("Could not read specific qubit properties: %s", exc)

    return {
        "configured": True,
        "backend_name": target_backend.name,
        "num_qubits": target_backend.num_qubits,
        "calibration_data": qubit_data,
        "message": f"Real calibration data retrieved for {target_backend.name}.",
    }

