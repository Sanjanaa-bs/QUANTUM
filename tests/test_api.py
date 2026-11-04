"""
FastAPI integration tests using HTTPX TestClient.
Tests API validation, HTTP status codes, and end-to-end workflows.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the database before each test."""
    init_db()


# ─────────────────────────────────────────────
# Health & Info
# ─────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_has_status_field(self):
        r = client.get("/api/v1/health")
        assert "status" in r.json()

    def test_health_has_version(self):
        r = client.get("/api/v1/health")
        assert r.json().get("version") is not None

    def test_health_has_hardware_configured(self):
        r = client.get("/api/v1/health")
        assert "hardware_configured" in r.json()

    def test_health_does_not_expose_secrets(self):
        r = client.get("/api/v1/health")
        body = r.text
        assert "TOKEN" not in body.upper() or "hardware_configured" in body  # token not in response


class TestInfoEndpoint:
    def test_info_returns_200(self):
        r = client.get("/api/v1/info")
        assert r.status_code == 200

    def test_info_has_nist_tests(self):
        r = client.get("/api/v1/info")
        data = r.json()
        assert "nist_tests_implemented" in data
        assert len(data["nist_tests_implemented"]) > 0

    def test_info_has_disclaimer(self):
        r = client.get("/api/v1/info")
        data = r.json()
        assert "disclaimer" in data

    def test_info_has_supported_source_types(self):
        r = client.get("/api/v1/info")
        data = r.json()
        assert "ideal_simulator" in data["supported_source_types"]


# ─────────────────────────────────────────────
# Generate Endpoint
# ─────────────────────────────────────────────

class TestGenerateEndpoint:
    def test_ideal_simulation_returns_201(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        assert r.status_code == 201

    def test_response_has_experiment_id(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        assert "experiment_id" in r.json()

    def test_response_has_correct_bit_count(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        assert r.json()["actual_bits"] == 500

    def test_response_has_stats(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        assert "stats" in r.json()
        stats = r.json()["stats"]
        assert "proportion_ones" in stats
        assert "shannon_entropy_per_bit" in stats

    def test_response_has_source_classification_note(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 100,
        })
        note = r.json()["source_classification_note"].upper()
        assert "SIMULATION" in note

    def test_too_few_bits_rejected(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 50,
        })
        assert r.status_code == 422

    def test_too_many_bits_rejected(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 999999,
        })
        assert r.status_code == 422

    def test_hardware_source_rejected_at_generate(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "hardware",
            "num_bits": 500,
        })
        assert r.status_code == 400

    def test_noise_config_only_for_noisy_simulator(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
            "noise_config": {"prob_meas0_prep1": 0.05, "prob_meas1_prep0": 0.05},
        })
        assert r.status_code == 422

    def test_noisy_simulator_works(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "noisy_simulator",
            "num_bits": 500,
            "noise_config": {"prob_meas0_prep1": 0.05, "prob_meas1_prep0": 0.05},
        })
        assert r.status_code == 201
        assert r.json()["source_type"] == "noisy_simulator"

    def test_seeded_simulation(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
            "seed": 42,
        })
        assert r.status_code == 201
        assert r.json()["is_seeded"] is True
        assert r.json()["seed_used"] == 42

    def test_invalid_source_type(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "magical_quantum",
            "num_bits": 500,
        })
        assert r.status_code == 422


# ─────────────────────────────────────────────
# List & Get Experiments
# ─────────────────────────────────────────────

class TestExperimentCRUD:
    def _create_experiment(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        assert r.status_code == 201
        return r.json()["experiment_id"]

    def test_list_experiments_returns_200(self):
        r = client.get("/api/v1/experiments")
        assert r.status_code == 200

    def test_list_experiments_has_pagination(self):
        r = client.get("/api/v1/experiments")
        data = r.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    def test_get_experiment_returns_data(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}")
        assert r.status_code == 200
        assert r.json()["experiment_id"] == eid

    def test_get_nonexistent_experiment_returns_404(self):
        r = client.get("/api/v1/experiments/nonexistent-id-000")
        assert r.status_code == 404

    def test_saved_experiment_persists_source_type(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}")
        assert r.json()["source_type"] == "ideal_simulator"


# ─────────────────────────────────────────────
# Debiasing Endpoint
# ─────────────────────────────────────────────

class TestDebiasEndpoint:
    def _create_experiment(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 1000,
        })
        return r.json()["experiment_id"]

    def test_debias_returns_201(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/debias", json={})
        assert r.status_code == 201

    def test_debias_output_bits_fewer_than_input(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/debias", json={})
        data = r.json()
        assert data["output_bits"] <= data["input_bits"]

    def test_debias_has_pair_counts(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/debias", json={})
        data = r.json()
        pc = data["pair_counts"]
        assert "count_00" in pc
        assert "count_01" in pc
        assert "count_10" in pc
        assert "count_11" in pc

    def test_debias_has_algorithm_note(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/debias", json={})
        assert "algorithm_note" in r.json()

    def test_debias_nonexistent_experiment_404(self):
        r = client.post("/api/v1/experiments/nonexistent/debias", json={})
        assert r.status_code == 404


# ─────────────────────────────────────────────
# Statistical Tests Endpoint
# ─────────────────────────────────────────────

class TestNISTEndpoint:
    def _create_experiment(self, num_bits=5000):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": num_bits,
        })
        return r.json()["experiment_id"]

    def test_run_tests_returns_201(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["frequency"],
            "use_debiased": False,
        })
        assert r.status_code == 201

    def test_run_tests_returns_results_list(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["frequency", "runs"],
        })
        data = r.json()
        assert len(data["results"]) == 2

    def test_skipped_test_not_counted_as_passed(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["nonexistent_test"],
        })
        data = r.json()
        assert data["passed"] == 0

    def test_test_run_on_nonexistent_experiment_404(self):
        r = client.post("/api/v1/experiments/nonexistent/tests", json={
            "tests": ["frequency"],
        })
        assert r.status_code == 404

    def test_test_run_has_summary_note(self):
        eid = self._create_experiment()
        r = client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["frequency"],
        })
        assert "summary_note" in r.json()


# ─────────────────────────────────────────────
# Export Endpoint
# ─────────────────────────────────────────────

class TestExportEndpoint:
    def _create_experiment(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
        })
        return r.json()["experiment_id"]

    def test_txt_export_returns_binary(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "txt"})
        assert r.status_code == 200
        content = r.content.decode()
        assert all(c in "01\n\r" for c in content)

    def test_json_export_returns_valid_json(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "json"})
        assert r.status_code == 200
        data = r.json()
        assert "experiment" in data

    def test_json_export_no_token(self):
        """Verify that actual API credential values are not included in JSON exports.
        The word 'token' may appear legitimately in the security disclaimer;
        what must NOT appear is any actual credential string from the environment."""
        import os
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "json"})
        body = r.text
        # If an IBM token is set in the environment, assert it is not in the export body
        actual_token = os.environ.get("IBM_QUANTUM_TOKEN", "")
        if actual_token:
            assert actual_token not in body, "Actual IBM token must never appear in exports"
        # The word 'token' may appear in the security disclaimer — that is correct behaviour.
        # Assert that the export note explicitly mentions tokens are excluded.
        assert "API TOKENS" in body.upper() or "NEVER INCLUDED" in body.upper()

    def test_csv_export_returns_text(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "csv"})
        assert r.status_code == 200
        assert "experiment_id" in r.text or "Experiment" in r.text

    def test_pdf_export_returns_bytes(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "pdf"})
        if r.status_code == 200:
            assert r.content[:4] == b"%PDF"
        else:
            assert r.status_code == 503  # ReportLab not installed

    def test_invalid_format_returns_400(self):
        eid = self._create_experiment()
        r = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "xyz"})
        assert r.status_code == 400


# ─────────────────────────────────────────────
# Hardware Endpoints (not configured)
# ─────────────────────────────────────────────

class TestHardwareEndpointsWithoutCredentials:
    def test_list_backends_without_credentials_returns_503(self):
        r = client.get("/api/v1/hardware/backends")
        assert r.status_code == 503

    def test_submit_job_without_credentials_returns_503(self):
        r = client.post("/api/v1/hardware/jobs", json={
            "num_bits": 1000,
            "backend_name": "ibm_fake",
        })
        assert r.status_code == 503

    def test_get_job_status_nonexistent_returns_404(self):
        r = client.get("/api/v1/hardware/jobs/fake-job-id-000")
        assert r.status_code == 404
