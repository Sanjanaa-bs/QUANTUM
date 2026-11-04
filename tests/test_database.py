"""
Database persistence and report generation tests.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.main import app
from backend.app.database.models import Base, Experiment, DebiasingRecord, StatisticalTestRun
from backend.app.database.session import init_db
from backend.app.services import experiment_service as exp_svc

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


# ─────────────────────────────────────────────
# Database persistence tests
# ─────────────────────────────────────────────

class TestDatabasePersistence:
    def _generate_and_get_id(self, num_bits=500):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": num_bits,
        })
        assert r.status_code == 201
        return r.json()["experiment_id"]

    def test_generated_experiment_is_saved(self):
        eid = self._generate_and_get_id()
        r = client.get(f"/api/v1/experiments/{eid}")
        assert r.status_code == 200

    def test_experiment_id_is_unique(self):
        id1 = self._generate_and_get_id()
        id2 = self._generate_and_get_id()
        assert id1 != id2

    def test_experiment_source_type_persisted(self):
        eid = self._generate_and_get_id()
        r = client.get(f"/api/v1/experiments/{eid}")
        assert r.json()["source_type"] == "ideal_simulator"

    def test_experiment_bit_count_persisted(self):
        eid = self._generate_and_get_id(num_bits=750)
        r = client.get(f"/api/v1/experiments/{eid}")
        assert r.json()["actual_bits"] == 750

    def test_experiment_seed_persisted(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 500,
            "seed": 99,
        })
        eid = r.json()["experiment_id"]
        r2 = client.get(f"/api/v1/experiments/{eid}")
        assert r2.json()["seed_used"] == 99
        assert r2.json()["is_seeded"] is True

    def test_experiment_appears_in_list(self):
        eid = self._generate_and_get_id()
        r = client.get("/api/v1/experiments")
        ids = [e["experiment_id"] for e in r.json()["experiments"]]
        assert eid in ids

    def test_debiasing_record_saved(self):
        eid = self._generate_and_get_id(num_bits=1000)
        r = client.post(f"/api/v1/experiments/{eid}/debias", json={})
        assert r.status_code == 201
        # Check the experiment now has has_debiased=True
        r2 = client.get(f"/api/v1/experiments/{eid}")
        assert r2.json()["has_debiased"] is True

    def test_test_run_record_saved(self):
        eid = self._generate_and_get_id(num_bits=5000)
        r = client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["frequency"],
        })
        assert r.status_code == 201
        r2 = client.get(f"/api/v1/experiments/{eid}")
        assert r2.json()["has_test_results"] is True

    def test_no_api_tokens_in_db_records(self):
        eid = self._generate_and_get_id()
        r = client.get(f"/api/v1/experiments/{eid}")
        body = r.text
        assert "TOKEN" not in body.upper()

    def test_experiment_has_source_classification_note(self):
        eid = self._generate_and_get_id()
        r = client.get(f"/api/v1/experiments/{eid}")
        note = r.json().get("source_classification_note", "")
        assert len(note) > 0
        assert "SIMULATION" in note.upper()


# ─────────────────────────────────────────────
# Report generation tests
# ─────────────────────────────────────────────

class TestReportGeneration:
    def _create_full_experiment(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 2000,
        })
        eid = r.json()["experiment_id"]
        client.post(f"/api/v1/experiments/{eid}/debias", json={})
        client.post(f"/api/v1/experiments/{eid}/tests", json={
            "tests": ["frequency", "runs"],
        })
        return eid

    def test_txt_export_contains_only_bits(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator", "num_bits": 200
        })
        eid = r.json()["experiment_id"]
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "txt"})
        content = export.content.decode().strip()
        assert all(c in "01" for c in content)

    def test_json_export_has_experiment_section(self):
        eid = self._create_full_experiment()
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "json"})
        data = export.json()
        assert "experiment" in data
        assert data["experiment"]["experiment_id"] == eid

    def test_json_export_has_debiasing_section(self):
        eid = self._create_full_experiment()
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "json"})
        data = export.json()
        assert data.get("debiasing") is not None

    def test_json_export_has_stats_tests(self):
        eid = self._create_full_experiment()
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "json"})
        data = export.json()
        assert len(data.get("statistical_tests", [])) > 0

    def test_csv_export_has_experiment_metadata(self):
        eid = self._create_full_experiment()
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "csv"})
        content = export.content.decode()
        assert "experiment_id" in content.lower() or eid[:8] in content

    def test_pdf_generates_without_error(self):
        eid = self._create_full_experiment()
        export = client.get(f"/api/v1/experiments/{eid}/export", params={"format": "pdf"})
        # Either success (200) or reportlab not installed (503)
        assert export.status_code in (200, 503)
        if export.status_code == 200:
            assert export.content[:4] == b"%PDF"


# ─────────────────────────────────────────────
# Hardware missing/invalid config tests
# ─────────────────────────────────────────────

class TestHardwareSafety:
    def test_hardware_generation_returns_400_not_simulation(self):
        """Hardware requests go to /hardware/jobs, not /experiments/generate.
        Attempting hardware via generate endpoint must return 400, not silently simulate."""
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "hardware",
            "num_bits": 500,
        })
        assert r.status_code == 400
        # Must NOT return simulation result
        assert "experiment_id" not in r.json() or r.status_code != 201

    def test_hardware_never_silently_falls_back(self):
        """Without credentials, hardware job submission returns 503, never a simulation result."""
        r = client.post("/api/v1/hardware/jobs", json={
            "num_bits": 500,
            "backend_name": "fake_backend",
        })
        assert r.status_code == 503
        # No bitstream in error response
        assert "bitstream" not in r.json()

    def test_hardware_job_status_for_unknown_job_is_404(self):
        r = client.get("/api/v1/hardware/jobs/completely-fake-job-99999")
        assert r.status_code == 404
