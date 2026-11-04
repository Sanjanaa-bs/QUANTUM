"""
Integration tests for newly added API routes and features.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


class TestNewEndpoints:
    def _create_experiment(self, num_bits=1000):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": num_bits,
            "num_qubits": 2,
        })
        assert r.status_code == 201
        return r.json()["experiment_id"]

    def test_multi_qubit_generate_endpoint(self):
        r = client.post("/api/v1/experiments/generate", json={
            "source_type": "ideal_simulator",
            "num_bits": 800,
            "num_qubits": 4,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["num_qubits"] == 4
        assert data["actual_bits"] == 800
        assert data["stats"]["min_entropy_per_bit"] > 0

    def test_entropy_analysis_endpoint(self):
        eid = self._create_experiment(num_bits=1200)
        # Apply debiasing first so comparison works
        client.post(f"/api/v1/experiments/{eid}/debias", json={})

        r = client.post(f"/api/v1/experiments/{eid}/entropy-analysis", json={
            "window_size": 128,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["experiment_id"] == eid
        assert "profile" in data
        assert "window_analysis" in data
        assert data["window_analysis"]["num_windows"] > 0
        assert data["debiased_comparison"] is not None

    def test_noise_sweep_endpoint(self):
        r = client.post("/api/v1/experiments/noise-sweep", json={
            "noise_levels": [0.0, 0.05],
            "num_bits": 1000,
            "seed": 42,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total_points"] == 2
        assert len(data["points"]) == 2

    def test_batch_evaluate_endpoint(self):
        r = client.post("/api/v1/experiments/batch-evaluate", json={
            "num_sequences": 5,
            "bits_per_sequence": 500,
            "tests": ["frequency", "runs"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["num_sequences"] == 5
        assert "frequency" in data["test_summaries"]

    def test_benchmark_endpoint(self):
        r = client.post("/api/v1/benchmarks/run", json={
            "num_bits": 800,
            "trials": 3,
            "qubit_configs": [1, 2],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["trials_executed"] == 3
        assert "qubit_scaling" in data

    def test_circuit_visualize_endpoint(self):
        r = client.get("/api/v1/circuits/visualize?num_qubits=2")
        assert r.status_code == 200
        data = r.json()
        assert data["num_qubits"] == 2
        assert len(data["stages"]) == 4

    def test_hardware_status_and_calibration_endpoints(self):
        r_status = client.get("/api/v1/hardware/status")
        assert r_status.status_code == 200
        assert "configured" in r_status.json()

        r_cal = client.get("/api/v1/hardware/calibration")
        assert r_cal.status_code == 200
        assert "configured" in r_cal.json()

