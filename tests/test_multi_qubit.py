"""
Tests for Feature 5: Multi-Qubit Generation
"""

import pytest
from backend.app.services.quantum_engine import (
    _build_hadamard_circuit,
    run_ideal_simulation,
    run_noisy_simulation,
    get_circuit_visualization_data,
)


class TestMultiQubitCircuit:
    def test_circuit_qubit_count_construction(self):
        for q in [1, 2, 4, 8]:
            qc = _build_hadamard_circuit(num_qubits=q)
            assert qc.num_qubits == q
            assert qc.num_clbits == q

    def test_circuit_qubit_clamping(self):
        qc_low = _build_hadamard_circuit(num_qubits=0)
        assert qc_low.num_qubits == 1
        qc_high = _build_hadamard_circuit(num_qubits=16)
        assert qc_high.num_qubits == 8

    def test_multi_qubit_ideal_simulation_bit_count(self):
        for q in [1, 2, 4, 8]:
            res = run_ideal_simulation(num_bits=1000, num_qubits=q, seed=42)
            assert res["actual_bits"] == 1000
            assert res["num_qubits"] == q
            assert len(res["bitstream"]) == 1000
            assert set(res["bitstream"]).issubset({"0", "1"})
            # Shots should be ceil(1000 / q)
            expected_shots = (1000 + q - 1) // q
            assert res["shots_executed"] == expected_shots

    def test_multi_qubit_noisy_simulation(self):
        res = run_noisy_simulation(num_bits=800, num_qubits=4, prob_meas0_prep1=0.08, seed=99)
        assert res["actual_bits"] == 800
        assert res["num_qubits"] == 4
        assert res["noise_params"]["num_qubits"] == 4

    def test_circuit_visualization_data(self):
        vis = get_circuit_visualization_data(num_qubits=2)
        assert vis["num_qubits"] == 2
        assert vis["total_basis_states"] == 4
        assert vis["theoretical_prob_per_state"] == 0.25
        assert len(vis["stages"]) == 4
        assert "circuit_ascii" in vis

