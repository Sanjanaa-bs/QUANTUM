"""
Tests for the Quantum Engine Service.
Uses Qiskit Aer for simulation — requires qiskit and qiskit-aer to be installed.
"""

import pytest
from backend.app.services import quantum_engine as qe


class TestQiskitAvailability:
    def test_qiskit_available(self):
        assert qe.QISKIT_AVAILABLE, "Qiskit and Qiskit-Aer must be installed"


class TestIdealSimulation:
    def test_returns_requested_bit_count(self):
        result = qe.run_ideal_simulation(num_bits=1000)
        assert result["actual_bits"] == 1000

    def test_bitstream_is_binary(self):
        result = qe.run_ideal_simulation(num_bits=500)
        assert all(c in "01" for c in result["bitstream"])

    def test_bitstream_length_matches_actual_bits(self):
        result = qe.run_ideal_simulation(num_bits=500)
        assert len(result["bitstream"]) == result["actual_bits"]

    def test_measurement_counts_sum_to_bits(self):
        result = qe.run_ideal_simulation(num_bits=1000)
        total = sum(result["measurement_counts"].values())
        assert total == 1000

    def test_source_type_labeled_correctly(self):
        result = qe.run_ideal_simulation(num_bits=100)
        assert result["source_type"] == "ideal_simulator"

    def test_source_note_mentions_simulation(self):
        result = qe.run_ideal_simulation(num_bits=100)
        note = result["source_classification_note"].upper()
        assert "SIMULATION" in note

    def test_source_note_not_physical_quantum(self):
        result = qe.run_ideal_simulation(num_bits=100)
        note = result["source_classification_note"].upper()
        # Must NOT claim it IS physical quantum
        assert "PSEUDO-RANDOM" in note or "NOT PHYSICAL" in note or "CLASSICAL" in note

    def test_stats_proportion_between_0_and_1(self):
        result = qe.run_ideal_simulation(num_bits=10000)
        assert 0.0 <= result["stats"]["proportion_ones"] <= 1.0

    def test_stats_zeros_plus_ones_equals_total(self):
        result = qe.run_ideal_simulation(num_bits=1000)
        s = result["stats"]
        assert s["zeros"] + s["ones"] == result["actual_bits"]

    def test_seeded_simulation_reproducible(self):
        r1 = qe.run_ideal_simulation(num_bits=1000, seed=42)
        r2 = qe.run_ideal_simulation(num_bits=1000, seed=42)
        assert r1["bitstream"] == r2["bitstream"], "Seeded simulation must be reproducible"

    def test_different_seeds_produce_different_results(self):
        r1 = qe.run_ideal_simulation(num_bits=1000, seed=1)
        r2 = qe.run_ideal_simulation(num_bits=1000, seed=2)
        # Different seeds should almost certainly produce different streams
        assert r1["bitstream"] != r2["bitstream"]

    def test_seed_recorded_in_metadata(self):
        result = qe.run_ideal_simulation(num_bits=100, seed=99)
        assert result["seed_used"] == 99
        assert result["is_seeded"] is True

    def test_unseeded_is_not_marked_seeded(self):
        result = qe.run_ideal_simulation(num_bits=100, seed=None)
        assert result["is_seeded"] is False
        assert result["seed_used"] is None

    def test_seeded_warning_issued(self):
        result = qe.run_ideal_simulation(num_bits=100, seed=42)
        assert any("seeded" in w.lower() or "reproducible" in w.lower()
                   for w in result["warnings"])

    def test_execution_time_positive(self):
        result = qe.run_ideal_simulation(num_bits=100)
        assert result["execution_time_seconds"] > 0

    def test_shannon_entropy_between_0_and_1(self):
        result = qe.run_ideal_simulation(num_bits=10000)
        assert 0.0 <= result["stats"]["shannon_entropy_per_bit"] <= 1.0

    def test_backend_name_contains_aer(self):
        result = qe.run_ideal_simulation(num_bits=100)
        assert "Aer" in result["backend_name"] or "aer" in result["backend_name"].lower()


class TestNoisySimulation:
    def test_returns_requested_bit_count(self):
        result = qe.run_noisy_simulation(num_bits=500, prob_meas0_prep1=0.05, prob_meas1_prep0=0.05)
        assert result["actual_bits"] == 500

    def test_source_type_labeled_noisy(self):
        result = qe.run_noisy_simulation(num_bits=100)
        assert result["source_type"] == "noisy_simulator"

    def test_noise_params_recorded(self):
        result = qe.run_noisy_simulation(num_bits=100, prob_meas0_prep1=0.1, prob_meas1_prep0=0.1)
        assert result["noise_params"]["prob_meas0_prep1"] == pytest.approx(0.1)
        assert result["noise_params"]["prob_meas1_prep0"] == pytest.approx(0.1)

    def test_noise_model_note_present(self):
        result = qe.run_noisy_simulation(num_bits=100)
        note = result["noise_params"].get("note", "")
        assert "not reproduce" in note.lower() or "simplified" in note.lower()

    def test_source_note_mentions_noisy_simulation(self):
        result = qe.run_noisy_simulation(num_bits=100)
        note = result["source_classification_note"].upper()
        assert "SIMULATION" in note and ("NOISY" in note or "READOUT" in note or "NOISE" in note)

    def test_high_noise_biases_stream(self):
        """With extreme noise p=0.9, the proportion of 1s should deviate significantly from 0.5."""
        result = qe.run_noisy_simulation(
            num_bits=5000, prob_meas0_prep1=0.01, prob_meas1_prep0=0.9, seed=42
        )
        # Extreme bias toward 1 when measuring 0→ flips heavily
        p1 = result["stats"]["proportion_ones"]
        # With p(1|0)=0.9, most 0s become 1s, so p1 >> 0.5
        assert p1 > 0.7 or p1 < 0.3, f"Expected biased output, got proportion_ones={p1}"

    def test_seeded_noisy_reproducible(self):
        r1 = qe.run_noisy_simulation(num_bits=500, seed=7)
        r2 = qe.run_noisy_simulation(num_bits=500, seed=7)
        assert r1["bitstream"] == r2["bitstream"]


class TestComputeStats:
    def test_all_zeros(self):
        stats = qe._compute_stats("000000")
        assert stats["zeros"] == 6
        assert stats["ones"] == 0
        assert stats["proportion_ones"] == 0.0
        assert stats["bias_deviation"] == pytest.approx(0.5)

    def test_all_ones(self):
        stats = qe._compute_stats("111111")
        assert stats["zeros"] == 0
        assert stats["ones"] == 6
        assert stats["proportion_ones"] == 1.0

    def test_equal_split(self):
        stats = qe._compute_stats("01" * 500)
        assert stats["proportion_ones"] == pytest.approx(0.5)
        assert stats["bias_deviation"] == pytest.approx(0.0)
        assert stats["shannon_entropy_per_bit"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_stream(self):
        stats = qe._compute_stats("")
        assert stats["total_bits"] == 0
        assert stats["shannon_entropy_per_bit"] == 0.0

    def test_single_bit_0(self):
        stats = qe._compute_stats("0")
        assert stats["zeros"] == 1
        assert stats["ones"] == 0


class TestCountsToBitstream:
    def test_counts_sum_to_bitstream_length(self):
        counts = {"0": 300, "1": 200}
        bs = qe._counts_to_bitstream(counts)
        assert len(bs) == 500

    def test_only_zeros(self):
        bs = qe._counts_to_bitstream({"0": 10})
        assert bs == "0" * 10

    def test_only_ones(self):
        bs = qe._counts_to_bitstream({"1": 5})
        assert bs == "1" * 5

    def test_empty_counts(self):
        bs = qe._counts_to_bitstream({})
        assert bs == ""
