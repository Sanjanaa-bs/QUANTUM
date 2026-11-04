# QRNG Project Report

## Abstract

This project implements a Quantum Random Number Generator (QRNG) for the Quantum Computing course (25AM3504), Project #15. The system generates binary sequences using Qiskit quantum circuit simulation, applies Von Neumann debiasing, evaluates the output with selected NIST SP 800-22 statistical tests, and provides a full-stack application with a Streamlit dashboard and FastAPI REST API. Optional integration with real IBM Quantum hardware is supported when credentials are available. All modes are explicitly labeled and distinguished; simulator output is never claimed to be physical quantum randomness.

## 1. Introduction

Quantum random number generation exploits the fundamental indeterminacy of quantum measurement. A qubit initialized in the ground state |0⟩ and measured after a Hadamard gate yields outcome 0 or 1 with equal probability (P = 0.5 each), due to the collapse of the superposition state (|0⟩ + |1⟩)/√2. Unlike classical pseudo-random number generators (PRNGs), ideal quantum measurement outcomes are not determined by prior state; the indeterminacy is a physical property of quantum mechanics.

**Important:** Classical simulation of quantum circuits (as performed by Qiskit Aer) uses pseudo-random sampling and does NOT produce physical quantum randomness. This distinction is documented throughout the application.

## 2. Problem Statement

Generate, evaluate, and compare random bitstreams from:
1. Ideal quantum circuit simulation
2. Noisy quantum circuit simulation (readout-error model)
3. (Optional) Real IBM Quantum hardware

Evaluate the resulting bitstreams with the Von Neumann extractor and a selected subset of NIST SP 800-22 statistical tests.

## 3. Objectives

1. Build and execute a Hadamard-gate quantum circuit using Qiskit
2. Implement Von Neumann debiasing
3. Implement 5 selected NIST SP 800-22 tests with correct p-value computation
4. Provide a full-stack dashboard (Streamlit + FastAPI)
5. Persist experiments to SQLite with full reproducibility metadata
6. Support optional real hardware execution without silently substituting simulation

## 4. Scope

- **In scope:** Ideal simulation, noisy simulation, debiasing, 5 NIST tests, full-stack app, PDF reports
- **Out of scope:** Full 15-test NIST suite; cryptographic certification; production security review

## 5. Related Work

- NIST SP 800-22 Rev. 1a (Bassham et al., 2010): The authoritative reference for statistical tests for RNGs used in cryptography.
- Von Neumann (1951): The original description of the pair-based debiasing extractor.
- Qiskit (IBM, 2023): Open-source framework for quantum circuit creation and execution.
- Nielsen & Chuang (2010): Standard textbook reference for quantum computation fundamentals.

## 6. System Architecture

See docs/ARCHITECTURE.md for full details.

Key components:
- **Quantum Engine** (`quantum_engine.py`): Ideal and noisy Hadamard circuit simulation
- **Hardware Engine** (`hardware_engine.py`): Optional IBM Quantum integration
- **Debiasing** (`debiasing.py`): Von Neumann extractor
- **Statistical Tests** (`statistical_tests.py`): 5 NIST SP 800-22 tests
- **Experiment Service** (`experiment_service.py`): SQLite CRUD
- **Report Service** (`report_service.py`): TXT/CSV/JSON/PDF export
- **API** (`routes.py`): FastAPI REST endpoints
- **Frontend** (`app.py`): 8-section Streamlit dashboard

## 7. Quantum Theory and Circuit Design

### Hadamard Gate

The Hadamard gate H is a single-qubit unitary operation:
```
H = (1/√2) * [[1, 1], [1, -1]]
```

Applied to |0⟩:
```
H|0⟩ = (|0⟩ + |1⟩)/√2
```

Measurement of this state:
- P(0) = |⟨0|H|0⟩|² = 0.5
- P(1) = |⟨1|H|0⟩|² = 0.5

### Circuit
```
q[0]: |0⟩ ─── H ─── M ─── c[0]
```

### Shot-based execution
Each "shot" is one independent execution of the circuit. With n shots, the measurement results are aggregated into counts, e.g., {"0": 5012, "1": 4988}. The temporal ordering of individual shots is not preserved in aggregated counts.

## 8. Implementation Methodology

### 8.1 Ideal Simulation
- Uses Qiskit Aer's AerSimulator (noiseless)
- Optional reproducible seed via `seed_simulator`
- Source: classical pseudo-random sampling of quantum state amplitudes

### 8.2 Noisy Simulation
- Adds a readout-error noise model to AerSimulator
- Configurable P(measure 0 | prepared |1⟩) and P(measure 1 | prepared |0⟩)
- Does NOT reproduce any specific real quantum device

### 8.3 Hardware (Optional)
- Uses IBM Quantum Runtime API (qiskit-ibm-runtime)
- Job submitted only on explicit user request
- Token stored only in environment variable, never in database

## 9. Debiasing Algorithm

Von Neumann extractor (1951):
- Input: binary string of n bits
- Process: non-overlapping consecutive pairs
  - 01 → output 0
  - 10 → output 1
  - 00 → discard
  - 11 → discard
- Expected efficiency: ~25% for i.i.d. Bernoulli(0.5) input; lower for biased input

**Limitations:**
- Assumes i.i.d. input bits
- Does not remove arbitrary correlations
- Substantially reduces available bit count
- Does not prove output is truly random

## 10. Statistical Testing Methodology

This application implements 5 of the 15 NIST SP 800-22 tests:

| Test | NIST Section | Min Bits | Tests For |
|------|-------------|----------|-----------|
| Frequency (Monobit) | 2.1 | 100 | Equal proportion of 0s and 1s |
| Block Frequency | 2.2 | 100 | Equal proportion within blocks |
| Runs | 2.3 | 100 | Appropriate run lengths |
| Longest Run of Ones | 2.4 | 128 | Appropriate maximum run length |
| Spectral (DFT) | 2.6 | 1000 | Absence of periodic patterns |

**Not implemented:** Serial, Non-overlapping Template, Overlapping Template, Maurer's Universal, Linear Complexity, Cumulative Sums, Random Excursions, Random Excursions Variant, Approximate Entropy, Binary Matrix Rank.

## 11. Experimental Results

*[This section will be populated after running actual experiments. Run the application, generate experiments, and re-generate this report to include real results.]*

Placeholder: After generating experiments via the dashboard or API, the PDF report and JSON export will contain actual measured values for:
- Bit counts and proportions
- Execution times
- Debiasing extraction efficiency
- NIST test p-values and outcomes

## 12. Discussion and Limitations

- Simulator output represents pseudo-random sampling; results reflect the properties of the Qiskit Aer PRNG, not physical quantum mechanics
- NIST tests assess specific statistical properties; they are not exhaustive proofs of randomness
- The Von Neumann extractor's effectiveness depends on the i.i.d. assumption
- A single p-value < α does not prove a generator is defective; multiple-comparison effects apply

## 13. Conclusion and Future Work

This project delivers a working QRNG application with:
- Three source modes (ideal sim, noisy sim, hardware-optional)
- Von Neumann debiasing
- 5 NIST SP 800-22 tests
- Complete REST API and Streamlit dashboard
- Full experiment persistence and reproducibility

**Future work:**
- Implement remaining NIST SP 800-22 tests (10 of 15)
- Add support for multi-qubit circuits for higher throughput
- Integrate with additional quantum hardware providers
- Add real-time streaming bit generation

## 14. References

[1] Bassham, L. E., et al. (2010). NIST SP 800-22 Rev. 1a. NIST.
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

[2] Qiskit contributors (2023). Qiskit: An Open-source Framework for Quantum Computing.
    https://qiskit.org/documentation/

[3] Qiskit Aer (2023). https://qiskit.github.io/qiskit-aer/

[4] Von Neumann, J. (1951). Various techniques used in connection with random digits.
    NBS Applied Mathematics Series, 12, 36–38. National Bureau of Standards.

[5] Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information.
    Cambridge University Press.

[6] IBM Quantum (2023). IBM Quantum Platform. https://quantum.ibm.com
