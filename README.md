# Quantum Random Number Generator (QRNG)
## Comprehensive Quantum Entropy Generation, Von Neumann Debiasing & NIST SP 800-22 Randomness Analysis

**Course:** Quantum Computing (25AM3504) | **Project #15**  
**Technology Stack:** Python 3.10+ · Qiskit 1.2.4 · Qiskit Aer 0.15.1 · FastAPI · Streamlit · SQLite · ReportLab  
**Theme:** Minimalist Pastel Teal Academic Analytics Dashboard  
**Test Suite:** 209 Automated Pytest Cases (100% Passing)

---

## ⚠️ Scientific Integrity & Disclaimers

1. **Simulation vs. Physical Entropy:**  
   By default, this application executes on **Qiskit Aer**, a high-performance quantum circuit simulator running on host CPU architecture via classical pseudo-random algorithms. Output from classical simulators does **NOT** contain physical quantum entropy and must **NOT** be used as a cryptographically secure random number generator (CSPRNG).
2. **Hardware Execution (Optional):**  
   When configured with genuine IBM Quantum cloud credentials, circuits execute on real superconducting transmon qubits. These physical quantum measurements are subject to environmental thermal noise, state relaxation ($T_1$), dephasing ($T_2$), crosstalk, and measurement readout errors.
3. **Statistical Testing Scope:**  
   This application implements 5 selected tests from the NIST SP 800-22 Rev. 1a statistical test suite. Passing individual statistical tests does **NOT** prove mathematical or physical true randomness, nor does it certify cryptographic suitability.
4. **Descriptive Sample Metrics:**  
   Shannon entropy $H(X)$ and empirical min-entropy $H_{min} = -\log_2(\max(p_0, p_1))$ reported by this software are empirical descriptive sample statistics, not certified physical device-independent entropy lower bounds.

---

## 📋 Features

### 1. Quantum State Preparation & Multi-Qubit Register Simulation
- **Single and Multi-Qubit Registers:** Selectable 1, 2, 4, or 8-qubit register configurations.
- **Hadamard Transform Layer:** Initializes $|0\rangle^{\otimes n}$ and applies a parallel Hadamard transform layer to establish maximal superposition state:
  $$|\psi\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$$
- **Chronological Single-Shot Memory Acquisition:** Utilizes Qiskit Aer's memory retrieval mode (`memory=True`) to reconstruct the exact time-ordered sequence of measurement events.

### 2. Quantum Entropy Quality Analyzer
- **Shannon Entropy per Bit:**
  $$H(X) = -p_0 \log_2(p_0) - p_1 \log_2(p_1)$$
- **Empirical Min-Entropy Bound:**
  $$H_{min} = -\log_2(\max(p_0, p_1))$$
- **Sliding-Window Stationarity Analysis:** Configurable sliding-window proportion $P(1) = \text{count}(1) / W$ across time-ordered bitstreams with hover tooltips and ideal reference baseline ($y=0.5$).
- **Raw vs. Debiased Profile Comparison:** Side-by-side quantification of Shannon entropy gain, min-entropy gain, and bias reduction.

### 3. Noise Impact Experiment Lab
- **Automated Readout Noise Sweep:** Automated parametric sweep across readout error rates (e.g. 0%, 1%, 2%, 5%, 10%, 15%) using Qiskit Aer's noise model.
- **Parametric Tracking:** Evaluates bit bias deviation, Shannon entropy, min-entropy, extraction efficiency, and NIST test pass/fail rates as a function of detector infidelity.
- **Interactive Visualizations:** Interactive Plotly charts for entropy degradation curves and extraction yield drop-offs.

### 4. Von Neumann Debiasing Extractor
- **Algorithmic Bias Filtering:** Implements John von Neumann's classic unbiasing technique on non-overlapping consecutive pairs:
  - `01` $\to$ output bit `0`
  - `10` $\to$ output bit `1`
  - `00` $\to$ discarded
  - `11` $\to$ discarded
- **Rigorous Efficiency Metrics:**
  - $\text{Output-to-Input Ratio} = \frac{\text{Output Bits}}{\text{Input Bits}}$ (expected $\approx 25\%$ for balanced i.i.d. bits)
  - $\text{Pair Retention Ratio} = \frac{\text{Output Bits}}{\text{Complete Pairs}}$ (expected $\approx 50\%$ for balanced i.i.d. bits)
- **Pair Category Accounting:** Visual distribution and counts for `00`, `01`, `10`, and `11`.

### 5. NIST SP 800-22 Statistical Testing (Selected Suite)
- Implements 5 foundational statistical hypothesis tests with exact $p$-value computations:
  1. **Frequency (Monobit) Test:** Assesses ratio of zeros to ones.
  2. **Block Frequency Test:** Evaluates frequency within fixed $M$-bit blocks.
  3. **Runs Test:** Evaluates total number of uninterrupted runs of identical bits.
  4. **Longest Run of Ones in a Block Test:** Analyzes maximum run length within blocks.
  5. **Discrete Fourier Transform (Spectral) Test:** Detects periodic structures via FFT power spectrum.
- **Explicit Outcome Categorization:** Clearly marks outcomes as `PASS`, `FAIL`, or `INSUFFICIENT_DATA`.

### 6. Batch Statistical Evaluation Module
- **Multi-Sequence Sampling:** Generates $K$ independent sequence batches ($K \in [5, 100]$).
- **Proportion of Passes Confidence Interval (NIST Section 4.2.1):**
  $$(1 - \alpha) \pm 3 \sqrt{\frac{\alpha(1 - \alpha)}{K}}$$
- **p-Value Distribution Histogram:** 10 equal bins in $[0, 1]$ verifying uniform dispersion under $H_0$.
- **Chi-Square Uniformity Test (NIST Section 4.2.2):** Goodness-of-fit test for $p$-value distribution uniformity.

### 7. End-to-End Pipeline Benchmarking
- **Latency & Throughput Tracking:** Benchmarks circuit execution, Von Neumann debiasing, NIST testing, and PDF generation across repeated trials.
- **Metrics Computed:** Median, mean, standard deviation, IQR, and bits-per-second throughput across 1, 2, 4, and 8-qubit configurations.

### 8. Interactive Quantum Circuit Visualizer
- Visualizes the 4 conceptual stages of QRNG bit generation:
  1. Initial Ground State $|0\rangle^{\otimes n}$
  2. Hadamard Gate Transformation ($H$)
  3. Equal Superposition State
  4. Computational Basis Measurement
- Dynamic ASCII circuit diagram rendering from actual Qiskit circuit instances.

### 9. Optional IBM Quantum Hardware Integration
- Safe, non-blocking hardware backend querying with IBM Quantum Runtime SDK.
- Status and device calibration data display.
- Separation of physical QPU queue time from local simulator run times.

### 10. Persistence & Scientific Reporting
- **SQLite Database Persistence:** Full experiment history, debiasing logs, noise sweep points, and test results logged automatically with SQLAlchemy ORM.
- **Multi-Format Export:** Raw binary bitstream (`.txt`), structured JSON metadata (`.json`), tabular test summary (`.csv`), and publication-grade PDF report generated via ReportLab (`.pdf`).

---

## 🏗️ System Architecture

```
                               ┌────────────────────────────────────────┐
                               │     Streamlit Frontend (Port 8501)      │
                               │  Minimalist Pastel Teal Dashboard (8p) │
                               └──────────────────┬─────────────────────┘
                                                  │ HTTP / JSON
                               ┌──────────────────▼─────────────────────┐
                               │       FastAPI Backend (Port 8000)      │
                               │   REST API Routes, Pydantic Schemas    │
                               └──────┬───────────┬──────────────┬──────┘
                                      │           │              │
           ┌──────────────────────────┘           │              └──────────────────────────┐
           ▼                                      ▼                                         ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌───────────────────────────────────┐
│     Quantum Engine     │      │   Information Engine   │      │        Persistence & Export       │
├────────────────────────┤      ├────────────────────────┤      ├───────────────────────────────────┤
│ • Qiskit Aer (Ideal)   │      │ • Von Neumann Debias   │      │ • SQLite 3 (qrng.db)              │
│ • Qiskit Aer (Noisy)   │      │ • Shannon Entropy      │      │ • SQLAlchemy 2.0 ORM              │
│ • Multi-Qubit (1-8q)   │      │ • Empirical H_min      │      │ • ReportLab PDF Generator         │
│ • Memory (Shot Order)  │      │ • NIST SP 800-22 (5)   │      │ • CSV / JSON / TXT Exporters      │
│ • IBM Quantum Hardware │      │ • Batch Chi-Square     │      │ • Schema Auto-Migration           │
└────────────────────────┘      └────────────────────────┘      └───────────────────────────────────┘
```

---

## 💻 Technology Stack

| Layer | Component | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10 / 3.11 | Core runtime environment |
| **Quantum Framework** | Qiskit | 1.2.4 | Circuit construction & gate operations |
| **Quantum Simulator** | Qiskit Aer | 0.15.1 | Ideal & noisy readout error simulation |
| **Hardware Driver** | qiskit-ibm-runtime | Optional | IBM Quantum physical device connectivity |
| **Backend API** | FastAPI / Uvicorn | 0.141.1 / 0.52.4 | Async REST API, OpenAPI docs |
| **Data Validation** | Pydantic / Settings | 2.13.5 / 2.15.0 | Request/response typing & validation |
| **Database** | SQLite + SQLAlchemy | 2.0.52 | Relational experiment & audit persistence |
| **Scientific Math** | NumPy / SciPy | 2.2.6 / 1.15.0 | Vector math, FFT, incomplete gamma functions |
| **Data Processing** | Pandas | 2.3.0 | Tabular data manipulation & export formatting |
| **Visual Analytics** | Plotly Express / GO | 7.1.0 | Interactive scientific charts & heatmaps |
| **Frontend UI** | Streamlit | 1.65.0+ | Responsive pastel teal scientific dashboard |
| **Report Generation** | ReportLab | 5.0.0+ | Vector graphics PDF report generator |
| **Test Suite** | Pytest / AnyIO | 9.1.1 / 4.15.0 | 209 unit and integration test cases |

---

## 📁 Project Structure

```
qrng-project/
├── .streamlit/
│   └── config.toml               # Streamlit pastel teal theme configuration
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── config.py             # Environment settings & directory defaults
│       ├── main.py               # FastAPI application lifecycle & startup
│       ├── schemas.py            # Pydantic models for API serialization
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py         # REST endpoints for generation, tests, debias
│       ├── database/
│       │   ├── __init__.py
│       │   ├── models.py         # SQLAlchemy ORM models (Experiment, etc.)
│       │   └── session.py        # Database engine, session, & schema migration
│       └── services/
│           ├── __init__.py
│           ├── quantum_engine.py # Qiskit Aer simulator with memory & multi-qubit
│           ├── hardware_engine.py# IBM Quantum execution & status module
│           ├── debiasing.py      # Von Neumann pair extraction algorithms
│           ├── entropy_analyzer.py# Shannon entropy, min-entropy, sliding window
│           ├── statistical_tests.py# 5 NIST SP 800-22 statistical test suite
│           ├── noise_experiment.py# Automated readout error noise sweeps
│           ├── batch_evaluator.py# Multi-sequence confidence interval analysis
│           ├── benchmarker.py    # Computational latency & throughput benchmark
│           ├── experiment_service.py # CRUD service for experiments & test runs
│           └── report_service.py # ReportLab PDF, CSV, JSON, TXT generator
├── frontend/
│   ├── __init__.py
│   ├── app.py                   # 8-page interactive Streamlit dashboard
│   └── components.py            # CSS styling, pastel palette, Plotly layout
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest path & test session setup
│   ├── test_api.py              # API endpoint unit & integration tests
│   ├── test_database.py         # SQLAlchemy persistence & schema tests
│   ├── test_quantum_engine.py   # Quantum circuit & Qiskit Aer simulation tests
│   ├── test_multi_qubit.py      # Multi-qubit register & bitstream ordering tests
│   ├── test_debiasing.py        # Von Neumann pair mapping & boundary tests
│   ├── test_debiasing_upgrade.py# Efficiency ratio naming & formula tests
│   ├── test_entropy_analyzer.py # Shannon & min-entropy calculation tests
│   ├── test_sliding_window.py   # Sliding-window proportion stationarity tests
│   ├── test_statistical_tests.py# NIST SP 800-22 numerical accuracy tests
│   ├── test_noise_experiment.py # Noise sweep parameter & data tests
│   ├── test_batch_evaluator.py  # Multi-sequence chi-square uniformity tests
│   ├── test_benchmarker.py      # Benchmark statistics & IQR computation tests
│   └── test_new_endpoints.py    # Integration tests for upgraded endpoints
├── docs/
│   └── PROJECT_REPORT.md        # Comprehensive academic project report
├── data/                        # SQLite storage directory (.gitkeep)
├── reports/                     # Output directory for PDF exports (.gitkeep)
├── .env.example                 # Environment configuration template
├── .gitignore                   # Security & build ignore patterns
├── pytest.ini                   # Pytest testpath & warning configurations
├── README.md                    # Project documentation & execution guide
└── requirements.txt             # Exact project dependencies
```

---

## ⚙️ Installation Instructions (Windows)

### 1. Clone the Repository
```powershell
git clone https://github.com/Sanjanaa-bs/QUANTUM.git
cd QUANTUM/qrng-project
```

### 2. Create and Activate Virtual Environment
```powershell
# Create Python 3.10+ virtual environment
python -m venv venv

# Activate in PowerShell:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
```powershell
# Copy template to active .env file:
Copy-Item .env.example .env

# Edit .env if configuring IBM Quantum hardware tokens:
# IBM_QUANTUM_TOKEN=your_token_here
```

---

## ▶️ Running the Application

Open **two separate terminal windows** (with your virtual environment activated):

### Terminal 1 — Start the FastAPI Backend
```powershell
cd "C:\Users\sanjana\Downloads\QUANTUM\qrng-project"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend will be running at `http://127.0.0.1:8000` with Swagger documentation at `http://127.0.0.1:8000/docs`.*

### Terminal 2 — Start the Streamlit Dashboard
```powershell
cd "C:\Users\sanjana\Downloads\QUANTUM\qrng-project"
streamlit run frontend/app.py --server.port 8501
```
*Open your browser and navigate to `http://localhost:8501`.*

---

## 🧪 Executing the Test Suite

Run the complete test suite:
```powershell
cd "C:\Users\sanjana\Downloads\QUANTUM\qrng-project"
python -m pytest -v
```

**Expected output:**
```
======================= 209 passed in ~10s =======================
```

To run individual test categories:
```powershell
# Test quantum simulation and multi-qubit register construction:
python -m pytest tests/test_quantum_engine.py tests/test_multi_qubit.py -v

# Test sliding window stationarity analysis:
python -m pytest tests/test_sliding_window.py -v

# Test NIST SP 800-22 statistical test suite:
python -m pytest tests/test_statistical_tests.py -v

# Test Von Neumann debiasing and extraction ratios:
python -m pytest tests/test_debiasing.py tests/test_debiasing_upgrade.py -v
```

---

## 🔬 QRNG Execution Workflow

1. **Generation:** Open `http://localhost:8501` and navigate to **⚡ Generate Bits**. Select Simulator Mode (Ideal or Noisy), Qubit Register Size (1, 2, 4, 8), and Shot Count (e.g., 10,000). Click **⚡ Generate Random Bits**.
2. **Analysis:** Go to **📊 Bitstream Analysis** to inspect the zero/one distribution, calculate Shannon entropy, empirical min-entropy ($H_{min}$), and evaluate sliding-window stationarity ($P(1) = \text{count}(1)/W$).
3. **Debiasing:** Navigate to **🔀 Debiasing** to filter readout bias via the Von Neumann extractor. Inspect retained pairs (`01`, `10`), discarded pairs (`00`, `11`), and efficiency ratios.
4. **Statistical Testing:** Navigate to **🧪 NIST Tests** to run the 5 selected NIST tests on raw or debiased streams.
5. **Experiments & Reports:** Navigate to **📄 Reports & Help** to run automated noise sweeps, batch evaluations, and export a comprehensive PDF experiment report.

---

## 📸 Screenshots

*(Dashboard rendered in the minimalist pastel teal theme `#F8F7F2` / `#6FAFA5` / `#286D67`)*

- **Interactive Quantum Circuit Visualizer:** Step-by-step state preparation and measurement probabilities.
- **Bitstream Analysis & Sliding Window:** Local proportion tracking around ideal 0.5 reference line with interactive hover cards.
- **Von Neumann Debiasing Lab:** Extraction efficiency accounting and pair distribution.
- **Noise Impact Curve:** Entropy degradation and extraction drop-off under increasing readout errors.
- **Batch Evaluation:** p-value uniformity histogram and 99.7% confidence interval bounds.

---

## 🔮 Future Enhancements

- **Toeplitz Hashing Extractor:** Implement 2-universal hash family extractors (Leftover Hash Lemma) as an alternative to Von Neumann debiasing.
- **Physical Device-Independent Bell Tests:** Support experimental Clauser-Horne-Shimony-Holt (CHSH) inequality verification for certified physical randomness bounds.
- **Full NIST SP 800-22 Implementation:** Expand from the current 5 foundational tests to the full 15-test suite (including Maurer's Universal Statistical, Approximate Entropy, Random Excursions).
- **Quantum Hardware Streaming:** Real-time WebSocket streaming of live quantum measurements from IBM Quantum hardware backends.
