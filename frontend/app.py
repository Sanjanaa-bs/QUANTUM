"""
QRNG Streamlit Frontend — Main Application
Course: Quantum Computing (25AM3504) | Project #15
Minimalist Pastel Teal Scientific Dashboard

All pages connect to the real FastAPI backend.
No hardcoded demonstration results in production views.
"""

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from frontend.components import (
    GLOBAL_CSS, API_BASE,
    api_get, api_post, api_get_bytes,
    source_badge_html, outcome_badge_html, format_bits_preview,
    PALETTE, get_plotly_layout,
)
from backend.app.services.entropy_analyzer import calculate_sliding_window_proportions

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="QRNG — Quantum Random Number Generator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar navigation (Pastel Teal Design)
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 14px 0 10px 0;">
        <div style="font-size:30px; line-height: 1.2;">⚛️</div>
        <div style="font-size:17px; font-weight:700; color:#286D67; letter-spacing:1.5px; margin-top:4px;">QRNG</div>
        <div style="font-size:11px; color:#71817E; margin-top:2px;">Quantum Random Number Generator</div>
        <div style="font-size:10px; color:#95A5A1; margin-top:2px;">25AM3504 · Project #15</div>
    </div>
    <hr style="border-color:#DCE7E3; margin:6px 0 12px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "⚡ Generate Bits",
            "📊 Bitstream Analysis",
            "🔀 Debiasing",
            "🧪 NIST Tests",
            "⚖️ Experiment Comparison",
            "📁 Experiment History",
            "📄 Reports & Help",
        ],
        label_visibility="collapsed",
    )

    # Backend and system status in sidebar footer
    st.markdown("<hr style='border-color:#DCE7E3; margin:16px 0 8px 0;'>", unsafe_allow_html=True)
    health = api_get("/health")
    if "_error" in health:
        st.markdown('<div class="badge-fail">⚠ Backend offline</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-size:11px; color:#71817E; line-height:1.7; padding: 4px 6px;">
            <span style="font-weight:600; color:#286D67;">System Status</span><br>
            Backend: <span style="color:#4D9A7A; font-weight:600;">✓ Online</span><br>
            Hardware: {'<span style="color:#286D67; font-weight:600;">✓ Configured</span>' if health.get('hardware_configured') else '<span style="color:#71817E;">✗ Simulator Mode</span>'}<br>
            Database: {'<span style="color:#4D9A7A; font-weight:600;">✓ OK</span>' if health.get('database_ok') else '<span style="color:#C96C70; font-weight:600;">✗ Error</span>'}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────

def _sget(key: str, default=None):
    return st.session_state.get(key, default)


def _sset(key: str, value):
    st.session_state[key] = value


# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown('<div class="section-header">Quantum Random Number Generator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Course:</b> Quantum Computing (25AM3504) &nbsp;|&nbsp; <b>Project #15</b><br>
    A scientifically grounded QRNG pipeline featuring Qiskit Aer simulation, optional IBM Quantum hardware integration,
    Von Neumann debiasing, Shannon & empirical min-entropy quantification, and selected NIST SP 800-22 statistical analysis.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🔬 Principles of Quantum Randomness")
        st.markdown("""
        A **Quantum Random Number Generator (QRNG)** extracts unpredictability directly from the measurement postulate of quantum mechanics:
        
        1. **State Initialization:** A qubit register is initialized in ground state **|0⟩**.
        2. **Superposition:** A Hadamard gate $H$ generates an equal quantum superposition:
           $$H|0\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}$$
        3. **Projective Measurement:** Measuring in the computational basis collapses the state to **0** or **1** with theoretical probability $P(0) = P(1) = 0.5$.
        
        In physical quantum systems, this measurement process is fundamentally non-deterministic.
        """)

        st.markdown("### ⚠️ Simulation vs. Physical Hardware")
        st.markdown("""
        <div class="disclaimer-box">
        <b>SIMULATION MODES (Default):</b><br>
        Qiskit Aer simulates quantum circuits using classical pseudo-random sampling algorithms on the host CPU.
        Simulator output is deterministic and pseudo-random; it contains <i>no physical entropy</i> and does not provide cryptographic security.
        <br><br>
        <b>HARDWARE MODE (Optional):</b><br>
        When IBM Quantum credentials are provided, circuits execute on superconducting quantum processors.
        These represent genuine quantum measurements subjected to real-world decoherence ($T_1, T_2$), gate errors, and readout infidelities.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🏗️ Technical Architecture")
        st.markdown("""
        | Layer | Technology | Primary Function |
        |---|---|---|
        | **Frontend** | Streamlit | Minimalist pastel analytics dashboard |
        | **Backend API** | FastAPI + Uvicorn | RESTful endpoints & computation pipeline |
        | **Quantum Engine** | Qiskit 1.2.4 + Qiskit Aer 0.15.1 | Ideal & noisy readout error simulations |
        | **Debiasing** | Von Neumann Extractor | Pair-wise algorithmic bias reduction |
        | **Statistical Tests** | NIST SP 800-22 (5 tests) | Frequency, Block Freq, Runs, Longest Run, Spectral |
        | **Entropy Engine** | Custom Analyzer | Shannon entropy & empirical min-entropy ($H_{min}$) |
        | **Persistence** | SQLite + SQLAlchemy 2.0 | Complete experiment & test history logging |
        | **Reporting** | ReportLab | Scientific PDF & CSV/JSON export |
        """)

    with col2:
        st.markdown("### 📋 Quick Workflow")
        st.markdown("""
        <div class="metric-card">
        <b>Step 1 — Generate Bits</b><br>
        Configure simulator mode or hardware, qubit register size (1–8q), and shot count in <b>⚡ Generate Bits</b>.
        </div>
        <div class="metric-card">
        <b>Step 2 — Entropy & Stationarity</b><br>
        Inspect zero/one proportions, Shannon entropy, min-entropy ($H_{min}$), and sliding-window stability in <b>📊 Bitstream Analysis</b>.
        </div>
        <div class="metric-card">
        <b>Step 3 — Von Neumann Debiasing</b><br>
        Filter measurement bias via non-overlapping pair extraction ($01\\to 0, 10\\to 1$) in <b>🔀 Debiasing</b>.
        </div>
        <div class="metric-card">
        <b>Step 4 — NIST Statistical Testing</b><br>
        Evaluate $p$-values against significance level $\\alpha$ in <b>🧪 NIST Tests</b>.
        </div>
        <div class="metric-card">
        <b>Step 5 — Benchmarks & Export</b><br>
        Run automated noise sweeps, batch evaluations, and export PDF reports in <b>📄 Reports & Help</b>.
        </div>
        """, unsafe_allow_html=True)

        # System status card
        info = api_get("/info")
        if "_error" not in info:
            st.markdown("### 🔌 Backend Information")
            st.success(f"✓ Backend v{info.get('version', '1.0.0')} Online")
            hw_status = "✓ Configured" if info.get("hardware_configured") else "✗ Not configured (simulator mode)"
            if info.get("hardware_configured"):
                st.success(f"Hardware: {hw_status}")
            else:
                st.info(f"Hardware: {hw_status}")
            st.caption(f"NIST tests loaded: {', '.join(info.get('nist_tests_implemented', []))}")
        else:
            st.error("⚠ Backend unreachable. Run: `uvicorn backend.app.main:app --reload`")

    # ── Interactive Circuit Visualizer ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚛️ Interactive Quantum Circuit Visualizer")

    cv_col1, cv_col2 = st.columns([1, 3])
    with cv_col1:
        viz_qubits = st.selectbox("Circuit Qubit Count", [1, 2, 4, 8], index=0, key="cv_qubits")
        if st.button("🔍 Load Circuit", key="cv_load"):
            cv_data = api_get(f"/circuits/visualize?num_qubits={viz_qubits}")
            _sset("circuit_viz", cv_data)

    cv_data = _sget("circuit_viz")
    if cv_data and "_error" not in cv_data:
        with cv_col2:
            st.markdown(f"**{cv_data.get('circuit_name', 'Hadamard Circuit')}** — {cv_data.get('description', '')}")
            stages = cv_data.get("stages", [])
            if stages:
                tabs_stage = st.tabs([s["stage_name"] for s in stages])
                for i, s in enumerate(stages):
                    with tabs_stage[i]:
                        st.markdown(f"**{s['stage_name']}**")
                        st.info(s.get("description", ""))
                        if s.get("state_vector"):
                            st.code(s["state_vector"], language="text")
                        if s.get("probabilities"):
                            prob = s["probabilities"]
                            probs_text = "  |  ".join([f"P({k}) = {v:.4f}" for k, v in prob.items()])
                            st.markdown(f"**Measurement Probabilities:** {probs_text}")
                        if s.get("math_note"):
                            st.caption(s["math_note"])
            if cv_data.get("circuit_diagram"):
                st.markdown("**Qiskit Circuit Diagram:**")
                st.code(cv_data["circuit_diagram"], language="text")
            if cv_data.get("num_qubits"):
                st.caption(
                    f"Circuit: {cv_data['num_qubits']}-qubit Hadamard layer. "
                    "Note: Aggregated shot counts do not preserve chronological order of individual measurements."
                )

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-box">
    <b>Scientific Limitations:</b>
    This application implements a <i>selected subset</i> of 5 NIST SP 800-22 tests (not the full 15-test suite). 
    Passing statistical tests does NOT prove true randomness or cryptographic security. 
    This application is NOT certified as a cryptographically secure random number generator. 
    Shannon entropy and min-entropy are sample descriptive statistics.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2: GENERATE BITS
# ─────────────────────────────────────────────

elif page == "⚡ Generate Bits":
    st.markdown('<div class="section-header">Generate Quantum Random Bits</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        source_map = {
            "🔵 Ideal Simulator (Qiskit Aer, noiseless)": "ideal_simulator",
            "🟡 Noisy Simulator (readout-error model)": "noisy_simulator",
        }
        source_label = st.selectbox(
            "Source Mode",
            list(source_map.keys()),
            help="Ideal: perfect Hadamard simulation. Noisy: simulated readout error. Hardware requires IBM credentials."
        )
        source_type = source_map[source_label]

        num_bits = st.slider(
            "Number of Bits (Shots)",
            min_value=100, max_value=100000, value=10000, step=100,
        )

        num_qubits = st.select_slider(
            "Qubit Count per Shot",
            options=[1, 2, 4, 8],
            value=1,
            help="Bits from all qubits in each shot are concatenated. Larger counts improve throughput but do NOT guarantee independent randomness between qubits."
        )
        if num_qubits > 1:
            st.info(
                f"ℹ Using {num_qubits} qubits: each shot yields {num_qubits} bits. "
                "Aggregated counts reflect multi-qubit register measurement outcomes."
            )

        use_seed = st.checkbox("Use reproducible seed", value=False)
        seed = None
        if use_seed:
            seed = st.number_input("Seed value", min_value=0, max_value=2**31 - 1, value=42)
            st.markdown("""
            <div class="warning-box">
            ⚠ Seeded simulation is <b>reproducible but NOT random</b> in the cryptographic sense.
            </div>""", unsafe_allow_html=True)

    with col2:
        noise_params = {}
        if source_type == "noisy_simulator":
            st.markdown("**Readout Error Configuration**")
            p01 = st.slider("P(measure 0 | qubit in |1⟩)", 0.0, 0.5, 0.08, 0.01)
            p10 = st.slider("P(measure 1 | qubit in |0⟩)", 0.0, 0.5, 0.01, 0.01)
            noise_params = {"prob_meas0_prep1": p01, "prob_meas1_prep0": p10}
            st.markdown("""
            <div class="info-box">
            <b>Physical Asymmetric Noise:</b> Superconducting qubits exhibit energy relaxation ($T_1$) 
            where $|1\\rangle \\to |0\\rangle$ occurs more frequently than thermal excitation ($|0\\rangle \\to |1\\rangle$) 
            at cryogenic temperatures ($P(0|1) > P(1|0)$). This creates realistic bit bias, which can then 
            be removed using Von Neumann debiasing.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <b>Ideal Simulation:</b> Perfect Hadamard gate with no noise.<br>
            P(0) = P(1) = 0.5 exactly in expectation.<br>
            Statistical fluctuations are expected in finite samples.
            </div>""", unsafe_allow_html=True)

        st.markdown("**Quantum Circuit**")
        st.code(
            f"q[0..{num_qubits-1}] = |0⟩\nH gate layer → (|0⟩ + |1⟩)/√2\nMeasure → classical register",
            language="text"
        )

    st.markdown("---")

    if st.button("⚡ Generate Random Bits", use_container_width=True, type="primary"):
        payload = {
            "source_type": source_type,
            "num_bits": num_bits,
            "num_qubits": num_qubits,
            "seed": seed if use_seed else None,
        }
        if source_type == "noisy_simulator" and noise_params:
            payload["noise_config"] = noise_params

        with st.spinner(f"Running quantum simulation ({num_bits:,} shots, {num_qubits}q)…"):
            result = api_post("/experiments/generate", payload)

        if "_error" in result:
            st.error(f"❌ Generation failed: {result['_error']}")
        else:
            _sset("last_experiment", result)
            _sset("last_experiment_id", result["experiment_id"])

            st.markdown(
                source_badge_html(result["source_type"]),
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="disclaimer-box">{result.get("source_classification_note", "")}</div>',
                unsafe_allow_html=True
            )

            # Metrics row in pastel cards
            stats = result.get("stats", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Bits", f"{result['actual_bits']:,}")
            c2.metric("Zeros", f"{stats.get('zeros', 0):,}")
            c3.metric("Ones", f"{stats.get('ones', 0):,}")
            c4.metric("Proportion 1s", f"{stats.get('proportion_ones', 0):.4f}")
            c5.metric("Bias |p-0.5|", f"{stats.get('bias_deviation', 0):.4f}")

            c6, c7, c8, c9 = st.columns(4)
            c6.metric("Shannon Entropy", f"{stats.get('shannon_entropy_per_bit', 0):.6f}")
            c7.metric("Empirical H_min", f"{stats.get('min_entropy_per_bit', 0):.6f}")
            c8.metric("Exec. Time", f"{result.get('execution_time_seconds', 0):.4f}s")
            c9.metric("Qubits / Seeded", f"{result.get('num_qubits', 1)} / {'Yes' if result.get('is_seeded') else 'No'}")

            # Bitstream preview
            st.markdown("**Bitstream Preview** (first 400 chars)")
            preview = format_bits_preview(result.get("bitstream_preview", ""), 400)
            st.markdown(
                f'<div class="bitstream-box">{preview}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Full stream: {result.get('bitstream_length', 0):,} bits")

            for w in result.get("warnings", []):
                st.markdown(f'<div class="warning-box">⚠ {w}</div>', unsafe_allow_html=True)

            st.success(f"✓ Experiment saved as ID: `{result['experiment_id']}`")

            # Downloads
            eid = result["experiment_id"]
            dl1, dl2 = st.columns(2)
            with dl1:
                txt_data = api_get_bytes(f"/experiments/{eid}/export", {"format": "txt"})
                if txt_data:
                    st.download_button("⬇ Download Bits (.txt)", txt_data, f"qrng_{eid[:8]}.txt", "text/plain")
            with dl2:
                json_data = api_get_bytes(f"/experiments/{eid}/export", {"format": "json"})
                if json_data:
                    st.download_button("⬇ Download Metadata (.json)", json_data, f"qrng_{eid[:8]}.json", "application/json")

    elif _sget("last_experiment"):
        st.info("Previous experiment loaded. Click Generate to run a new one.")
        result = _sget("last_experiment")
        st.markdown(source_badge_html(result.get("source_type", "")), unsafe_allow_html=True)
        stats = result.get("stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Bits", f"{result['actual_bits']:,}")
        c2.metric("Proportion 1s", f"{stats.get('proportion_ones', 0):.4f}")
        c3.metric("Bias", f"{stats.get('bias_deviation', 0):.4f}")
        c4.metric("Entropy", f"{stats.get('shannon_entropy_per_bit', 0):.4f}")


# ─────────────────────────────────────────────
# PAGE 3: BITSTREAM ANALYSIS
# ─────────────────────────────────────────────

elif page == "📊 Bitstream Analysis":
    st.markdown('<div class="section-header">Bitstream Analysis</div>', unsafe_allow_html=True)

    exps = api_get("/experiments", {"page": 1, "page_size": 50})
    if "_error" in exps or not exps.get("experiments"):
        st.warning("No experiments found. Generate bits first.")
        st.stop()

    exp_options = {
        f"{e['experiment_id'][:8]}… | {e['source_type']} | {e['actual_bits']:,} bits | {e['created_at'][:16]}": e['experiment_id']
        for e in exps["experiments"]
    }
    selected_label = st.selectbox("Select Experiment", list(exp_options.keys()))
    eid = exp_options[selected_label]

    exp = api_get(f"/experiments/{eid}")
    if "_error" in exp:
        st.error(exp["_error"])
        st.stop()

    st.markdown(source_badge_html(exp.get("source_type", "")), unsafe_allow_html=True)

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bits", f"{exp.get('actual_bits', 0):,}")
    c2.metric("Zeros", f"{exp.get('zeros', 0):,}")
    c3.metric("Ones", f"{exp.get('ones', 0):,}")
    c4.metric("Proportion 1s", f"{exp.get('proportion_ones', 0):.6f}")
    c5.metric("Bias |p-0.5|", f"{exp.get('bias_deviation', 0):.6f}")

    st.markdown("---")

    # Binary distribution bar chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Binary Distribution")
        fig = go.Figure(data=[
            go.Bar(
                x=["0 (Zero)", "1 (One)"],
                y=[exp.get("zeros", 0), exp.get("ones", 0)],
                marker_color=[PALETTE["chart_zeros"], PALETTE["chart_ones"]],
                text=[f"{exp.get('zeros', 0):,}", f"{exp.get('ones', 0):,}"],
                textposition="outside",
            )
        ])
        fig.add_hline(
            y=(exp.get("actual_bits", 0) or 1) / 2,
            line_dash="dash", line_color=PALETTE["chart_expected"],
            annotation_text="Expected (n/2)",
            annotation_position="right",
        )
        dist_layout = get_plotly_layout(height=280, title="Zero vs. One Distribution")
        fig.update_layout(**dist_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Statistics Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | **Total bits** | {exp.get('actual_bits', 0):,} |
        | **Zeros** | {exp.get('zeros', 0):,} ({(exp.get('zeros', 0) or 0) / max(exp.get('actual_bits', 1), 1) * 100:.2f}%) |
        | **Ones** | {exp.get('ones', 0):,} ({(exp.get('ones', 0) or 0) / max(exp.get('actual_bits', 1), 1) * 100:.2f}%) |
        | **Proportion of 1s** | {exp.get('proportion_ones', 0):.6f} |
        | **Bias from 0.5** | {exp.get('bias_deviation', 0):.6f} |
        | **Shannon Entropy** | {exp.get('shannon_entropy', 0):.6f} bits/symbol |
        | **Empirical $H_{{min}}$** | {exp.get('min_entropy', 0):.6f} bits/symbol |
        | **Backend** | {exp.get('backend_name', 'N/A')} |
        | **Qubits** | {exp.get('num_qubits', 1)} |
        | **Execution time** | {exp.get('execution_time_seconds', 0):.4f}s |
        """)

        st.markdown("""
        <div class="disclaimer-box">
        ⚠ Shannon entropy and empirical min-entropy are <i>sample descriptive statistics</i>.
        They do NOT certify cryptographic entropy or prove true physical randomness.
        </div>""", unsafe_allow_html=True)

    # ── FIXED SLIDING WINDOW PROPORTION CHART ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Sliding Window Proportion of 1s")
    st.caption("Calculates local proportion P(1) = count(1) / W across sliding segments of the bitstream.")

    bitstream_data = exp.get("bitstream") or exp.get("bitstream_preview", "")

    if exp.get("num_qubits", 1) > 1:
        st.info(
            f"ℹ Note on Multi-Qubit ({exp.get('num_qubits')}q) Generation: "
            "Simulated bits are concatenated across qubits per shot. "
            "Aggregated counts reflect joint register measurement samples."
        )

    if not bitstream_data:
        st.warning("⚠ Chronological bitstream data is unavailable for this experiment. Cannot compute sliding window analysis.")
    elif len(bitstream_data) < 10:
        st.info("Sliding window analysis requires at least 10 bits in the bitstream.")
    else:
        max_win = min(len(bitstream_data), 500)
        default_win = min(max_win, 40)
        sw_c1, sw_c2 = st.columns(2)
        with sw_c1:
            window = st.slider("Window Size W (bits)", min_value=5, max_value=max_win, value=default_win, step=5)
        with sw_c2:
            step_choice = st.slider("Step Size (bits)", min_value=1, max_value=max(1, window), value=max(1, window // 2))

        try:
            sw_res = calculate_sliding_window_proportions(bitstream_data, window_size=window, step_size=step_choice)
            if sw_res["valid"] and sw_res["positions"]:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=sw_res["positions"],
                    y=sw_res["proportions"],
                    mode="lines+markers",
                    line=dict(color=PALETTE["primary"], width=2),
                    marker=dict(size=4, color=PALETTE["dark"]),
                    customdata=list(zip(sw_res["positions"], sw_res["ones_counts"], sw_res["proportions"])),
                    hovertemplate=(
                        "<b>Window Start:</b> %{customdata[0]}<br>"
                        f"<b>Ones Count:</b> %{{customdata[1]}} / {window}<br>"
                        "<b>Proportion 1s:</b> %{customdata[2]:.4f}<extra></extra>"
                    ),
                    name="Window Proportion P(1)",
                ))
                fig2.add_hline(
                    y=0.5, line_dash="dash", line_color=PALETTE["chart_expected"],
                    annotation_text="0.5 ideal", annotation_position="top right"
                )
                sw_layout = get_plotly_layout(height=260, title=f"Sliding Window Proportion of 1s (W={window})")
                sw_layout["yaxis"]["range"] = [0, 1]
                sw_layout["xaxis"]["title"] = "Bitstream Start Position"
                sw_layout["yaxis"]["title"] = "Proportion of 1s"
                fig2.update_layout(**sw_layout)
                st.plotly_chart(fig2, use_container_width=True)

                st.caption(
                    f"Analyzed {len(bitstream_data):,} bits across {sw_res['total_windows']} windows. "
                    f"Mean local proportion: {sw_res['mean_proportion']:.4f}."
                )
            else:
                st.warning("Could not calculate sliding window proportions.")
        except Exception as e:
            st.warning(f"Sliding window analysis warning: {e}")

    # ── Entropy Quality Analyzer ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔢 Quantum Entropy Quality Analyzer")
    st.markdown("""
    <div class="info-box">
    <b>Entropy Metrics (Empirical Estimates Only):</b><br>
    <b>Shannon Entropy $H(X)$</b>: Average uncertainty per bit $= -p_0\\log_2(p_0) - p_1\\log_2(p_1)$. Max = 1.0 for balanced bits.<br>
    <b>Empirical Min-Entropy $H_{min}$</b>: $H_{min} = -\\log_2(\\max(p_0, p_1))$ — conservative lower bound on randomness quality.<br>
    <b>Important:</b> These are sample descriptive statistics, NOT certified physical entropy lower bounds.
    Do NOT use for cryptographic security decisions.
    </div>""", unsafe_allow_html=True)

    ea_col1, ea_col2 = st.columns([1, 3])
    with ea_col1:
        window_size = st.number_input(
            "Window Size (bits)", min_value=50, max_value=2000, value=200, step=50,
            help="Number of bits per entropy analysis window"
        )
        if st.button("🔢 Run Entropy Analysis", type="primary", key="btn_entropy"):
            with st.spinner("Analysing bitstream entropy…"):
                ea_res = api_post(f"/experiments/{eid}/entropy-analysis", {"window_size": window_size})
            _sset("entropy_analysis", (eid, ea_res))

    ea_state = _sget("entropy_analysis")
    if ea_state and ea_state[0] == eid and "_error" not in ea_state[1]:
        ea_res = ea_state[1]
        with ea_col2:
            profile = ea_res.get("profile", {})
            wa = ea_res.get("window_analysis", {})
            comp = ea_res.get("debiased_comparison")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shannon H (raw)", f"{profile.get('shannon_entropy', 0):.6f}")
            m2.metric("H_min (raw)", f"{profile.get('min_entropy', 0):.6f}")
            m3.metric("Proportion 0s", f"{profile.get('proportion_zeros', 0):.4f}")
            m4.metric("Proportion 1s", f"{profile.get('proportion_ones', 0):.4f}")

            windows = wa.get("windows", [])
            if windows:
                fig_ea = go.Figure()
                win_x = [w["window_index"] for w in windows]
                fig_ea.add_trace(go.Scatter(
                    x=win_x,
                    y=[w.get("shannon_entropy", 0) for w in windows],
                    mode="lines+markers", name="Shannon Entropy",
                    line=dict(color=PALETTE["chart_zeros"], width=2),
                ))
                fig_ea.add_trace(go.Scatter(
                    x=win_x,
                    y=[w.get("min_entropy", 0) for w in windows],
                    mode="lines+markers", name="Min-Entropy",
                    line=dict(color=PALETTE["chart_expected"], width=2, dash="dash"),
                ))
                fig_ea.add_hline(y=1.0, line_dash="dot", line_color=PALETTE["success"], annotation_text="Max (1.0)")
                ea_layout = get_plotly_layout(height=280, title=f"Window Entropy ({wa.get('window_size', window_size)}-bit windows)")
                ea_layout["yaxis"]["range"] = [0, 1.1]
                ea_layout["xaxis"]["title"] = "Window Index"
                ea_layout["yaxis"]["title"] = "Entropy (bits/symbol)"
                fig_ea.update_layout(**ea_layout)
                st.plotly_chart(fig_ea, use_container_width=True)
                st.caption(
                    f"Num windows: {wa.get('num_windows', 0)} | "
                    f"Mean H: {wa.get('mean_shannon_entropy', 0):.4f} | "
                    f"Std H: {wa.get('std_shannon_entropy', 0):.4f} | "
                    f"Mean H_min: {wa.get('mean_min_entropy', 0):.4f}"
                )

            if comp:
                st.markdown("#### Raw vs. Debiased Entropy Comparison")
                comp_df = pd.DataFrame([
                    {"Stream": "Raw Input", "Shannon H": f"{comp.get('raw_shannon_entropy', 0):.6f}",
                     "Min-Entropy": f"{comp.get('raw_min_entropy', 0):.6f}",
                     "Total Bits": f"{comp.get('raw_bits', 0):,}"},
                    {"Stream": "Debiased Output", "Shannon H": f"{comp.get('debiased_shannon_entropy', 0):.6f}",
                     "Min-Entropy": f"{comp.get('debiased_min_entropy', 0):.6f}",
                     "Total Bits": f"{comp.get('debiased_bits', 0):,}"},
                ])
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
                st.caption(
                    f"Output-to-input ratio: {comp.get('output_to_input_ratio', 0):.4f} "
                    f"({comp.get('output_to_input_ratio', 0)*100:.2f}%)"
                )
    elif ea_state and ea_state[0] == eid and "_error" in ea_state[1]:
        st.error(f"Entropy analysis failed: {ea_state[1]['_error']}")


# ─────────────────────────────────────────────
# PAGE 4: DEBIASING
# ─────────────────────────────────────────────

elif page == "🔀 Debiasing":
    st.markdown('<div class="section-header">Von Neumann Debiasing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Von Neumann Extractor Algorithm:</b><br>
    Process non-overlapping consecutive pairs: <code>01→0</code>, <code>10→1</code>, discard <code>00</code> and <code>11</code>.<br>
    Assumes i.i.d. input bits. Does NOT remove arbitrary correlations or prove randomness.
    Reduces available bit count substantially.
    </div>""", unsafe_allow_html=True)

    exps = api_get("/experiments", {"page": 1, "page_size": 50})
    if "_error" in exps or not exps.get("experiments"):
        st.warning("No experiments found. Generate bits first.")
        st.stop()

    exp_options = {
        f"{e['experiment_id'][:8]}… | {e['source_type']} | {e['actual_bits']:,} bits": e['experiment_id']
        for e in exps["experiments"]
    }
    selected_label = st.selectbox("Select Experiment to Debias", list(exp_options.keys()))
    eid = exp_options[selected_label]

    if st.button("🔀 Apply Von Neumann Debiasing", type="primary"):
        with st.spinner("Running Von Neumann extraction…"):
            result = api_post(f"/experiments/{eid}/debias", {})

        if "_error" in result:
            st.error(f"❌ Debiasing failed: {result['_error']}")
        else:
            _sset("last_debias", result)
            _sset("last_debias_eid", eid)

            st.success(f"✓ Debiasing complete! Output: {result.get('output_bits', 0):,} bits")

            pc = result.get("pair_counts", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input Bits", f"{result.get('input_bits', 0):,}")
            c2.metric("Output Bits", f"{result.get('output_bits', 0):,}")
            c3.metric("Output/Input Ratio", f"{result.get('output_to_input_ratio', result.get('extraction_efficiency', 0))*100:.2f}%",
                      help="output_bits / input_bits (expected ~25% for balanced iid)")
            c4.metric("Pair Retention", f"{result.get('pair_retention_ratio', 0)*100:.2f}%",
                      help="output_bits / complete_pairs (expected ~50% for balanced iid)")

            c5, c6, c7 = st.columns(3)
            c5.metric("Discarded Bits", f"{result.get('pair_counts', {}).get('discarded_bits', 0):,}")
            c6.metric("Min-Entropy (output)", f"{result.get('output_min_entropy', result.get('output_stats', {}).get('min_entropy_per_bit', 0) if result.get('output_stats') else 0):.6f}")
            c7.metric("Discarded Pairs", f"{pc.get('discarded_pairs', 0):,}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Pair Category Counts")
                fig = go.Figure(data=[
                    go.Bar(
                        x=["00 (discard)", "01 → 0", "10 → 1", "11 (discard)"],
                        y=[pc.get("count_00", 0), pc.get("count_01", 0),
                           pc.get("count_10", 0), pc.get("count_11", 0)],
                        marker_color=[PALETTE["error"], PALETTE["success"], PALETTE["success"], PALETTE["error"]],
                    )
                ])
                fig.update_layout(**get_plotly_layout(height=280, title="Pair Category Counts"))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Before vs. After Comparison")
                in_stats = result.get("input_stats", {})
                out_stats = result.get("output_stats", {})
                df = pd.DataFrame({
                    "Metric": ["Proportion 1s", "Bias |p-0.5|", "Shannon Entropy", "Min-Entropy (H_min)", "Total Bits"],
                    "Input (Raw)": [
                        f"{in_stats.get('proportion_ones', 0):.6f}",
                        f"{in_stats.get('bias_deviation', 0):.6f}",
                        f"{in_stats.get('shannon_entropy_per_bit', 0):.6f}",
                        f"{in_stats.get('min_entropy_per_bit', 0):.6f}",
                        f"{in_stats.get('total_bits', 0):,}",
                    ],
                    "Output (Debiased)": [
                        f"{out_stats.get('proportion_ones', 0):.6f}" if out_stats else "N/A",
                        f"{out_stats.get('bias_deviation', 0):.6f}" if out_stats else "N/A",
                        f"{out_stats.get('shannon_entropy_per_bit', 0):.6f}" if out_stats else "N/A",
                        f"{out_stats.get('min_entropy_per_bit', 0):.6f}" if out_stats else "N/A",
                        f"{out_stats.get('total_bits', 0):,}" if out_stats else "0",
                    ],
                })
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.caption(
                    "Efficiency definitions: "
                    "output_to_input_ratio = output_bits / input_bits (expected ~25% for balanced iid). "
                    "pair_retention_ratio = output_bits / complete_pairs (expected ~50%)."
                )

            preview = result.get("output_bitstream_preview", "")
            if preview:
                st.markdown("**Debiased Bitstream Preview**")
                st.markdown(
                    f'<div class="bitstream-box">{format_bits_preview(preview, 400)}</div>',
                    unsafe_allow_html=True
                )

            for w in result.get("warnings", []):
                st.markdown(f'<div class="warning-box">⚠ {w}</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="disclaimer-box">
            {result.get("algorithm_note", "")}
            </div>""", unsafe_allow_html=True)

    elif _sget("last_debias"):
        st.info("Last debiasing result available below.")
        result = _sget("last_debias")
        pc = result.get("pair_counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Output Bits", f"{result.get('output_bits', 0):,}")
        c2.metric("Output/Input Ratio", f"{result.get('output_to_input_ratio', result.get('extraction_efficiency', 0))*100:.2f}%")
        c3.metric("Pair Retention", f"{result.get('pair_retention_ratio', 0)*100:.2f}%")
        c4.metric("Discarded", f"{pc.get('discarded_percentage', 0):.1f}%")


# ─────────────────────────────────────────────
# PAGE 5: NIST TESTS
# ─────────────────────────────────────────────

elif page == "🧪 NIST Tests":
    st.markdown('<div class="section-header">Selected NIST SP 800-22 Statistical Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    <b>IMPORTANT:</b> This implements 5 of the 15 NIST SP 800-22 tests (Frequency, Block Frequency, 
    Runs, Longest Run, Spectral). This is NOT the complete NIST suite. Passing tests does NOT prove 
    randomness or cryptographic security. A p-value is NOT the probability that data is random.
    </div>""", unsafe_allow_html=True)

    exps = api_get("/experiments", {"page": 1, "page_size": 50})
    if "_error" in exps or not exps.get("experiments"):
        st.warning("No experiments found. Generate bits first.")
        st.stop()

    exp_options = {
        f"{e['experiment_id'][:8]}… | {e['source_type']} | {e['actual_bits']:,} bits | Debiased: {'✓' if e['has_debiased'] else '✗'}": e['experiment_id']
        for e in exps["experiments"]
    }
    selected_label = st.selectbox("Select Experiment", list(exp_options.keys()))
    eid = exp_options[selected_label]

    col1, col2 = st.columns(2)
    with col1:
        test_options = {
            "Frequency (Monobit) Test": "frequency",
            "Block Frequency Test": "block_frequency",
            "Runs Test": "runs",
            "Longest Run of Ones in a Block": "longest_run",
            "Discrete Fourier Transform (Spectral) Test": "spectral",
        }
        selected_tests = st.multiselect(
            "Select Tests to Run",
            list(test_options.keys()),
            default=list(test_options.keys()),
        )
        use_debiased = st.checkbox(
            "Run on debiased output (requires debiasing step first)",
            value=False,
        )

    with col2:
        alpha = st.select_slider(
            "Significance level α",
            options=[0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01,
        )
        block_size = st.number_input(
            "Block size (for Block Frequency test)", min_value=8, max_value=1000, value=128
        )
        st.info(f"Tests will fail if p-value < {alpha}")

    if st.button("🧪 Run Statistical Tests", type="primary", use_container_width=True):
        if not selected_tests:
            st.warning("Select at least one test.")
        else:
            test_keys = [test_options[t] for t in selected_tests]
            payload = {
                "tests": test_keys,
                "use_debiased": use_debiased,
                "block_size": block_size,
                "significance_level": alpha,
            }
            with st.spinner("Running statistical tests…"):
                result = api_post(f"/experiments/{eid}/tests", payload)

            if "_error" in result:
                st.error(f"❌ Test failed: {result['_error']}")
            else:
                _sset("last_test_result", result)

                passed = result.get("passed", 0)
                failed = result.get("failed", 0)
                not_run = result.get("not_run", 0)

                c1, c2, c3 = st.columns(3)
                c1.metric("✓ Passed", passed)
                c2.metric("✗ Failed", failed)
                c3.metric("— Not Run / Insufficient", not_run)

                st.markdown("#### Test Results")
                for r in result.get("results", []):
                    with st.expander(
                        f"{outcome_badge_html(r['outcome'])} &nbsp; {r['test_name']}",
                        expanded=True,
                    ):
                        if r.get("statistic") is not None:
                            rc1, rc2, rc3, rc4 = st.columns(4)
                            rc1.metric("Test Statistic", f"{r['statistic']:.4f}")
                            rc2.metric("p-value", f"{r['p_value']:.4f}" if r.get("p_value") is not None else "N/A")
                            rc3.metric("Threshold α", f"{r['significance_level']}")
                            rc4.metric("Bits Used", f"{r['actual_bits_used']:,}")

                        st.markdown(f"**Purpose:** {r['test_purpose']}")
                        st.markdown(f"**Interpretation:** {r['interpretation']}")
                        if r.get("warning"):
                            st.markdown(
                                f'<div class="warning-box">⚠ {r["warning"]}</div>',
                                unsafe_allow_html=True
                            )

                st.markdown(f"""
                <div class="disclaimer-box">
                {result.get("summary_note", "")}
                </div>""", unsafe_allow_html=True)

                import io, csv as _csv
                csv_buf = io.StringIO()
                w = _csv.writer(csv_buf)
                w.writerow(["Test", "Outcome", "Statistic", "p-value", "alpha", "Bits Used", "Interpretation"])
                for r in result.get("results", []):
                    w.writerow([
                        r["test_name"], r["outcome"],
                        r.get("statistic", ""), r.get("p_value", ""),
                        r["significance_level"], r["actual_bits_used"],
                        r["interpretation"][:100],
                    ])
                st.download_button(
                    "⬇ Download Results (CSV)",
                    csv_buf.getvalue().encode(),
                    "nist_results.csv",
                    "text/csv",
                )


# ─────────────────────────────────────────────
# PAGE 6: EXPERIMENT COMPARISON
# ─────────────────────────────────────────────

elif page == "⚖️ Experiment Comparison":
    st.markdown('<div class="section-header">Experiment Comparison</div>', unsafe_allow_html=True)

    exps = api_get("/experiments", {"page": 1, "page_size": 50})
    if "_error" in exps or len(exps.get("experiments", [])) < 2:
        st.warning("Need at least 2 experiments to compare. Generate more experiments first.")
        st.stop()

    exp_options = {
        f"{e['experiment_id'][:8]}… | {e['source_type']} | {e['actual_bits']:,} bits": e['experiment_id']
        for e in exps["experiments"]
    }
    selected_labels = st.multiselect(
        "Select 2–5 Experiments to Compare",
        list(exp_options.keys()),
        max_selections=5,
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 experiments.")
    elif st.button("⚖️ Compare Experiments", type="primary"):
        eids = [exp_options[l] for l in selected_labels]
        result = api_post("/experiments/compare", {"experiment_ids": eids})

        if "_error" in result:
            st.error(result["_error"])
        else:
            for w in result.get("warnings", []):
                st.markdown(f'<div class="warning-box">⚠ {w}</div>', unsafe_allow_html=True)

            entries = result.get("entries", [])
            if entries:
                df = pd.DataFrame([{
                    "ID": e["experiment_id"][:8] + "…",
                    "Source": e["source_type"],
                    "Backend": e["backend_name"][:30],
                    "Bits": e["actual_bits"],
                    "P(1s)": f"{e['proportion_ones']:.6f}",
                    "Bias": f"{e['bias_deviation']:.6f}",
                    "Entropy": f"{e['shannon_entropy']:.4f}",
                    "Time(s)": f"{e['execution_time_seconds']:.4f}",
                    "NIST Pass": e.get("nist_passed", "N/A"),
                    "NIST Fail": e.get("nist_failed", "N/A"),
                } for e in entries])
                st.dataframe(df, hide_index=True, use_container_width=True)

                fig = go.Figure()
                palette_cycle = [
                    PALETTE["primary"],
                    PALETTE["dark"],
                    PALETTE["chart_debiased"],
                    PALETTE["chart_noise"],
                    PALETTE["chart_ones"],
                ]
                for idx, e in enumerate(entries):
                    fig.add_trace(go.Bar(
                        name=f"{e['experiment_id'][:8]} ({e['source_type']})",
                        x=["Proportion 1s", "Bias |p-0.5|", "Shannon Entropy"],
                        y=[e["proportion_ones"], e["bias_deviation"], e["shannon_entropy"]],
                        marker_color=palette_cycle[idx % len(palette_cycle)],
                    ))
                fig.add_hline(y=0.5, line_dash="dash", line_color=PALETTE["chart_expected"], annotation_text="P(1s)=0.5 ideal")
                cmp_layout = get_plotly_layout(height=340, title="Multi-Experiment Metrics Comparison")
                cmp_layout["barmode"] = "group"
                fig.update_layout(**cmp_layout)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="disclaimer-box">
            {result.get("comparison_note", "")}
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 7: EXPERIMENT HISTORY
# ─────────────────────────────────────────────

elif page == "📁 Experiment History":
    st.markdown('<div class="section-header">Experiment History</div>', unsafe_allow_html=True)

    page_num = st.number_input("Page", min_value=1, value=1)
    exps = api_get("/experiments", {"page": page_num, "page_size": 20})

    if "_error" in exps:
        st.error(exps["_error"])
        st.stop()

    st.metric("Total Saved Experiments", exps.get("total", 0))

    experiments = exps.get("experiments", [])
    if not experiments:
        st.info("No experiments saved yet.")
    else:
        for e in experiments:
            src = e.get("source_type", "")
            icon = {"ideal_simulator": "🔵", "noisy_simulator": "🟡", "hardware": "🔴"}.get(src, "⚪")
            with st.expander(
                f"{icon} {e['experiment_id'][:8]}… | {e['actual_bits']:,} bits | {e['created_at'][:16]} | {src}",
                expanded=False,
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bits", f"{e.get('actual_bits', 0):,}")
                c2.metric("Seeded", "Yes" if e.get("is_seeded") else "No")
                c3.metric("Debiased", "✓" if e.get("has_debiased") else "✗")
                c4.metric("NIST Tests", "✓" if e.get("has_test_results") else "✗")

                eid = e["experiment_id"]

                dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
                with dl_col1:
                    txt = api_get_bytes(f"/experiments/{eid}/export", {"format": "txt"})
                    if txt:
                        st.download_button("⬇ TXT", txt, f"qrng_{eid[:8]}.txt", key=f"txt_{eid}")
                with dl_col2:
                    csv_d = api_get_bytes(f"/experiments/{eid}/export", {"format": "csv"})
                    if csv_d:
                        st.download_button("⬇ CSV", csv_d, f"qrng_{eid[:8]}.csv", key=f"csv_{eid}")
                with dl_col3:
                    json_d = api_get_bytes(f"/experiments/{eid}/export", {"format": "json"})
                    if json_d:
                        st.download_button("⬇ JSON", json_d, f"qrng_{eid[:8]}.json", key=f"json_{eid}")
                with dl_col4:
                    pdf_d = api_get_bytes(f"/experiments/{eid}/export", {"format": "pdf"})
                    if pdf_d:
                        st.download_button("⬇ PDF", pdf_d, f"qrng_{eid[:8]}.pdf", key=f"pdf_{eid}")


# ─────────────────────────────────────────────
# PAGE 8: REPORTS & HELP
# ─────────────────────────────────────────────

elif page == "📄 Reports & Help":
    st.markdown('<div class="section-header">Reports & Help</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Generate Report",
        "🌊 Noise Sweep",
        "📦 Batch Eval",
        "⏱ Benchmarking",
        "🔌 Hardware",
        "📚 Theory",
        "⚙️ Hardware Setup",
        "🔗 References",
    ])

    with tab1:
        st.markdown("### Generate PDF Experiment Report")
        exps = api_get("/experiments", {"page": 1, "page_size": 50})
        if "_error" in exps or not exps.get("experiments"):
            st.warning("No experiments available. Generate bits first.")
        else:
            exp_options = {
                f"{e['experiment_id'][:8]}… | {e['source_type']} | {e['actual_bits']:,} bits": e['experiment_id']
                for e in exps["experiments"]
            }
            sel = st.selectbox("Select Experiment for Report", list(exp_options.keys()))
            eid = exp_options[sel]

            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report via ReportLab…"):
                    pdf = api_get_bytes(f"/experiments/{eid}/export", {"format": "pdf"})
                if pdf:
                    st.download_button("⬇ Download PDF Report", pdf, f"QRNG_Report_{eid[:8]}.pdf", "application/pdf")
                    st.success("PDF generated from real experiment data.")
                else:
                    st.error("PDF generation failed. Ensure ReportLab is installed.")

    with tab2:
        # ── NOISE SWEEP ──────────────────────────────────────────────────────────
        st.markdown("### 🌊 Noise Impact Experiment")
        st.markdown("""
        <div class="info-box">
        Automated noise sweep: generates bitstreams at each readout-error probability level using the
        Qiskit Aer noisy simulator. Records observed bit bias, Shannon entropy, min-entropy,
        and Von Neumann extraction efficiency. All data comes from actual simulations.
        </div>""", unsafe_allow_html=True)

        ns_col1, ns_col2 = st.columns([1, 2])
        with ns_col1:
            ns_levels_opts = {
                "Quick (0%, 5%, 10%)": [0.0, 0.05, 0.10],
                "Standard (0%, 1%, 2%, 5%, 10%, 15%)": [0.0, 0.01, 0.02, 0.05, 0.10, 0.15],
                "Custom": None,
            }
            ns_preset = st.selectbox("Noise Levels Preset", list(ns_levels_opts.keys()), key="ns_preset")
            if ns_levels_opts[ns_preset] is None:
                ns_custom = st.text_input("Custom levels (comma-separated %, e.g. 0,3,7,12)", value="0,5,10", key="ns_custom")
                ns_levels = [float(x.strip()) / 100 for x in ns_custom.split(",") if x.strip()]
            else:
                ns_levels = ns_levels_opts[ns_preset]
            ns_bits = st.slider("Bits per Level", 500, 10000, 2000, 500, key="ns_bits")
            ns_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=99999, value=0, key="ns_seed")

        with ns_col2:
            if st.button("🌊 Run Noise Sweep", type="primary", key="btn_noise"):
                payload_ns = {
                    "noise_levels": ns_levels,
                    "num_bits": ns_bits,
                    "seed": int(ns_seed) if ns_seed > 0 else None,
                }
                with st.spinner(f"Running noise sweep across {len(ns_levels)} levels…"):
                    ns_res = api_post("/experiments/noise-sweep", payload_ns)
                _sset("noise_sweep_res", ns_res)

            ns_res = _sget("noise_sweep_res")
            if ns_res and "_error" not in ns_res:
                pts = ns_res.get("points", [])
                if pts:
                    fig_ns = go.Figure()
                    xs = [p["noise_percentage"] for p in pts]
                    fig_ns.add_trace(go.Scatter(x=xs, y=[p["raw_shannon_entropy"] for p in pts],
                                                mode="lines+markers", name="Raw Shannon H",
                                                line=dict(color=PALETTE["chart_zeros"])))
                    fig_ns.add_trace(go.Scatter(x=xs, y=[p["raw_min_entropy"] for p in pts],
                                                mode="lines+markers", name="Raw Min-Entropy H_min",
                                                line=dict(color=PALETTE["chart_noise"], dash="dash")))
                    fig_ns.add_trace(go.Scatter(x=xs, y=[p["debiased_shannon_entropy"] for p in pts],
                                                mode="lines+markers", name="Debiased Shannon H",
                                                line=dict(color=PALETTE["chart_debiased"])))
                    fig_ns.add_hline(y=1.0, line_dash="dot", line_color=PALETTE["chart_expected"], annotation_text="Max (1.0)")
                    ns_layout = get_plotly_layout(height=280, title="Entropy vs. Readout Error Level")
                    ns_layout["yaxis"]["range"] = [0, 1.1]
                    ns_layout["xaxis"]["title"] = "Readout Error (%)"
                    ns_layout["yaxis"]["title"] = "Entropy (bits/symbol)"
                    fig_ns.update_layout(**ns_layout)
                    st.plotly_chart(fig_ns, use_container_width=True)

                    fig_eff = go.Figure()
                    fig_eff.add_trace(go.Bar(
                        x=xs, y=[p["output_to_input_ratio"] * 100 for p in pts],
                        name="Output/Input Ratio (%)", marker_color=PALETTE["chart_zeros"],
                    ))
                    fig_eff.add_hline(y=25, line_dash="dash", line_color=PALETTE["chart_expected"], annotation_text="Expected ~25%")
                    eff_layout = get_plotly_layout(height=240, title="Extraction Efficiency vs. Readout Error")
                    eff_layout["xaxis"]["title"] = "Readout Error (%)"
                    eff_layout["yaxis"]["title"] = "Output/Input Ratio (%)"
                    fig_eff.update_layout(**eff_layout)
                    st.plotly_chart(fig_eff, use_container_width=True)

                    ns_df = pd.DataFrame([{
                        "Noise %": f"{p['noise_percentage']:.1f}%",
                        "Raw H": f"{p['raw_shannon_entropy']:.4f}",
                        "Raw H_min": f"{p['raw_min_entropy']:.4f}",
                        "Raw Bias": f"{p['raw_bias_deviation']:.4f}",
                        "Debiased Bits": p["debiased_bits"],
                        "Out/In Ratio": f"{p['output_to_input_ratio']*100:.2f}%",
                        "Pair Retention": f"{p['pair_retention_ratio']*100:.2f}%",
                        "Freq Test": p["raw_tests"]["frequency"]["outcome"],
                        "Runs Test": p["raw_tests"]["runs"]["outcome"],
                        "Time (s)": f"{p['execution_time_seconds']:.3f}",
                    } for p in pts])
                    st.dataframe(ns_df, hide_index=True, use_container_width=True)
                    st.caption(ns_res.get("scientific_summary", ""))
            elif ns_res and "_error" in ns_res:
                st.error(f"Noise sweep failed: {ns_res['_error']}")

    with tab3:
        # ── BATCH EVALUATION ─────────────────────────────────────────────────────
        st.markdown("### 📦 Batch Statistical Evaluation")
        st.markdown("""
        <div class="info-box">
        Generates multiple independent bitstreams and evaluates NIST statistical tests on each.
        Computes observed rejection rates, p-value distributions, and confidence intervals.<br>
        <b>Multiple-testing note:</b> Isolated failures are expected by chance at rate α even for ideal generators.
        Assess the overall rejection rate across K sequences, not individual p-values.
        </div>""", unsafe_allow_html=True)

        be_col1, be_col2 = st.columns([1, 2])
        with be_col1:
            be_seqs = st.slider("Number of Sequences", 5, 50, 10, 5, key="be_seqs")
            be_bits = st.slider("Bits per Sequence", 200, 5000, 1000, 100, key="be_bits")
            be_tests = st.multiselect(
                "Tests", ["frequency", "block_frequency", "runs", "longest_run", "spectral"],
                default=["frequency", "runs"], key="be_tests"
            )
            be_alpha = st.select_slider("Significance α", [0.001, 0.005, 0.01, 0.05, 0.10], value=0.01, key="be_alpha")
            be_qubits = st.selectbox("Qubit Count", [1, 2, 4], index=0, key="be_qubits")

        with be_col2:
            if st.button("📦 Run Batch Evaluation", type="primary", key="btn_batch"):
                payload_be = {
                    "num_sequences": be_seqs,
                    "bits_per_sequence": be_bits,
                    "tests": be_tests,
                    "alpha": be_alpha,
                    "num_qubits": be_qubits,
                }
                with st.spinner(f"Evaluating {be_seqs} sequences × {be_bits} bits…"):
                    be_res = api_post("/experiments/batch-evaluate", payload_be)
                _sset("batch_eval_res", be_res)

            be_res = _sget("batch_eval_res")
            if be_res and "_error" not in be_res:
                summaries = be_res.get("test_summaries", {})
                ci = be_res.get("confidence_interval", {})
                st.caption(
                    f"K={be_res['num_sequences']} sequences | "
                    f"Expected pass rate: {ci.get('expected_pass_rate', 0):.3f} "
                    f"[{ci.get('lower_bound', 0):.3f}, {ci.get('upper_bound', 0):.3f}] "
                    f"({ci.get('confidence_level', '')})"
                )

                for t_key, t_sum in summaries.items():
                    with st.expander(f"🧪 {t_key} — Pass: {t_sum['pass_count']}, Fail: {t_sum['fail_count']}", expanded=False):
                        bc1, bc2, bc3 = st.columns(3)
                        bc1.metric("Pass Rate", f"{t_sum['observed_pass_rate']*100:.1f}%",
                                   delta="✓ In CI" if t_sum.get('within_confidence_interval') else "⚠ Outside CI")
                        bc2.metric("Rejection Rate", f"{t_sum['observed_rejection_rate']*100:.1f}%")
                        bc3.metric("Evaluated Seq.", t_sum["evaluated_sequences"])

                        bins = t_sum.get("p_value_histogram_bins", [])
                        if any(b > 0 for b in bins):
                            bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
                            fig_pv = go.Figure(data=[go.Bar(x=bin_labels, y=bins, marker_color=PALETTE["primary"])])
                            fig_pv.add_hline(y=t_sum["evaluated_sequences"] / 10, line_dash="dash",
                                             line_color=PALETTE["chart_expected"], annotation_text="Expected (uniform)")
                            pv_layout = get_plotly_layout(height=200, title="p-value Distribution (10 equal bins)")
                            pv_layout["xaxis"]["title"] = "p-value Range"
                            pv_layout["yaxis"]["title"] = "Count"
                            fig_pv.update_layout(**pv_layout)
                            st.plotly_chart(fig_pv, use_container_width=True)
                            if t_sum.get("uniformity_p_value") is not None:
                                st.caption(
                                    f"Chi-square uniformity test: p={t_sum['uniformity_p_value']:.4f} "
                                    f"({'uniform' if t_sum['p_values_uniform'] else 'non-uniform'})"
                                )

                st.markdown(f"""
                <div class="disclaimer-box">
                {be_res.get('multiple_testing_note', '')}
                </div>""", unsafe_allow_html=True)
            elif be_res and "_error" in be_res:
                st.error(f"Batch evaluation failed: {be_res['_error']}")

    with tab4:
        # ── BENCHMARKING ─────────────────────────────────────────────────────────
        st.markdown("### ⏱ Experiment Benchmarking")
        st.markdown("""
        <div class="info-box">
        Measures actual CPU + Qiskit Aer simulation execution time across the pipeline.
        Reports median, mean, std, and throughput (bits/second) across repeated trials.<br>
        <b>Disclaimer:</b> These benchmarks reflect classical host CPU performance, NOT physical quantum gate times.
        No quantum advantage is claimed.
        </div>""", unsafe_allow_html=True)

        bm_col1, bm_col2 = st.columns([1, 2])
        with bm_col1:
            bm_bits = st.slider("Bits per Trial", 500, 10000, 2000, 500, key="bm_bits")
            bm_trials = st.slider("Number of Trials", 3, 10, 5, 1, key="bm_trials")
            bm_qubits = st.multiselect(
                "Qubit Configs to Benchmark", [1, 2, 4, 8],
                default=[1, 2, 4], key="bm_qubits"
            )

        with bm_col2:
            if st.button("⏱ Run Benchmark", type="primary", key="btn_bm"):
                payload_bm = {
                    "num_bits": bm_bits,
                    "trials": bm_trials,
                    "qubit_configs": bm_qubits if bm_qubits else [1, 2],
                }
                with st.spinner(f"Running {bm_trials} benchmark trials…"):
                    bm_res = api_post("/benchmarks/run", payload_bm)
                _sset("bm_res", bm_res)

            bm_res = _sget("bm_res")
            if bm_res and "_error" not in bm_res:
                stages = bm_res.get("pipeline_stages", {})
                qubit_sc = bm_res.get("qubit_scaling", {})

                thru_labels = []
                thru_vals = []
                for q_key, q_data in qubit_sc.items():
                    if q_data:
                        thru_labels.append(q_key.replace("_", " "))
                        thru_vals.append(q_data.get("throughput_bits_per_second", 0))

                if thru_vals:
                    fig_bm = go.Figure(data=[go.Bar(x=thru_labels, y=thru_vals, marker_color=PALETTE["primary"])])
                    bm_layout = get_plotly_layout(height=250, title="Generation Throughput by Qubit Register Size")
                    bm_layout["xaxis"]["title"] = "Qubit Configuration"
                    bm_layout["yaxis"]["title"] = "Throughput (bits/s)"
                    fig_bm.update_layout(**bm_layout)
                    st.plotly_chart(fig_bm, use_container_width=True)

                stage_rows = []
                stage_name_map = {
                    "generation_1qubit": "Generation (1 qubit)",
                    "debiasing": "Von Neumann Debiasing",
                    "statistical_testing_5tests": "NIST Testing (5 tests)",
                    "pdf_export": "PDF Export",
                }
                for sk, sv in stages.items():
                    if sv:
                        stage_rows.append({
                            "Stage": stage_name_map.get(sk, sk),
                            "Median (s)": f"{sv.get('median_seconds', 0):.5f}",
                            "Mean (s)": f"{sv.get('mean_seconds', 0):.5f}",
                            "Std (s)": f"{sv.get('std_seconds', 0):.5f}",
                            "Min (s)": f"{sv.get('min_seconds', 0):.5f}",
                            "Max (s)": f"{sv.get('max_seconds', 0):.5f}",
                            "Throughput (bit/s)": f"{sv.get('throughput_bits_per_second', 0):.0f}",
                        })
                if stage_rows:
                    st.dataframe(pd.DataFrame(stage_rows), hide_index=True, use_container_width=True)

                st.caption(
                    f"Total benchmark time: {bm_res.get('total_benchmark_time_seconds', 0):.2f}s | "
                    f"Bits tested: {bm_res.get('num_bits_tested', 0):,} | "
                    f"Trials: {bm_res.get('trials_executed', 0)}"
                )
                st.markdown(f"""
                <div class="disclaimer-box">{bm_res.get('disclaimer', '')}</div>""", unsafe_allow_html=True)
            elif bm_res and "_error" in bm_res:
                st.error(f"Benchmarking failed: {bm_res['_error']}")

    with tab5:
        # ── HARDWARE STATUS ───────────────────────────────────────────────────────
        st.markdown("### 🔌 Hardware & Calibration Status")
        hw_status = api_get("/hardware/status")
        if "_error" not in hw_status:
            if hw_status.get("configured"):
                st.success("✓ IBM Quantum credentials are configured.")
                st.json(hw_status)
                st.markdown("---")
                hw_cal = api_get("/hardware/calibration")
                if "_error" not in hw_cal and hw_cal.get("calibration_available"):
                    st.markdown("#### Calibration Data")
                    st.json(hw_cal)
                else:
                    st.info("Calibration data not available (requires active hardware backend connection).")
            else:
                st.info("IBM Quantum hardware is NOT configured. Running in simulator-only mode.")
                st.json(hw_status)
        else:
            st.error(f"Hardware status check failed: {hw_status['_error']}")

    with tab6:
        st.markdown("""
        ### Quantum Theory Background

        #### Hadamard Gate and Measurement
        A qubit initialized in **|0⟩** after a Hadamard gate H becomes:
        $$H|0\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}$$
        Measurement collapses the superposition with $P(0) = P(1) = 0.5$ (ideal case).

        #### Von Neumann Extractor
        Invented by John von Neumann (1951), the extractor reduces bias:
        - Split input into consecutive non-overlapping pairs
        - **01 → 0** (retained)
        - **10 → 1** (retained)
        - **00** → discarded
        - **11** → discarded

        **Efficiency definitions:**
        - `output_to_input_ratio = output_bits / input_bits` — expected **~25%** for balanced iid input
        - `pair_retention_ratio = output_bits / complete_pairs` — expected **~50%** for balanced iid input

        **Assumption:** i.i.d. input bits. **Limitation:** reduces bit count; does not remove correlations.

        #### Entropy Metrics
        - **Shannon Entropy $H(X)$** $= -p_0\\log_2(p_0) - p_1\\log_2(p_1) \\in [0, 1]$. Max = 1 for balanced bits.
        - **Empirical Min-Entropy $H_{min}$** $= -\\log_2(\\max(p_0, p_1))$. Conservative lower bound.
        - These are **sample descriptive statistics only**. NOT certified physical entropy lower bounds. NOT suitable for cryptographic use.

        #### NIST SP 800-22 Tests
        NIST SP 800-22 Rev. 1a defines 15 statistical tests for random number generators.
        This application implements **5 of 15**:
        1. **Frequency (Monobit)**: Proportion of 1s ≈ 0.5?
        2. **Block Frequency**: Proportion within blocks ≈ 0.5?
        3. **Runs**: Number of consecutive identical bits consistent with randomness?
        4. **Longest Run**: Longest run of 1s within blocks consistent?
        5. **Spectral (DFT)**: No periodic patterns in the spectrum?

        **Reference:** https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
        """)

    with tab7:
        st.markdown("""
        ### Optional IBM Quantum Hardware Setup

        **⚠ Hardware is completely optional. The full application works without credentials.**

        #### Step 1: Get IBM Quantum Account
        Create a free account at [quantum.ibm.com](https://quantum.ibm.com).

        #### Step 2: Get Your API Token
        From your IBM Quantum account dashboard, copy your API token.

        **NEVER share your token or commit it to version control.**

        #### Step 3: Create .env File
        Copy `.env.example` to `.env` in the project root:
        ```
        IBM_QUANTUM_TOKEN=your_actual_token_here
        IBM_QUANTUM_INSTANCE=ibm-q/open/main
        IBM_QUANTUM_BACKEND=ibm_brisbane
        ```

        #### Step 4: Restart the Backend
        ```powershell
        # In the qrng-project directory:
        uvicorn backend.app.main:app --reload
        ```

        #### Step 5: Check Status
        The sidebar will show "Hardware: ✓ Configured" when credentials are valid.
        Use **GET /api/v1/hardware/backends** to list available backends.

        #### Important Notes
        - Jobs queue on IBM Quantum; results may take minutes to hours
        - Never submit jobs automatically — always explicitly click Submit
        - Hardware results are labeled with the actual backend name and job ID
        - Simulator output is NEVER substituted for hardware results
        """)

    with tab8:
        st.markdown("""
        ### Official References

        | Resource | Link |
        |----------|------|
        | NIST SP 800-22 Rev. 1a | https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf |
        | Qiskit Documentation | https://docs.quantum.ibm.com/ |
        | Qiskit Aer | https://qiskit.github.io/qiskit-aer/ |
        | IBM Quantum | https://quantum.ibm.com |
        | Von Neumann (1951) | Various techniques used in connection with random digits. NBS Applied Mathematics Series, 12, 36-38 |
        | Nielsen & Chuang | Quantum Computation and Quantum Information. Cambridge University Press, 2010 |

        ### API Documentation
        When the backend is running, visit:
        - **Swagger UI:** http://localhost:8000/docs
        - **ReDoc:** http://localhost:8000/redoc
        - **OpenAPI JSON:** http://localhost:8000/openapi.json
        """)
