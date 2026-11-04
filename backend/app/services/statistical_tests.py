"""
NIST SP 800-22 Statistical Randomness Tests (Selected Subset)
==============================================================

This module implements a selected subset of tests from NIST SP 800-22 Rev. 1a.
Reference: NIST SP 800-22 Rev. 1a, "A Statistical Test Suite for Random and
           Pseudorandom Number Generators for Cryptographic Applications."
           https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

Tests implemented:
    1. Frequency (Monobit) Test         — Section 2.1
    2. Block Frequency Test             — Section 2.2
    3. Runs Test                        — Section 2.3
    4. Longest Run of Ones in a Block   — Section 2.4
    5. Discrete Fourier Transform (Spectral) Test — Section 2.6

IMPORTANT DISCLAIMERS:
    - This is a SELECTED SUBSET, not the complete NIST SP 800-22 suite.
    - Passing these tests does NOT prove true randomness.
    - Passing does NOT establish cryptographic suitability.
    - A p-value is NOT the probability that the data is random.
    - A single failure does NOT automatically prove the generator is defective.
    - A pass does NOT establish security.
    - Tests marked NOT_RUN or INSUFFICIENT_DATA are not counted as passed.
    - Tests are not skipped silently; every run is explicitly labeled.

All test functions return a dict with:
    test_name, test_purpose, outcome, statistic, p_value,
    significance_level, minimum_bits_required, actual_bits_used,
    interpretation, warning.
"""

import logging
import math
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import special as sp_special
from scipy.stats import chi2 as sp_chi2

logger = logging.getLogger(__name__)

IMPLEMENTATION_VERSION = "1.0.0"
DISCLAIMER = (
    "SELECTED NIST SP 800-22 tests. Passing does NOT prove randomness or "
    "cryptographic security. A p-value is not the probability the data is random."
)

# Minimum bit lengths per NIST SP 800-22 guidance
MIN_BITS_FREQUENCY = 100
MIN_BITS_BLOCK_FREQUENCY = 100
MIN_BITS_RUNS = 100
MIN_BITS_LONGEST_RUN = 128
MIN_BITS_SPECTRAL = 1000


def _bits_to_array(bitstream: str) -> np.ndarray:
    """Convert a binary string to a numpy int16 array of 0s and 1s.
    Uses int16 (not uint8) so that the {0,1} → {-1,+1} mapping via
    '2*bits - 1' does not wrap around under unsigned arithmetic."""
    return (np.frombuffer(bitstream.encode(), dtype=np.uint8) - ord("0")).astype(np.int16)


def _make_result(
    test_name: str,
    test_purpose: str,
    outcome: str,
    statistic: Optional[float],
    p_value: Optional[float],
    alpha: float,
    min_bits: int,
    actual_bits: int,
    interpretation: str,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "test_name": test_name,
        "test_purpose": test_purpose,
        "outcome": outcome,
        "statistic": round(statistic, 6) if statistic is not None else None,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "significance_level": alpha,
        "minimum_bits_required": min_bits,
        "actual_bits_used": actual_bits,
        "interpretation": interpretation,
        "warning": warning,
    }


# ─────────────────────────────────────────────
# Test 1: Frequency (Monobit) Test
# NIST SP 800-22 Section 2.1
# ─────────────────────────────────────────────

def frequency_test(bitstream: str, alpha: float = 0.01) -> Dict[str, Any]:
    """
    NIST SP 800-22 Section 2.1 — Frequency (Monobit) Test.

    Purpose: Determine whether the number of 1s is approximately equal to n/2.
    H0: The proportion of 1s is approximately 0.5.
    """
    n = len(bitstream)
    name = "Frequency (Monobit) Test"
    purpose = (
        "Tests whether the proportion of 1s and 0s is approximately equal. "
        "A significantly unequal proportion suggests non-randomness."
    )
    if n < MIN_BITS_FREQUENCY:
        return _make_result(
            name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
            MIN_BITS_FREQUENCY, n,
            f"Requires at least {MIN_BITS_FREQUENCY} bits. Got {n}.",
        )
    try:
        bits = _bits_to_array(bitstream)
        s_n = int(np.sum(bits))          # count of ones
        # Sobs = (sum of +1/-1 mapped bits) / sqrt(n)
        mapped = 2 * bits - 1            # {0,1} → {-1,+1}
        s_obs = abs(float(np.sum(mapped))) / math.sqrt(n)
        p_value = math.erfc(s_obs / math.sqrt(2))

        outcome = "PASS" if p_value >= alpha else "FAIL"
        interpretation = (
            f"Test statistic S_obs = {s_obs:.4f}, p-value = {p_value:.4f}. "
            f"Count of 1s: {s_n} ({s_n/n*100:.2f}%). "
            + ("Insufficient evidence to reject uniformity (PASS)."
               if outcome == "PASS"
               else f"Significantly non-uniform proportion (FAIL at α={alpha}).")
            + " Note: this tests one statistical property only."
        )
        return _make_result(name, purpose, outcome, s_obs, p_value, alpha,
                            MIN_BITS_FREQUENCY, n, interpretation)
    except Exception as exc:
        logger.exception("frequency_test error: %s", exc)
        return _make_result(name, purpose, "ERROR", None, None, alpha,
                            MIN_BITS_FREQUENCY, n,
                            f"Test encountered a numerical error: {exc}")


# ─────────────────────────────────────────────
# Test 2: Block Frequency Test
# NIST SP 800-22 Section 2.2
# ─────────────────────────────────────────────

def block_frequency_test(
    bitstream: str, block_size: int = 128, alpha: float = 0.01
) -> Dict[str, Any]:
    """
    NIST SP 800-22 Section 2.2 — Frequency Test within a Block.

    Purpose: Determine if the proportion of 1s within each non-overlapping
             block of M bits is approximately M/2.
    """
    n = len(bitstream)
    name = "Block Frequency Test"
    purpose = (
        f"Divides the bitstream into non-overlapping blocks of {block_size} bits "
        "and tests whether the proportion of 1s within each block is approximately 0.5."
    )
    min_bits = max(MIN_BITS_BLOCK_FREQUENCY, block_size)
    if n < min_bits:
        return _make_result(
            name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
            min_bits, n,
            f"Requires at least {min_bits} bits for block_size={block_size}.",
        )
    try:
        bits = _bits_to_array(bitstream)
        num_blocks = n // block_size
        chi_sq = 0.0
        for i in range(num_blocks):
            block = bits[i * block_size:(i + 1) * block_size]
            pi_i = float(np.sum(block)) / block_size
            chi_sq += (pi_i - 0.5) ** 2

        chi_sq *= 4 * block_size
        p_value = sp_special.gammaincc(num_blocks / 2.0, chi_sq / 2.0)

        outcome = "PASS" if p_value >= alpha else "FAIL"
        interpretation = (
            f"χ² = {chi_sq:.4f}, {num_blocks} blocks of {block_size} bits, "
            f"p-value = {p_value:.4f}. "
            + ("Block proportions consistent with 0.5 (PASS)."
               if outcome == "PASS"
               else f"Block proportions deviate significantly (FAIL at α={alpha}).")
        )
        return _make_result(name, purpose, outcome, chi_sq, p_value, alpha,
                            min_bits, num_blocks * block_size, interpretation)
    except Exception as exc:
        logger.exception("block_frequency_test error: %s", exc)
        return _make_result(name, purpose, "ERROR", None, None, alpha,
                            min_bits, n, f"Numerical error: {exc}")


# ─────────────────────────────────────────────
# Test 3: Runs Test
# NIST SP 800-22 Section 2.3
# ─────────────────────────────────────────────

def runs_test(bitstream: str, alpha: float = 0.01) -> Dict[str, Any]:
    """
    NIST SP 800-22 Section 2.3 — Runs Test.

    Purpose: Count the number of runs (maximal sequences of identical bits).
             A run is a consecutive sequence of identical bits.
             Too many or too few runs indicate non-randomness.
    """
    n = len(bitstream)
    name = "Runs Test"
    purpose = (
        "Counts runs (maximal consecutive sequences of identical bits). "
        "Excessive or insufficient runs indicate non-random patterns."
    )
    if n < MIN_BITS_RUNS:
        return _make_result(
            name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
            MIN_BITS_RUNS, n,
            f"Requires at least {MIN_BITS_RUNS} bits. Got {n}.",
        )
    try:
        bits = _bits_to_array(bitstream)
        pi = float(np.sum(bits)) / n

        # Pre-test: |pi - 0.5| >= 2/sqrt(n) → fail immediately
        if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
            return _make_result(
                name, purpose, "FAIL", None, 0.0, alpha,
                MIN_BITS_RUNS, n,
                f"Pre-test failed: proportion of 1s = {pi:.4f} is too far from 0.5. "
                "Run the frequency test first.",
                warning="Runs test pre-condition not satisfied; result is FAIL."
            )

        # Count runs
        v_n = 1 + int(np.sum(bits[:-1] != bits[1:]))

        # p-value
        numerator = abs(v_n - 2 * n * pi * (1 - pi))
        denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
        p_value = math.erfc(numerator / denominator)

        outcome = "PASS" if p_value >= alpha else "FAIL"
        interpretation = (
            f"V_n (observed runs) = {v_n}, π = {pi:.4f}, "
            f"p-value = {p_value:.4f}. "
            + ("Number of runs consistent with random sequence (PASS)."
               if outcome == "PASS"
               else f"Run count deviates significantly (FAIL at α={alpha}).")
        )
        return _make_result(name, purpose, outcome, float(v_n), p_value, alpha,
                            MIN_BITS_RUNS, n, interpretation)
    except Exception as exc:
        logger.exception("runs_test error: %s", exc)
        return _make_result(name, purpose, "ERROR", None, None, alpha,
                            MIN_BITS_RUNS, n, f"Numerical error: {exc}")


# ─────────────────────────────────────────────
# Test 4: Longest Run of Ones in a Block
# NIST SP 800-22 Section 2.4
# ─────────────────────────────────────────────

def longest_run_test(bitstream: str, alpha: float = 0.01) -> Dict[str, Any]:
    """
    NIST SP 800-22 Section 2.4 — Longest Run of Ones in a Block.

    Purpose: Detect deviations in the longest run of consecutive 1s within blocks.
    Uses n=128, 512, or ≥ 6272 bits as per NIST table.
    """
    n = len(bitstream)
    name = "Longest Run of Ones in a Block"
    purpose = (
        "Tests whether the longest run of 1s within fixed-size blocks "
        "is consistent with what is expected from a random sequence."
    )
    if n < MIN_BITS_LONGEST_RUN:
        return _make_result(
            name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
            MIN_BITS_LONGEST_RUN, n,
            f"Requires at least {MIN_BITS_LONGEST_RUN} bits. Got {n}.",
        )
    try:
        bits = _bits_to_array(bitstream)

        # Select parameters per NIST SP 800-22 Table 2.4.4
        if n < 6272:
            M = 8
            K = 3
            pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
            thresholds = [1, 2, 3, 4]  # categories ≤1, 2, 3, ≥4
        elif n < 750000:
            M = 128
            K = 5
            pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
            thresholds = [4, 5, 6, 7, 8, 9]
        else:
            M = 10000
            K = 6
            pi_vals = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
            thresholds = [10, 11, 12, 13, 14, 15, 16]

        N = n // M  # number of blocks

        if N < 1:
            return _make_result(
                name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
                MIN_BITS_LONGEST_RUN, n,
                f"Not enough bits to form blocks of size {M}.",
            )

        # Count longest run in each block
        freq = [0] * len(pi_vals)
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            longest = 0
            current = 0
            for b in block:
                if b == 1:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0

            # Categorise longest run
            if longest <= thresholds[0]:
                freq[0] += 1
            elif longest >= thresholds[-1]:
                freq[-1] += 1
            else:
                for j in range(1, len(thresholds) - 1):
                    if longest == thresholds[j]:
                        freq[j] += 1
                        break

        chi_sq = sum(
            (freq[j] - N * pi_vals[j]) ** 2 / (N * pi_vals[j])
            for j in range(len(pi_vals))
        )
        p_value = sp_special.gammaincc(K / 2.0, chi_sq / 2.0)

        outcome = "PASS" if p_value >= alpha else "FAIL"
        interpretation = (
            f"χ² = {chi_sq:.4f}, {N} blocks of {M} bits, "
            f"p-value = {p_value:.4f}. "
            + ("Longest run lengths consistent with random (PASS)."
               if outcome == "PASS"
               else f"Longest run deviates significantly (FAIL at α={alpha}).")
        )
        return _make_result(name, purpose, outcome, chi_sq, p_value, alpha,
                            MIN_BITS_LONGEST_RUN, n, interpretation)
    except Exception as exc:
        logger.exception("longest_run_test error: %s", exc)
        return _make_result(name, purpose, "ERROR", None, None, alpha,
                            MIN_BITS_LONGEST_RUN, n, f"Numerical error: {exc}")


# ─────────────────────────────────────────────
# Test 5: Discrete Fourier Transform (Spectral) Test
# NIST SP 800-22 Section 2.6
# ─────────────────────────────────────────────

def spectral_test(bitstream: str, alpha: float = 0.01) -> Dict[str, Any]:
    """
    NIST SP 800-22 Section 2.6 — Discrete Fourier Transform (Spectral) Test.

    Purpose: Detect periodic features in the sequence that would indicate
             non-randomness via its frequency spectrum.
    Minimum: 1000 bits per NIST guidance.
    """
    n = len(bitstream)
    name = "Discrete Fourier Transform (Spectral) Test"
    purpose = (
        "Applies the DFT to detect periodic patterns in the bitstream. "
        "Random sequences should not exhibit significant spectral peaks."
    )
    if n < MIN_BITS_SPECTRAL:
        return _make_result(
            name, purpose, "INSUFFICIENT_DATA", None, None, alpha,
            MIN_BITS_SPECTRAL, n,
            f"Requires at least {MIN_BITS_SPECTRAL} bits. Got {n}.",
        )
    try:
        bits = _bits_to_array(bitstream).astype(float)
        # Map {0,1} → {-1, +1}
        mapped = 2.0 * bits - 1.0

        # DFT
        X = np.fft.fft(mapped)
        # Take the first n/2 magnitudes (excluding DC and Nyquist if even)
        half = n // 2
        M = np.abs(X[:half])

        # Threshold T = sqrt(log(1/0.05) * n)
        T = math.sqrt(math.log(1.0 / 0.05) * n)

        # Count peaks exceeding threshold
        N0 = 0.95 * half  # expected number below threshold
        N1 = float(np.sum(M < T))  # observed number below threshold

        d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
        p_value = math.erfc(abs(d) / math.sqrt(2))

        outcome = "PASS" if p_value >= alpha else "FAIL"
        interpretation = (
            f"T = {T:.4f}, N0 (expected) = {N0:.1f}, N1 (observed below T) = {N1:.0f}, "
            f"d = {d:.4f}, p-value = {p_value:.4f}. "
            + ("No significant periodic pattern detected (PASS)."
               if outcome == "PASS"
               else f"Periodic features detected (FAIL at α={alpha}).")
        )
        return _make_result(name, purpose, outcome, d, p_value, alpha,
                            MIN_BITS_SPECTRAL, n, interpretation)
    except Exception as exc:
        logger.exception("spectral_test error: %s", exc)
        return _make_result(name, purpose, "ERROR", None, None, alpha,
                            MIN_BITS_SPECTRAL, n, f"Numerical error: {exc}")


# ─────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────

AVAILABLE_TESTS = {
    "frequency": frequency_test,
    "block_frequency": block_frequency_test,
    "runs": runs_test,
    "longest_run": longest_run_test,
    "spectral": spectral_test,
}

ALL_TEST_NAMES = list(AVAILABLE_TESTS.keys())


def run_selected_tests(
    bitstream: str,
    test_names: List[str],
    alpha: float = 0.01,
    block_size: int = 128,
) -> Dict[str, Any]:
    """
    Run the requested NIST tests and return a summary.

    Args:
        bitstream: Binary string of '0' and '1'.
        test_names: List of test keys from AVAILABLE_TESTS.
        alpha: Significance level (default 0.01).
        block_size: Block size for block_frequency test.

    Returns:
        dict with test_run_id, results list, counts, and summary note.
    """
    # Validate bitstream
    if not all(c in "01" for c in bitstream):
        raise ValueError("Bitstream contains characters other than '0' and '1'.")

    n = len(bitstream)
    results = []
    passed = failed = not_run = 0

    for name in test_names:
        fn = AVAILABLE_TESTS.get(name)
        if fn is None:
            results.append(_make_result(
                name, "Unknown test", "NOT_RUN", None, None, alpha, 0, n,
                f"Test '{name}' is not implemented in this version.",
                warning=f"Unrecognised test name: '{name}'."
            ))
            not_run += 1
            continue

        kwargs = {"alpha": alpha}
        if name == "block_frequency":
            kwargs["block_size"] = block_size

        result = fn(bitstream, **kwargs)
        results.append(result)

        if result["outcome"] in ("PASS",):
            passed += 1
        elif result["outcome"] in ("FAIL", "ERROR"):
            failed += 1
        else:
            not_run += 1

    multi_test_note = (
        f"Multiple tests run (n={len(test_names)}). "
        "With α={alpha}, roughly {exp_fail:.0f} test(s) may fail by chance alone "
        "(multiple-comparison effect). "
        "A p-value is NOT the probability that the data is random. "
        "Passing these tests does NOT prove randomness or cryptographic security."
    ).format(alpha=alpha, exp_fail=len(test_names) * alpha)

    return {
        "test_run_id": str(uuid.uuid4()),
        "bits_analyzed": n,
        "significance_level": alpha,
        "results": results,
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
        "implementation_note": (
            "Selected NIST SP 800-22 tests (5 of 15). "
            "This is NOT the complete NIST SP 800-22 suite."
        ),
        "summary_note": (
            f"{passed} passed, {failed} failed, {not_run} not run / insufficient data. "
            + multi_test_note
        ),
        "disclaimer": DISCLAIMER,
    }
