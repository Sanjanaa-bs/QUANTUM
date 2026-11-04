"""
Von Neumann Debiasing Service
==============================

Implements the Von Neumann extractor to reduce bias in a binary stream.

Algorithm:
    1. Split the input stream into non-overlapping consecutive pairs.
    2. Discard pairs 00 (both zeros).
    3. Discard pairs 11 (both ones).
    4. Map pair 01 → output bit 0.
    5. Map pair 10 → output bit 1.
    6. If the stream has an odd number of bits, the final unpaired bit is discarded.

Assumptions and Limitations:
    - Assumes independent, identically distributed (i.i.d.) input bits.
    - Does NOT remove arbitrary correlations.
    - Does NOT make adversarial inputs cryptographically secure.
    - Does NOT prove that the output is truly random.
    - May substantially reduce the number of available bits.
    - The pair-to-bit mapping is deterministic for a given input;
      the extractor preserves pair order exactly.
"""

import logging
import math
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ALGORITHM_NOTE = (
    "The von Neumann extractor reduces bias under the assumption that input bits "
    "are independent and identically distributed (i.i.d.). "
    "It does NOT automatically remove arbitrary correlations, does NOT make "
    "adversarial inputs secure, and does NOT prove the output is truly random. "
    "Extraction can substantially reduce the number of available bits."
)


def _validate_bitstream(bits: str) -> str:
    """
    Validate and clean a binary string.
    Raises ValueError for invalid characters.
    Returns the cleaned string.
    """
    cleaned = bits.strip()
    invalid = set(cleaned) - {"0", "1"}
    if invalid:
        raise ValueError(
            f"Bitstream contains invalid characters: {invalid!r}. "
            "Only '0' and '1' are accepted."
        )
    return cleaned


def _shannon_entropy(bitstream: str) -> float:
    n = len(bitstream)
    if n == 0:
        return 0.0
    p1 = bitstream.count("1") / n
    p0 = 1.0 - p1
    h = 0.0
    if p0 > 0:
        h -= p0 * math.log2(p0)
    if p1 > 0:
        h -= p1 * math.log2(p1)
    return round(h, 6)


def _compute_stats(bitstream: str) -> Dict[str, Any]:
    n = len(bitstream)
    if n == 0:
        return {
            "total_bits": 0, "zeros": 0, "ones": 0,
            "proportion_ones": 0.0, "proportion_zeros": 0.0,
            "bias_deviation": 0.5, "shannon_entropy_per_bit": 0.0,
            "min_entropy_per_bit": 0.0,
        }
    ones = bitstream.count("1")
    zeros = n - ones
    p1 = ones / n
    p0 = zeros / n
    max_p = max(p0, p1)
    min_entropy = 0.0 if (max_p <= 0.0 or max_p >= 1.0) else -math.log2(max_p)
    return {
        "total_bits": n,
        "zeros": zeros,
        "ones": ones,
        "proportion_ones": round(p1, 6),
        "proportion_zeros": round(p0, 6),
        "bias_deviation": round(abs(p1 - 0.5), 6),
        "shannon_entropy_per_bit": _shannon_entropy(bitstream),
        "min_entropy_per_bit": round(min_entropy, 6),
    }


def von_neumann_debias(bitstream: str) -> Dict[str, Any]:
    """
    Apply the von Neumann extractor to a binary string.

    Args:
        bitstream: A string of '0' and '1' characters.

    Returns:
        A dictionary with:
            - input_bits, output_bits, output_bitstream
            - pair counts (00, 01, 10, 11)
            - retained / discarded pairs
            - input and output statistics
            - extraction efficiency
            - warnings

    Edge cases handled:
        - Empty string → returns empty output with a warning.
        - String of length 1 → single bit discarded, warning issued.
        - All-equal pairs (all 00 or all 11) → zero output bits, warning issued.
        - Odd-length input → final bit discarded, warned.
        - Invalid characters → raises ValueError.
    """
    debias_id = str(uuid.uuid4())
    warnings = []

    # Validate
    bits = _validate_bitstream(bitstream)
    input_bits = len(bits)

    if input_bits == 0:
        warnings.append("Input bitstream is empty. No output produced.")
        return {
            "debias_id": debias_id,
            "input_bits": 0,
            "total_complete_pairs": 0,
            "count_00": 0,
            "count_01": 0,
            "count_10": 0,
            "count_11": 0,
            "retained_pairs": 0,
            "discarded_pairs": 0,
            "discarded_percentage": 0.0,
            "discarded_bits": 0,
            "output_bits": 0,
            "output_bitstream": "",
            "input_stats": _compute_stats(""),
            "output_stats": _compute_stats(""),
            "output_to_input_ratio": 0.0,
            "pair_retention_ratio": 0.0,
            "extraction_efficiency": 0.0,
            "warnings": warnings,
            "algorithm_note": ALGORITHM_NOTE,
        }

    if input_bits % 2 != 0:
        warnings.append(
            f"Odd-length input ({input_bits} bits): final unpaired bit discarded."
        )

    # Process pairs
    total_complete_pairs = input_bits // 2
    count_00 = count_01 = count_10 = count_11 = 0
    output_bits_list = []

    for i in range(0, 2 * total_complete_pairs, 2):
        pair = bits[i] + bits[i + 1]
        if pair == "00":
            count_00 += 1
        elif pair == "01":
            count_01 += 1
            output_bits_list.append("0")
        elif pair == "10":
            count_10 += 1
            output_bits_list.append("1")
        elif pair == "11":
            count_11 += 1

    retained_pairs = count_01 + count_10
    discarded_pairs = count_00 + count_11
    discarded_pct = (
        round(discarded_pairs / total_complete_pairs * 100, 2)
        if total_complete_pairs > 0 else 0.0
    )

    output_bitstream = "".join(output_bits_list)
    output_bits = len(output_bitstream)
    output_to_input_ratio = (
        round(output_bits / input_bits, 4) if input_bits > 0 else 0.0
    )
    pair_retention_ratio = (
        round(output_bits / total_complete_pairs, 4) if total_complete_pairs > 0 else 0.0
    )
    extraction_efficiency = output_to_input_ratio  # Backwards compatibility alias
    discarded_bits = input_bits - output_bits

    # Warnings for degenerate cases
    if output_bits == 0:
        warnings.append(
            "All pairs were equal (00 or 11). No output bits produced. "
            "This may indicate strong correlation or extreme bias in the input."
        )
    elif discarded_pct > 80:
        warnings.append(
            f"High discard rate ({discarded_pct:.1f}%): the input may be strongly biased."
        )

    if output_bits < 100:
        warnings.append(
            f"Output stream is short ({output_bits} bits). "
            "Statistical tests may not be reliable."
        )

    input_stats = _compute_stats(bits)
    output_stats = _compute_stats(output_bitstream)

    return {
        "debias_id": debias_id,
        "input_bits": input_bits,
        "total_complete_pairs": total_complete_pairs,
        "count_00": count_00,
        "count_01": count_01,
        "count_10": count_10,
        "count_11": count_11,
        "retained_pairs": retained_pairs,
        "discarded_pairs": discarded_pairs,
        "discarded_percentage": discarded_pct,
        "discarded_bits": discarded_bits,
        "output_bits": output_bits,
        "output_bitstream": output_bitstream,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "output_to_input_ratio": output_to_input_ratio,
        "pair_retention_ratio": pair_retention_ratio,
        "extraction_efficiency": extraction_efficiency,
        "warnings": warnings,
        "algorithm_note": ALGORITHM_NOTE,
    }
