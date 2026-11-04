"""
Unit tests for Sliding Window Proportion Analysis
Requirements:
1. All-zero bitstream
2. All-one bitstream
3. Alternating bitstream
4. Balanced bitstream
5. Invalid input
6. Window size greater than bitstream length
"""

import pytest
from backend.app.services.entropy_analyzer import calculate_sliding_window_proportions


class TestSlidingWindowProportions:
    def test_all_zeros(self):
        bs = "0000000000"  # 10 zeros
        res = calculate_sliding_window_proportions(bs, window_size=4, step_size=2)
        assert res["valid"] is True
        assert len(res["proportions"]) > 0
        assert all(p == 0.0 for p in res["proportions"])
        assert all(c == 0 for c in res["ones_counts"])
        assert res["mean_proportion"] == 0.0

    def test_all_ones(self):
        bs = "1111111111"  # 10 ones
        res = calculate_sliding_window_proportions(bs, window_size=4, step_size=2)
        assert res["valid"] is True
        assert len(res["proportions"]) > 0
        assert all(p == 1.0 for p in res["proportions"])
        assert all(c == 4 for c in res["ones_counts"])
        assert res["mean_proportion"] == 1.0

    def test_alternating_bitstream(self):
        bs = "010101010101"  # 12 alternating bits
        res = calculate_sliding_window_proportions(bs, window_size=4, step_size=2)
        assert res["valid"] is True
        # For even window size on alternating stream, proportion is always exactly 0.5
        assert all(p == 0.5 for p in res["proportions"])
        assert all(c == 2 for c in res["ones_counts"])
        assert res["mean_proportion"] == 0.5

    def test_balanced_bitstream(self):
        # 100 bits: half 0s, half 1s structured pseudo-randomly
        bs = "0110011010010110" * 10
        res = calculate_sliding_window_proportions(bs, window_size=16, step_size=8)
        assert res["valid"] is True
        assert len(res["proportions"]) > 0
        # Each 16-bit block has 8 ones, proportion 0.5
        assert all(p == 0.5 for p in res["proportions"])
        assert res["mean_proportion"] == 0.5

    def test_invalid_input_characters(self):
        with pytest.raises(ValueError, match="Invalid non-binary"):
            calculate_sliding_window_proportions("010120101", window_size=4)

        with pytest.raises(ValueError, match="Invalid non-binary"):
            calculate_sliding_window_proportions("0101abc", window_size=4)

    def test_invalid_input_empty_or_non_string(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_sliding_window_proportions("", window_size=4)

        with pytest.raises(ValueError, match="string"):
            calculate_sliding_window_proportions(None, window_size=4)

    def test_window_size_greater_than_bitstream_length(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            calculate_sliding_window_proportions("010101", window_size=20)

    def test_invalid_window_size_non_positive(self):
        with pytest.raises(ValueError, match="positive integer"):
            calculate_sliding_window_proportions("010101", window_size=0)

        with pytest.raises(ValueError, match="positive integer"):
            calculate_sliding_window_proportions("010101", window_size=-5)

    def test_window_edge_alignment(self):
        bs = "0000111100"  # 10 bits
        res = calculate_sliding_window_proportions(bs, window_size=4, step_size=3)
        assert res["valid"] is True
        # Starts at 0, 3, 6; last pos is 6 (6+4=10).
        assert res["positions"] == [0, 3, 6]
        assert res["ones_counts"] == [0, 3, 2]
