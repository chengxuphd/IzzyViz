"""Unit tests for izzyviz.utility.

utility.py has no imports at all, so these run without torch, matplotlib or
any model. They cover the geometry layer the region-detection features are
built on.
"""
import pytest

from izzyviz.utility import (
    build_prefix_sum,
    compute_ratio,
    find_non_overlapping_locally_maximal_rectangles,
    overlap,
    submatrix_sum,
)


class TestPrefixSum:
    def test_shape_is_padded_by_one(self):
        prefix = build_prefix_sum([[1, 2], [3, 4]])
        assert len(prefix) == 3
        assert len(prefix[0]) == 3

    def test_zero_border(self):
        prefix = build_prefix_sum([[1, 2], [3, 4]])
        assert prefix[0] == [0, 0, 0]
        assert [row[0] for row in prefix] == [0, 0, 0]

    def test_cumulative_values(self):
        prefix = build_prefix_sum([[1, 2], [3, 4]])
        assert prefix[1][1] == 1
        assert prefix[1][2] == 3
        assert prefix[2][2] == 10

    def test_single_cell(self):
        assert build_prefix_sum([[7]])[1][1] == 7


class TestSubmatrixSum:
    @pytest.fixture
    def prefix(self):
        return build_prefix_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_whole_matrix(self, prefix):
        assert submatrix_sum(prefix, 0, 0, 2, 2) == 45

    def test_single_cell(self, prefix):
        assert submatrix_sum(prefix, 1, 1, 1, 1) == 5

    def test_interior_block(self, prefix):
        assert submatrix_sum(prefix, 0, 0, 1, 1) == 12

    def test_first_row_and_first_column(self, prefix):
        assert submatrix_sum(prefix, 0, 0, 0, 2) == 6
        assert submatrix_sum(prefix, 0, 0, 2, 0) == 12


class TestComputeRatio:
    def test_mean_of_uniform_block_is_the_value(self):
        assert compute_ratio(build_prefix_sum([[4, 4], [4, 4]]), 0, 0, 1, 1) == 4.0

    def test_mean_of_mixed_block(self):
        assert compute_ratio(build_prefix_sum([[1, 2], [3, 4]]), 0, 0, 1, 1) == 2.5

    def test_single_cell_ratio_is_the_cell(self):
        assert compute_ratio(build_prefix_sum([[9]]), 0, 0, 0, 0) == 9.0


class TestOverlap:
    """overlap() takes flat (r1, c1, r2, c2) tuples, not nested corner pairs."""

    def test_identical_rectangles_overlap(self):
        assert overlap((0, 0, 2, 2), (0, 0, 2, 2))

    def test_disjoint_rectangles_do_not(self):
        assert not overlap((0, 0, 1, 1), (2, 2, 3, 3))

    def test_partial_overlap_detected(self):
        assert overlap((0, 0, 2, 2), (1, 1, 3, 3))

    def test_touching_rectangles_count_as_overlapping(self):
        """Boundaries are inclusive: rect ending at row 1 and one starting at
        row 1 share that row, so they overlap. Guards the >= vs > choice."""
        assert overlap((0, 0, 1, 1), (1, 0, 2, 1))

    def test_column_separated_rectangles_do_not_overlap(self):
        assert not overlap((0, 0, 1, 1), (0, 2, 1, 3))

    def test_containment_counts_as_overlap(self):
        assert overlap((0, 0, 5, 5), (2, 2, 3, 3))


class TestFindRegions:
    """Regression tests.

    These outputs were captured from the implementation as it stood at
    a858a74 and must not change. They are the oracle that guarded removal
    of the duplicated definitions in utility.py.
    """

    HOT_CORNER = [[1, 1, 0, 0], [1, 9, 0, 0], [0, 0, 5, 5], [0, 0, 5, 5]]

    def test_hot_corner(self):
        assert find_non_overlapping_locally_maximal_rectangles(self.HOT_CORNER) == (
            [(1, 1), (2, 2)],
            [(1, 1), (3, 3)],
        )

    def test_uniform_matrix(self):
        assert find_non_overlapping_locally_maximal_rectangles([[1, 1], [1, 1]]) == (
            [(0, 0)],
            [(1, 1)],
        )

    def test_single_cell(self):
        assert find_non_overlapping_locally_maximal_rectangles([[7]]) == (
            [(0, 0)],
            [(0, 0)],
        )

    def test_returns_matched_pairs(self):
        tops, bottoms = find_non_overlapping_locally_maximal_rectangles(self.HOT_CORNER)
        assert len(tops) == len(bottoms)

    def test_regions_are_well_formed(self):
        tops, bottoms = find_non_overlapping_locally_maximal_rectangles(self.HOT_CORNER)
        for (r1, c1), (r2, c2) in zip(tops, bottoms):
            assert r1 <= r2 and c1 <= c2

    def test_all_zero_matrix_does_not_crash(self):
        tops, bottoms = find_non_overlapping_locally_maximal_rectangles([[0, 0], [0, 0]])
        assert len(tops) == len(bottoms)
