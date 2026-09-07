import unittest

import numpy as np

from frt.audio.analysis import max_cqt_bins, shift_frequency_bins


class AnalysisTest(unittest.TestCase):
    def test_shift_frequency_bins_positive_offset_moves_rows_up(self) -> None:
        matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        shifted = shift_frequency_bins(matrix, 1, fill_value=0.0)

        np.testing.assert_allclose(
            shifted,
            np.array([[0, 0], [1, 2], [3, 4]], dtype=np.float32),
        )

    def test_shift_frequency_bins_negative_offset_moves_rows_down(self) -> None:
        matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        shifted = shift_frequency_bins(matrix, -1, fill_value=0.0)

        np.testing.assert_allclose(
            shifted,
            np.array([[3, 4], [5, 6], [0, 0]], dtype=np.float32),
        )

    def test_shift_frequency_bins_clears_fully_out_of_range_shift(self) -> None:
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

        shifted = shift_frequency_bins(matrix, 2, fill_value=-100.0)

        np.testing.assert_allclose(
            shifted,
            np.full(matrix.shape, -100.0, dtype=np.float32),
        )

    def test_shift_frequency_bins_does_not_mutate_input(self) -> None:
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        original = matrix.copy()

        shifted = shift_frequency_bins(matrix, 0, fill_value=0.0)
        shifted[0, 0] = 99.0

        np.testing.assert_allclose(matrix, original)

    def test_max_cqt_bins_stays_under_nyquist(self) -> None:
        self.assertEqual(max_cqt_bins(sample_rate=44100), 113)


if __name__ == "__main__":
    unittest.main()
