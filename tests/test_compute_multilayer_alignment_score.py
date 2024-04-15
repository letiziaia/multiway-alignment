import unittest

import pandas as pd

from multiway_alignment.score import multiway_alignment_score


class TestComputeMultiwayAlignmentScore(unittest.TestCase):
    """
    Test functionality of mutual_clusters.compute_multiway_alignment_score()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_compute_multiway_alignment_score
    """

    def test_on_empty(self):
        """
        compute_multiway_alignment_score returns a float
        """
        _a = pd.DataFrame({"A": [0, 1, 2]})
        _labels = ["a", "b", "c"]
        _res0 = multiway_alignment_score(_a, _labels)
        self.assertIsInstance(
            _res0,
            float,
            f"""compute_multiway_alignment_score should return a float, but returned {type(_res0)}""",
        )
        self.assertGreaterEqual(
            _res0,
            0.0,
            """compute_multiway_alignment_score should return the correct value of avg NMI""",
        )
        self.assertLessEqual(
            _res0,
            1.0,
            """compute_multiway_alignment_score should return the correct value of avg NMI""",
        )
        self.assertEqual(
            _res0,
            1.0,
            """compute_multiway_alignment_score should return the correct value of avg NMI""",
        )


if __name__ == "__main__":
    unittest.main()
