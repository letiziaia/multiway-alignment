import unittest

import pandas as pd

from multiway_alignment.score import multiway_alignment_score


class TestComputeMultiwayAlignmentScore(unittest.TestCase):
    """
    Test functionality of mutual_clusters.multiway_alignment_score()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.multiway_alignment_score
    """

    def test_on_empty(self):
        """
        multiway_alignment_score raises ZeroDivisionError if the dataframe is empty
        """
        _a = pd.DataFrame()
        # the function should raise ZeroDivisionError if the dataframe is empty
        with self.assertRaises(ZeroDivisionError):
            multiway_alignment_score(_a)

    def test_on_single_dimension(self):
        """
        multiway_alignment_score raises ValueError if there is only one dimension
        """
        _a = pd.DataFrame({"A": [0, 1, 2]})
        # the function should raise ValueError if the there is only one dimension
        with self.assertRaises(ValueError):
            multiway_alignment_score(_a)

    def test_on_two_dimensions(self):
        """
        multiway_alignment_score returns a float
        """
        _a = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2]})
        _res0 = multiway_alignment_score(_a, "nmi", False)
        self.assertIsInstance(
            _res0,
            float,
            f"""multiway_alignment_score should return a float, but returned {type(_res0)}""",
        )
        self.assertGreaterEqual(
            _res0,
            1.0,
            """multiway_alignment_score should return the correct value in case of perfect alignment when nmi is used""",
        )

        _res1 = multiway_alignment_score(_a, "ami", False)
        self.assertIsInstance(
            _res1,
            float,
            f"""multiway_alignment_score should return a float, but returned {type(_res1)}""",
        )
        self.assertGreaterEqual(
            _res1,
            1.0,
            """multiway_alignment_score should return the correct value in case of perfect alignment when ami is used""",
        )

    def test_on_three_dimensions(self):
        """
        multiway_alignment_score returns a float
        """
        _a = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})
        _res0 = multiway_alignment_score(_a, "nmi", False)

        self.assertIsInstance(
            _res0,
            float,
            f"""multiway_alignment_score should return a float, but returned {type(_res0)}""",
        )
        self.assertGreaterEqual(
            _res0,
            1.0,
            """multiway_alignment_score should return the correct value in case of perfect alignment when nmi is used""",
        )

        _res1 = multiway_alignment_score(_a, "ami", False)
        self.assertIsInstance(
            _res1,
            float,
            f"""multiway_alignment_score should return a float, but returned {type(_res1)}""",
        )
        self.assertGreaterEqual(
            _res1,
            1.0,
            """multiway_alignment_score should return the correct value in case of perfect alignment when ami is used""",
        )


if __name__ == "__main__":
    unittest.main()
