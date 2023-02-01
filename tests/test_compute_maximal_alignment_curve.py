import unittest

import pandas as pd

from src.alignment_score import compute_maximal_alignment_curve


class TestComputeMaximalAlignmentCurve(unittest.TestCase):
    """
    Test functionality of mutual_clusters.compute_maximal_alignment_curve()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_compute_maximal_alignment_curve
    """

    def test_on_empty(self):
        """
        compute_maximal_alignment_curve returns a dictionary
        """
        _a = pd.DataFrame()
        _res0 = compute_maximal_alignment_curve(_a)
        self.assertIsInstance(
            _res0,
            dict,
            f"""compute_maximal_alignment_curve should return a dictionary, but returned {type(_res0)}""",
        )
        self.assertDictEqual(
            _res0,
            dict(),
            f"""compute_maximal_alignment_curve on empty input should return an empty dictionary, but returned {_res0}""",
        )

    def test_on_one_layer(self):
        """
        compute_maximal_alignment_curve returns a dictionary
        """
        _a = pd.DataFrame({"A": [0, 1, 2]})
        _res0 = compute_maximal_alignment_curve(_a)
        self.assertIsInstance(
            _res0,
            dict,
            f"""compute_maximal_alignment_curve should return a dictionary, but returned {type(_res0)}""",
        )


if __name__ == "__main__":
    unittest.main()
