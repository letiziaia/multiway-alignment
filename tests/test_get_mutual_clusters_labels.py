import unittest

import pandas as pd

from src.mutual_clusters import get_mutual_clusters_labels


class TestGetMutualClustersLabels(unittest.TestCase):
    """
    Test functionality of mutual_clusters.get_mutual_clusters_labels()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_get_mutual_clusters_labels
    """

    def test_on_empty(self):
        """
        get_mutual_clusters_labels returns a pd.DataFrame
        """
        _a = pd.DataFrame()
        _res0 = get_mutual_clusters_labels(_a)
        self.assertIsInstance(
            _res0,
            pd.DataFrame,
            f"""get_mutual_clusters_labels should return a pd.DataFrame, but returned {type(_res0)}""",
        )
        self.assertTrue(
            _res0.empty,
            f"""get_mutual_clusters_labels called on empty dictionary should return an empty pd.DataFrame, but returned {_res0}""",
        )

    def test_on_simple_sets(self):
        """
        get_mutual_clusters_labels returns a pd.DataFrame
        """
        _a = {"A0_B1_C0": {0, 1}, "A1_B0_C1": {2}, "A1_B1_C0": {3}}
        _res0 = get_mutual_clusters_labels(_a)
        self.assertIsInstance(
            _res0,
            pd.DataFrame,
            f"""get_mutual_clusters_labels should return a pd.DataFrame, but returned {type(_res0)}""",
        )
        self.assertFalse(
            _res0.empty,
            f"""get_mutual_clusters_labels called on non-empty dictionary should return a non-empty pd.DataFrame,
            but returned {_res0}""",
        )


if __name__ == "__main__":
    unittest.main()
