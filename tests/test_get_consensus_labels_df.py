import unittest

import pandas as pd

from multilayer_alignment.consensus import _get_consensus_labels_df


class TestGetConsensusLabels(unittest.TestCase):
    """
    Test functionality of consensus.get_consensus_labels_df()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_get_consensus_labels_df
    """

    def test_on_empty(self):
        """
        _get_consensus_labels_df returns a pd.DataFrame
        """
        _a = dict()
        _res0 = _get_consensus_labels_df(_a)
        self.assertIsInstance(
            _res0,
            pd.DataFrame,
            f"""_get_consensus_labels_df should return a pd.DataFrame,
            but returned {type(_res0)}""",
        )
        self.assertTrue(
            _res0.empty,
            f"""_get_consensus_labels_df called on empty dictionary should return
            an empty pd.DataFrame, but returned {_res0}""",
        )

    def test_on_simple_sets(self):
        """
        _get_consensus_labels_df returns a pd.DataFrame
        """
        _a = {"A0_B1_C0": {0, 1}, "A1_B0_C1": {2}, "A1_B1_C0": {3}}
        _res0 = _get_consensus_labels_df(_a)
        self.assertIsInstance(
            _res0,
            pd.DataFrame,
            f"""_get_consensus_labels_df should return a pd.DataFrame,
            but returned {type(_res0)}""",
        )
        self.assertFalse(
            _res0.empty,
            f"""_get_consensus_labels_df called on non-empty dictionary should return
            a non-empty pd.DataFrame, but returned {_res0}""",
        )


if __name__ == "__main__":
    unittest.main()
