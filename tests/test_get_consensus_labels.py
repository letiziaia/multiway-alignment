import unittest

import pandas as pd

from multilayer_alignment.consensus import get_consensus_labels


class TestGetConsensusLabels(unittest.TestCase):
    """
    Test functionality of consensus.get_consensus_labels()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_get_consensus_labels
    """

    def test_on_empty(self):
        """
        get_consensus_labels returns a list
        """
        _a = pd.DataFrame()
        _res0 = get_consensus_labels(_a)
        self.assertIsInstance(
            _res0,
            list,
            f"""get_consensus_labels should return a list,
            but returned {type(_res0)}""",
        )
        self.assertTrue(
            len(_res0) == 0,
            f"""get_consensus_labels called on empty pd.DataFrame should return
            an empty list, but returned {_res0}""",
        )

    def test_on_simple_sets(self):
        """
        get_consensus_labels returns a list
        """
        _a = pd.DataFrame(
            {
                "A": [0, 0, 0, 1, 1, 1],
                "B": [1, 0, 0, 1, 0, 0],
                "C": [1, 1, 0, 0, 1, 1],
            }
        )
        _res0 = get_consensus_labels(_a)
        self.assertIsInstance(
            _res0,
            list,
            f"""get_consensus_labels should return a list,
            but returned {type(_res0)}""",
        )
        self.assertFalse(
            len(_res0) == 0,
            f"""get_consensus_labels called on non-empty pd.DataFrame should return
            a non-empty list, but returned {_res0}""",
        )
        self.assertListEqual(
            _res0,
            ["A0_B1_C1", "A0_B0_C1", "A0_B0_C0", "A1_B1_C0", "A1_B0_C1", "A1_B0_C1"],
            f"""get_consensus_labels called on non-empty pd.DataFrame should return
            the correct non-empty list, but returned {_res0}""",
        )


if __name__ == "__main__":
    unittest.main()
