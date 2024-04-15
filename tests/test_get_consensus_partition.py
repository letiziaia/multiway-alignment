import unittest

import pandas as pd

from multiway_alignment.consensus import get_consensus_partition


class TestGetConsensusPartition(unittest.TestCase):
    """
    Test functionality of mutual_clusters.get_consensus_partition()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_get_consensus_partition
    """

    def test_on_empty(self):
        """
        get_consensus_partition returns a dictionary
        """
        _a = pd.DataFrame()
        _res0 = get_consensus_partition(opinions=_a)
        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )
        self.assertEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on empty dataframe should return an empty dictionary,
            but returned {_res0}""",
        )

    def test_on_one_layer(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 1, 2]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A0": {0}, "A1": {1}, "A2": {2}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a non-empty dictionary,
            but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return
            the expected non-empty dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_two_layers_0(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A0_B3": {0}, "A1_B4": {1}, "A2_B5": {2}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a
            non-empty dictionary, but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return the expected non-empty
            dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_two_layers_1(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 0, 2, 2], "B": [0, 0, 2, 2]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A0_B0": {0, 1}, "A2_B2": {2, 3}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a
            non-empty dictionary, but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return
            the expected non-empty dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_two_layers_2(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [3, 3, 3, 3], "B": [0, 0, 2, 2]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A3_B0": {0, 1}, "A3_B2": {2, 3}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a
            non-empty dictionary, but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return
            the expected non-empty dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_three_layers_0(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 0, 2, 2], "B": [0, 0, 2, 2], "C": [0, 0, 2, 2]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A0_B0_C0": {0, 1}, "A2_B2_C2": {2, 3}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a
            non-empty dictionary, but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return
            the expected non-empty dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_three_layers_1(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [1, 0, 1, 0]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {
            "A0_B0_C1": {0},
            "A0_B1_C0": {1},
            "A1_B0_C1": {2},
            "A1_B1_C0": {3},
        }

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a non-empty dictionary,
            but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return
            the expected non-empty dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )

    def test_on_three_layers_2(self):
        """
        get_consensus_partition returns a dictionary, and returns all the nodes
        """
        _a = pd.DataFrame({"A": [0, 0, 1, 1], "B": [1, 1, 0, 1], "C": [0, 0, 1, 0]})
        _all_nodes = list(_a.index)
        _res0 = get_consensus_partition(opinions=_a)
        _returned_nodes = [n for mc in _res0.values() for n in mc]
        _expected0 = {"A0_B1_C0": {0, 1}, "A1_B0_C1": {2}, "A1_B1_C0": {3}}

        self.assertIsInstance(
            _res0,
            dict,
            f"""get_consensus_partition should return a dictionary, but returned {type(_res0)}""",
        )

        self.assertNotEqual(
            _res0,
            dict(),
            f"""get_consensus_partition called on non-empty dataframe should return a non-empty dictionary,
            but returned {_res0}""",
        )

        self.assertDictEqual(
            _res0,
            _expected0,
            f"""get_consensus_partition called on non-empty dataframe should return the expected non-empty
                dictionary, but returned {_res0}""",
        )

        self.assertSetEqual(
            set(_all_nodes),
            set(_returned_nodes),
            """get_consensus_partition called on non-empty dataframe should return all nodes""",
        )


if __name__ == "__main__":
    unittest.main()
