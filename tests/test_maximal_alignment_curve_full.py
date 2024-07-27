import unittest

import pandas as pd

from multiway_alignment.score import maximal_alignment_curve_fullpartition


class TestComputeMaximalAlignmentCurveFull(unittest.TestCase):
    """
    Test functionality of score.maximal_alignment_curve_fullpartition()
    ------------
    Example
    ------------
    >>> python3 -m unittest -v tests.test_maximal_alignment_curve_fullpartition
    """

    def test_on_empty(self):
        """
        maximal_alignment_curve_fullpartition returns a tuple with two dictionaries
        """
        _a = pd.DataFrame()
        _resall, _res0 = maximal_alignment_curve_fullpartition(_a)
        self.assertIsInstance(
            _resall,
            dict,
            f"""maximal_alignment_curve_fullpartition should return a tuple with two dictionaries, but one was {type(_resall)}""",
        )
        self.assertDictEqual(
            _resall,
            dict(),
            f"""maximal_alignment_curve_fullpartition on empty input should return a tuple with two empty dictionaries,
                    but one was {_resall}""",
        )
        self.assertIsInstance(
            _res0,
            dict,
            f"""maximal_alignment_curve_fullpartition should return a tuple with two dictionaries, but one was {type(_res0)}""",
        )
        self.assertDictEqual(
            _res0,
            dict(),
            f"""maximal_alignment_curve_fullpartition on empty input should return a tuple with two empty dictionaries,
            but one was {_res0}""",
        )

    def test_on_one_layer(self):
        """
        maximal_alignment_curve_fullpartition returns a tuple with two dictionaries
        """
        _a = pd.DataFrame({"A": [0, 1, 2]})
        _resall, _res0 = maximal_alignment_curve_fullpartition(_a)
        self.assertIsInstance(
            _resall,
            dict,
            f"""maximal_alignment_curve_fullpartition should return a tuple with two dictionaries, but one was {type(_resall)}""",
        )
        self.assertIsInstance(
            _res0,
            dict,
            f"""maximal_alignment_curve_fullpartition should return a tuple with two dictionaries, but one was {type(_res0)}""",
        )


if __name__ == "__main__":
    unittest.main()
