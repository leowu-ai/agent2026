import unittest

from multiscale_vqa_agent.evaluate_bcnb_histology_head import (
    collapse_probabilities,
    grouped_prediction,
    hard_collapse,
)


class BCNBHistologyHeadTest(unittest.TestCase):
    def test_hard_collapse(self):
        self.assertEqual(hard_collapse(1), "Invasive ductal carcinoma")
        self.assertEqual(hard_collapse(2), "Invasive lobular carcinoma")
        for index in (0, 3, 4, 5, 6, 7, 8):
            self.assertEqual(hard_collapse(index), "Other")

    def test_probability_collapse_sums_other_classes(self):
        grouped = collapse_probabilities([0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1])
        self.assertAlmostEqual(grouped["Invasive ductal carcinoma"], 0.3)
        self.assertAlmostEqual(grouped["Invasive lobular carcinoma"], 0.2)
        self.assertAlmostEqual(grouped["Other"], 0.5)
        self.assertEqual(grouped_prediction([0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1])["answer"], "Other")

    def test_probability_dimension_is_validated(self):
        with self.assertRaises(ValueError):
            collapse_probabilities([0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
