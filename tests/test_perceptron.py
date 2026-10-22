"""Controles du score, de la decision et de la regle du perceptron."""

import random
import unittest

from src.perceptron import compute_score, init_weights, predict, update


class PaperExampleTests(unittest.TestCase):
    """Exemple du sujet : x = [1, 0], w = [0.2, -0.1], b = -0.5, d = 1."""

    def test_output(self):
        score = compute_score([0.2, -0.1], -0.5, [1, 0])
        self.assertAlmostEqual(score, -0.3)  # 0.2 * 1 + (-0.1) * 0 - 0.5
        self.assertEqual(predict(score), 0)  # donc e = 1 - 0 = 1

    def test_update(self):
        weights, bias = update([0.2, -0.1], -0.5, [1, 0], 1, 0.01)
        self.assertAlmostEqual(weights[0], 0.21)  # g0 = -1
        self.assertEqual(weights[1], -0.1)        # x1 = 0 : poids inchange
        self.assertAlmostEqual(bias, -0.49)       # g_b = -1


class UpdateTests(unittest.TestCase):
    def test_no_correction_when_prediction_is_right(self):
        weights, bias = update([0.2, -0.1], -0.5, [1, 0], 0, 0.01)
        self.assertEqual((weights, bias), ([0.2, -0.1], -0.5))

    def test_lowers_score_when_target_is_zero(self):
        weights, bias = update([0.0, 0.0], 0.0, [1, 1], 0, 0.1)  # a = 0 -> y = 1
        self.assertEqual(weights, [-0.1, -0.1])
        self.assertAlmostEqual(bias, -0.1)

    def test_does_not_modify_received_weights(self):
        weights = [0.2, -0.1]
        update(weights, -0.5, [1, 0], 1, 0.01)
        self.assertEqual(weights, [0.2, -0.1])


class NeuronTests(unittest.TestCase):
    def test_threshold_is_included(self):
        self.assertEqual(predict(0.0), 1)
        self.assertEqual(predict(-1e-9), 0)

    def test_score_rejects_wrong_sizes(self):
        with self.assertRaises(ValueError):
            compute_score([0.1, 0.2], 0.0, [1, 0, 1])

    def test_init_weights_range_and_seed(self):
        first = init_weights(48, random.Random(42))
        self.assertEqual(len(first), 48)
        self.assertTrue(all(-0.1 <= weight <= 0.1 for weight in first))
        self.assertEqual(first, init_weights(48, random.Random(42)))
        self.assertNotEqual(first, init_weights(48, random.Random(43)))


if __name__ == "__main__":
    unittest.main()
