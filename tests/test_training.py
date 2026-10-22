"""Controles de l'apprentissage par epoques (activite 3 de l'etape 1)."""

from pathlib import Path
import random
import unittest

from src.patterns import read_pattern
from src.perceptron import init_weights
from src.training import count_errors, train


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.examples = [read_pattern(DATA_DIRECTORY / name)
                         for name in ("zero.txt", "un.txt")]

    def train_with_seed(self, seed):
        rng = random.Random(seed)
        weights = init_weights(48, rng)
        return train(weights, -0.5, self.examples, rng)

    def test_reaches_zero_error_on_both_patterns(self):
        weights, bias, history = self.train_with_seed(42)
        self.assertEqual(history[-1]["errors"], 0)
        self.assertEqual(count_errors(weights, bias, self.examples), 0)

    def test_stops_at_first_epoch_without_error(self):
        _, _, history = self.train_with_seed(42)
        self.assertTrue(all(state["errors"] > 0 for state in history[1:-1]))
        self.assertEqual([state["epoch"] for state in history],
                         list(range(len(history))))

    def test_each_epoch_presents_each_pattern_once(self):
        _, _, history = self.train_with_seed(42)
        self.assertEqual(history[0]["order"], [])  # epoque 0 : poids initiaux
        for state in history[1:]:
            self.assertEqual(sorted(state["order"]), [0, 1])

    def test_same_seed_gives_same_result(self):
        self.assertEqual(self.train_with_seed(42), self.train_with_seed(42))

    def test_signals_the_epoch_limit(self):
        # Meme entree avec deux cibles : aucune frontiere ne separe les exemples.
        pixels = self.examples[0][0]
        _, _, history = train([0.0] * 48, -0.5, [(pixels, 0), (pixels, 1)],
                              random.Random(0), max_epochs=20)
        self.assertEqual(len(history), 21)  # epoque 0 + 20 epoques
        self.assertEqual(history[-1]["errors"], 1)

    def test_counting_does_not_modify_weights(self):
        weights = init_weights(48, random.Random(42))
        copy = list(weights)
        count_errors(weights, -0.5, self.examples)
        self.assertEqual(weights, copy)


if __name__ == "__main__":
    unittest.main()
