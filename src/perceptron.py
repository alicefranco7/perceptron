"""Neurone simple de l'etape 1 : score, decision et regle du perceptron.

    a = somme_j w_j x_j + b ;  y = 1 si a >= 0, sinon 0
    e = d - y ;  g_j = -e x_j ;  g_b = -e
    w_j <- w_j - eta g_j ;  b <- b - eta g_b

d est la cible (target), y la prediction. Le biais initial b = -0.5
correspond au seuil theta = 0.5 (b = -theta).
"""

import random
from pathlib import Path

from src.patterns import read_pattern

# Reglages du sujet. Les modifier ici pour essayer, puis relancer.
SEED = 42
WEIGHT_RANGE = 0.1
INITIAL_BIAS = -0.5
LEARNING_RATE = 0.01


def init_weights(count: int, rng: random.Random) -> list[float]:
    """Tire `count` poids uniformement dans [-WEIGHT_RANGE ; WEIGHT_RANGE]."""
    return [rng.uniform(-WEIGHT_RANGE, WEIGHT_RANGE) for _ in range(count)]


def compute_score(weights: list[float], bias: float, pixels: list[int]) -> float:
    """Score a = somme des w_j x_j, plus le biais."""
    if len(weights) != len(pixels):
        raise ValueError("Il faut autant de poids que de pixels.")
    score = 0.0
    for weight, pixel in zip(weights, pixels):
        score += weight * pixel
    return score + bias


def predict(score: float) -> int:
    """Decision : 1 si le score est positif ou nul, sinon 0."""
    return 1 if score >= 0 else 0


def update(
    weights: list[float],
    bias: float,
    pixels: list[int],
    target: int,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[float], float]:
    """Applique une correction de la regle du perceptron sur un exemple.

    Renvoie de nouveaux poids et un nouveau biais ; la liste recue n'est pas
    modifiee. Si la prediction est deja juste, e = 0 et rien ne change.
    """
    prediction = predict(compute_score(weights, bias, pixels))
    error = target - prediction

    new_weights = []
    for weight, pixel in zip(weights, pixels):
        gradient = -error * pixel
        new_weights.append(weight - learning_rate * gradient)
    bias_gradient = -error
    new_bias = bias - learning_rate * bias_gradient
    return new_weights, new_bias


def show_step(
    weights: list[float], bias: float, pixels: list[int], target: int
) -> tuple[list[float], float]:
    """Affiche a, y, e pour un exemple, puis l'effet d'une seule correction."""
    score = compute_score(weights, bias, pixels)
    prediction = predict(score)
    print(f"  score a = {score:+.4f}, prediction = {prediction}, "
          f"cible = {target}, erreur e = {target - prediction:+d}")
    new_weights, new_bias = update(weights, bias, pixels, target)
    new_score = compute_score(new_weights, new_bias, pixels)
    print(f"  apres correction : b = {new_bias:+.4f}, score a = {new_score:+.4f}")
    return new_weights, new_bias


def main() -> None:
    """Trace le calcul sur papier, puis une correction sur chaque motif."""
    print("Calcul sur papier : x = [1, 0], w = [0.2, -0.1], b = -0.5, d = 1")
    new_weights, _ = show_step([0.2, -0.1], -0.5, [1, 0], 1)
    print(f"  w = {[round(w, 4) for w in new_weights]}")

    rng = random.Random(SEED)
    weights = init_weights(48, rng)
    data_directory = Path(__file__).resolve().parents[1] / "data"
    for filename in ("zero.txt", "un.txt"):
        pixels, target = read_pattern(data_directory / filename)
        print(f"\n{filename} avec les poids initiaux (graine {SEED}) :")
        show_step(weights, INITIAL_BIAS, pixels, target)


if __name__ == "__main__":
    main()
