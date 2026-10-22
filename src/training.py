"""Apprentissage par epoques de l'etape 1 (activite 3).

Une epoque presente chaque motif une fois, dans un ordre melange. A la fin
de l'epoque, on recompte les erreurs SANS modifier les poids. On s'arrete
a zero erreur, ou apres MAX_EPOCHS epoques en signalant la limite.
"""

import random
from pathlib import Path

from src.patterns import read_pattern
from src.perceptron import (
    INITIAL_BIAS,
    LEARNING_RATE,
    SEED,
    compute_score,
    init_weights,
    predict,
    update,
)

# Reglage de l'arret. Les autres reglages sont en haut de src/perceptron.py.
MAX_EPOCHS = 1000

Example = tuple[list[int], int]  # (pixels, cible)


def count_errors(weights: list[float], bias: float, examples: list[Example]) -> int:
    """Compte les mauvaises decisions ; les poids ne sont pas modifies."""
    errors = 0
    for pixels, target in examples:
        if predict(compute_score(weights, bias, pixels)) != target:
            errors += 1
    return errors


def snapshot(
    epoch: int, order: list[int], weights: list[float], bias: float,
    examples: list[Example],
) -> dict:
    """Etat mesure a poids figes : erreurs et score de chaque exemple."""
    return {
        "epoch": epoch,
        "order": list(order),  # copie : `order` est remelange a l'epoque suivante
        "errors": count_errors(weights, bias, examples),
        "scores": [compute_score(weights, bias, pixels) for pixels, _ in examples],
        "bias": bias,
    }


def train(
    weights: list[float],
    bias: float,
    examples: list[Example],
    rng: random.Random,
    learning_rate: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[list[float], float, list[dict]]:
    """Apprend jusqu'a zero erreur ou jusqu'a `max_epochs` epoques.

    Renvoie (poids, biais, historique). historique[0] decrit les poids
    initiaux (epoque 0, aucun exemple presente) ; historique[k] l'etat
    apres l'epoque k.
    """
    history = [snapshot(0, [], weights, bias, examples)]
    order = list(range(len(examples)))

    for epoch in range(1, max_epochs + 1):
        rng.shuffle(order)
        for index in order:
            pixels, target = examples[index]
            weights, bias = update(weights, bias, pixels, target, learning_rate)

        history.append(snapshot(epoch, order, weights, bias, examples))
        if history[-1]["errors"] == 0:
            break

    return weights, bias, history


def main() -> None:
    """Entraine le neurone sur zero.txt et un.txt et affiche chaque epoque."""
    data_directory = Path(__file__).resolve().parents[1] / "data"
    filenames = ("zero.txt", "un.txt")
    examples = [read_pattern(data_directory / filename) for filename in filenames]

    # Un seul generateur : d'abord les 48 poids, puis l'ordre de chaque epoque.
    # Meme graine -> memes poids initiaux et meme suite d'ordres.
    rng = random.Random(SEED)
    weights = init_weights(48, rng)
    print(f"Graine {SEED}, biais initial {INITIAL_BIAS}, pas {LEARNING_RATE}, "
          f"au plus {MAX_EPOCHS} epoques\n")

    weights, bias, history = train(weights, INITIAL_BIAS, examples, rng)

    for state in history:
        order = ", ".join(filenames[index] for index in state["order"]) or "-"
        scores = "  ".join(
            f"{filename} {score:+.4f}"
            for filename, score in zip(filenames, state["scores"])
        )
        print(f"epoque {state['epoch']:4d} | ordre {order:17s} | "
              f"erreurs {state['errors']} | b {state['bias']:+.4f} | {scores}")

    last = history[-1]
    if last["errors"] == 0:
        print(f"\nZero erreur apres {last['epoch']} epoque(s).")
    else:
        print(f"\nLimite de {MAX_EPOCHS} epoques atteinte : "
              f"{last['errors']} erreur(s) restante(s).")


if __name__ == "__main__":
    main()
