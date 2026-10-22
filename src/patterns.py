"""Lecture des motifs binaires de 8 lignes et 6 colonnes."""

from pathlib import Path


def read_pattern(path: str | Path) -> tuple[list[int], int]:
    """Renvoie les 48 pixels, lus par ligne, et la cible separee (0 ou 1).

    Le fichier contient huit lignes de six caracteres ('.' ou '*'), puis
    une neuvieme ligne avec la cible. Un format incorrect leve ValueError.
    Les erreurs d'acces au fichier sont laissees a l'appelant.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != 9:
        raise ValueError("Le fichier doit contenir 8 lignes de pixels et 1 cible.")

    pixels = []
    # L'ordre des pixels doit rester le meme pour tous les motifs.
    for row_number, line in enumerate(lines[:8], start=1):
        if len(line) != 6:
            raise ValueError(f"Ligne {row_number} : 6 pixels sont attendus.")
        for character in line:
            if character not in ".*":
                raise ValueError(
                    f"Ligne {row_number} : seuls '.' et '*' sont autorises."
                )
            pixels.append(1 if character == "*" else 0)

    if lines[8] not in ("0", "1"):
        raise ValueError("La cible doit etre 0 ou 1.")
    target = int(lines[8])
    return pixels, target


def main() -> None:
    """Affiche les donnees lues, sans entrainement ni prediction."""
    data_directory = Path(__file__).resolve().parents[1] / "data"
    for filename in ("zero.txt", "un.txt"):
        pixels, target = read_pattern(data_directory / filename)
        print(
            f"{filename} : {len(pixels)} pixels, "
            f"cible = {target}, pixels allumes = {sum(pixels)}"
        )


if __name__ == "__main__":
    main()
