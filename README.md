# Perceptron

TD de reconnaissance de chiffres, du neurone simple au reseau multicouche.

## Avancement

La premiere sous-etape est terminee : preparation et lecture des deux motifs
binaires. Le calcul du neurone et l'apprentissage restent a implementer.

## Prerequis

Python 3.10 ou plus recent. Cette sous-etape utilise uniquement la bibliotheque
standard de Python, sans dependance a installer.

## Format des donnees

`data/zero.txt` et `data/un.txt` reprennent les motifs du sujet : huit lignes de
six caracteres, puis la cible sur la neuvieme ligne. Un point devient 0 et une
etoile devient 1. Les pixels sont lus de gauche a droite puis de haut en bas.

`read_pattern(path)`, dans `src/patterns.py`, renvoie une liste de 48 entiers et
la cible separee. La cible n'est jamais ajoutee aux entrees du neurone.
Un format incorrect provoque une exception `ValueError`.

## Executer

Depuis la racine du depot :

```bash
python3 -m src.patterns
python3 -m unittest discover -s tests -v
```

La premiere commande affiche 48 pixels pour chaque motif, 16 pixels allumes
pour le zero et 9 pour le un. Elle ne realise aucune prediction.
Les tests controlent aussi l'ordre des pixels et le rejet des fichiers invalides.
Ils sont executes automatiquement sur GitHub a chaque push et pull request.
