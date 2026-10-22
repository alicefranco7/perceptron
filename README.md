# Perceptron

TD de reconnaissance de chiffres, du neurone simple au reseau multicouche.

## Avancement

Etape 1 :

- termine : preparation et lecture des deux motifs binaires (`src/patterns.py`) ;
- termine : score, decision et correction par la regle du perceptron
  (`src/perceptron.py`), verifies sur le calcul papier du sujet ;
- termine : apprentissage par epoques, arret a zero erreur ou apres 1000
  epoques (`src/training.py`) ;
- a faire : copies bruitees, CSV et figure.

Les previsions et observations du groupe sont dans
`fiches/predis_avant_de_lancer.md`.

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
python3 -m src.perceptron
python3 -m src.training
python3 -m unittest discover -s tests -v
```

La premiere commande affiche 48 pixels pour chaque motif, 16 pixels allumes
pour le zero et 9 pour le un. Elle ne realise aucune prediction.

La deuxieme refait le calcul papier (x = [1, 0], w = [0.2, -0.1], b = -0.5,
d = 1 : a = -0.3, y = 0, e = 1, puis w0 = 0.21 et b = -0.49), puis montre une
seule correction sur chaque motif avec les poids initiaux de graine 42. Les
reglages (graine, biais initial, pas) sont des constantes en haut du fichier.

La troisieme entraine le neurone sur les deux motifs. Elle affiche une ligne
par epoque : ordre de presentation, erreurs recomptees a poids figes, biais et
score de chaque motif. La limite d'epoques (`MAX_EPOCHS`) est en haut de
`src/training.py`.
Les tests controlent aussi l'ordre des pixels et le rejet des fichiers invalides.
Ils sont executes automatiquement sur GitHub a chaque push et pull request.
