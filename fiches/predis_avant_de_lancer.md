# Prédis avant de lancer

Recopie de la fiche du sujet (`fiches.pdf`, pages 1 et 2). Pour chaque
question : écrire la prévision **avant** de lancer, puis le résultat observé.
Ne pas effacer une prévision fausse : la corriger en dessous.

Groupe : __________ Membres : ____________________

## Étape 1 · un neurone

Commit avant essai : `__________`

**1.** Avec x=[1,0], w=[0,2 ; −0,1], b=−0,5 et d=1 : quels sont a, ŷ, e, puis
w₀ et b après une correction η=0,01 ?

- Prévision / calcul :
- Résultat observé :
- Pour vérifier : `python3 -m src.perceptron` (premier bloc) ; test `PaperExampleTests`.

**2.** Une erreur nulle sur les deux motifs garantit-elle une erreur nulle à
10 % de bruit ? Pourquoi ?

- Prévision :
- Résultat observé :
- Pour vérifier : activité 4 (copies bruitées), pas encore écrite.

**3.** À 100 %, combien de pixels changent ? Est-ce encore une petite perturbation ?

- Prévision :
- Résultat observé :
- Pour vérifier : activité 4.

**4.** Un classifieur qui répond toujours 0 sur 50 zéros et 50 uns bruités à
50 % reconnaît quelle proportion ?

- Prévision :
- Résultat observé :
- Pour vérifier : activité 4.

### Prévisions du groupe pour l'activité 3 (apprentissage par époques)

Questions ajoutées par le groupe, à remplir avant `python3 -m src.training`.

- Avec la graine 42, quel motif est mal classé avant l'apprentissage ?
  Prévision : ____ Observé : ____
- Combien d'époques avant zéro erreur ? Prévision : ____ Observé : ____
  (Indice : score initial de `un.txt` et variation du score après une
  correction, affichés par `python3 -m src.perceptron`.)
- L'ordre de présentation change-t-il le résultat ici ? Prévision : ____ Observé : ____

### Journal des essais

Un essai = un seul réglage modifié en haut de `src/perceptron.py` ou de
`src/training.py`, puis relance.

| Date | Commit | Réglage modifié | Prévision | Observé |
|---|---|---|---|---|
| | | | | |
| | | | | |
