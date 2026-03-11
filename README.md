# Maze Runner — Labyrinthe & Algorithme Génétique

**Projet Algorithmique & IA** | école d'ingénieurs — Oct. 2025–Jan. 2026 | Python

## Présentation

Résolution d'un labyrinthe 2D représenté sous forme de matrice NxN. Projet divisé en 3 sous-projets couvrant génération procédurale, résolution optimale par Dijkstra, et résolution par algorithme génétique inspiré de la sélection naturelle.

## SP1 — Génération du labyrinthe (DFS)

- Génération aléatoire par **DFS avec backtracking** (voisinage 8 directions)
- Tailles de 5×5 à 500×500
- Visualisation PIL/matplotlib

## SP2 — Résolution par Dijkstra

- Construction d'une **carte de distances** (distance map) depuis toutes les cellules
- Dérivation d'une **carte directionnelle** pour guider vers le but
- Chemin optimal depuis n'importe quel point de départ
- Analyse de la longueur moyenne du chemin selon N ∈ {8, 16, 32, 64, 128, 256, 512}

## SP3 — Algorithme Génétique

Inspiré de la sélection naturelle, avec une population de ~100 chemins candidats :

| Étape | Détail |
|:---|:---|
| **Génèse** | Population initiale de chemins aléatoires |
| **Fitness** | Distance au but + pénalités (collisions, longueur) |
| **Sélection** | Meilleurs individus promus |
| **Cross-over** | Coupure aléatoire autour du milieu, échange de segments |
| **Mutation** | Changement aléatoire de directions (gènes) |
| **Phéromones** | Mécanisme anti-minima-locaux, guide l'exploration |

- Visualisation de la **loss function** par génération
- Affichage du chemin solution final

## Stack

`Python` · `matplotlib` · `numpy` · `PIL`

---

*Projet académique — école d'ingénieurs*
