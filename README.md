# Maze Runner

Comparative study of two algorithmic approaches for solving randomly generated mazes: optimized Dijkstra (BFS) and a Genetic Algorithm with pheromones.

---

## Directory Tree

```
maze_runner/
├── src/
│   ├── maze_generation.py   # Module 1 – DFS maze generator
│   ├── dijkstra_solver.py   # Module 2 – Dijkstra/BFS solver
│   ├── genetic_algorithm.py # Module 3 – Genetic Algorithm solver
│   └── main.py              # Entry point (runs both solvers, saves results)
├── results/
│   ├── final/               # Output of main.py (5 PNG files)
│   ├── mazes/               # Output of maze_generation.py
│   ├── plots/               # Output of dijkstra_solver.py / genetic_algorithm.py
│   └── solutions/           # Output of dijkstra_solver.py / genetic_algorithm.py
└── README.md
```

---

## Implemented Components

| Module | Function | Description |
|---|---|---|
| `maze_generation` | `generate_maze(n)` | DFS iterative maze generation |
| `maze_generation` | `calculate_statistics(maze)` | Fill rate, cell counts |
| `dijkstra_solver` | `dijkstra(maze, goal)` | BFS distance map from goal |
| `dijkstra_solver` | `build_directional_map(distance_map)` | Per-cell best-direction pointer |
| `dijkstra_solver` | `solve_maze(maze, dir_map, start, goal)` | Path extraction |
| `dijkstra_solver` | `analyze_average_length(sizes, n)` | Statistical path length analysis |
| `genetic_algorithm` | `genetic_algorithm(...)` | GA main loop with pheromones |
| `genetic_algorithm` | `execute_program(maze, program, start)` | Individual simulation |
| `genetic_algorithm` | `calculate_fitness(...)` | Euclidean distance + collision + pheromone penalty |
| `main` | `main()` | End-to-end pipeline with result export |

---

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install numpy matplotlib
```

---

## How to Run

All commands must be run from the `src/` directory.

**Full pipeline (Dijkstra + GA, 100×100 maze):**
```bash
cd src
python main.py
```
Expected output: 5 PNG files in `results/final/`:
- `1_Dijkstra_Exploration.png` – distance heatmap
- `2_Dijkstra_Solution.png` – optimal path
- `3_GA_Fitness.png` – convergence curve
- `4_GA_Exploration.png` – population heatmap
- `5_GA_Solution.png` – best GA path

**Individual modules (with built-in tests and demos):**
```bash
python maze_generation.py       # tests + 3 demo mazes (5, 50, 500)
python dijkstra_solver.py       # tests + 3 demo resolutions + statistical analysis
python genetic_algorithm.py     # tests + 20×20 GA demo
```

---

## Design Notes

The maze is represented as an N×N binary matrix (0 = wall, 1 = path). The DFS generator enforces connectivity and acyclicity by requiring each new cell to have exactly one visited neighbor. Dijkstra is implemented as BFS (all edge weights = 1), achieving O(N²) rather than the standard O(N² log N). The genetic algorithm encodes individuals as sequences of direction indices (0–7); fitness combines Euclidean distance to the goal with a light collision penalty and optional pheromone penalties on dead-end zones (5% evaporation per generation). Dijkstra guarantees optimal paths in milliseconds; the GA trades optimality for exploratory flexibility but does not guarantee convergence, especially when program length L is shorter than the optimal path.

---

## References

- Cormen, T. H. et al. *Introduction to Algorithms*, 3rd ed. – BFS/Dijkstra
- Holland, J. H. *Adaptation in Natural and Artificial Systems* – Genetic Algorithms
- Dorigo, M. & Stützle, T. *Ant Colony Optimization* – Pheromone-based search
