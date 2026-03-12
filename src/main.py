"""
MAZE RUNNER PROJECT - Main entry point
Results are saved to results/final/
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from maze_generation import generate_maze
from dijkstra_solver import (
    dijkstra,
    build_directional_map,
    solve_maze
)
from genetic_algorithm import genetic_algorithm

MAZE_SIZE      = 100
RESULTS_DIR    = "../results/final"

# Genetic algorithm parameters
N_POP  = 2000   # population size
L_PROG = 1200   # max program length (genes)
TS     = 0.1    # selection rate (10%)
TM     = 0.3    # mutation rate
NG     = 1000   # number of generations


def plot_exploration(visit_matrix, title, filename):
    """Saves a heatmap of the exploration matrix."""
    plt.figure(figsize=(10, 10))
    plt.imshow(visit_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit frequency / Distance')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_solution(maze, path, title, filename):
    """Saves the maze image with the solution path in red."""
    n = len(maze)
    image = np.zeros((n, n, 3), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            if maze[i][j] == 1:
                image[i][j] = [255, 255, 255]  # path: white
            else:
                image[i][j] = [0, 0, 0]        # wall: black

    for (r, c) in path:
        image[r][c] = [255, 0, 0]  # solution: red

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fitness(history, title, filename):
    """Saves the GA fitness convergence curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['best'], label='Best Fitness (distance to goal)', color='blue')
    plt.plot(history['avg'], label='Average Fitness', linestyle='--', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Score (lower = better)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("==========================================")
    print(f"      MAZE RUNNER PROJECT : {MAZE_SIZE}x{MAZE_SIZE}")
    print("==========================================\n")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 1. MAZE GENERATION
    print(f"[1/3] Generating maze...")
    maze = generate_maze(MAZE_SIZE)

    valid_points = [(i, j) for i in range(MAZE_SIZE) for j in range(MAZE_SIZE) if maze[i][j] == 1]
    start = valid_points[0]
    goal  = valid_points[-1]

    # 2. DIJKSTRA
    print("\n[2/3] Running Dijkstra (reference solver)...")
    t0 = time.time()

    distance_map   = dijkstra(maze, goal)
    dir_map        = build_directional_map(distance_map)
    dijkstra_path  = solve_maze(maze, dir_map, start, goal)

    print(f" -> Done in {time.time() - t0:.4f} s")

    # Dijkstra exploration heatmap
    vis_dijkstra = np.array(distance_map, dtype=float)
    vis_dijkstra[vis_dijkstra == -1] = np.nan

    print(" -> Chart: Dijkstra Exploration")
    plot_exploration(
        vis_dijkstra,
        f"Dijkstra: Exploration Map (Distances)\nSize {MAZE_SIZE}x{MAZE_SIZE}",
        os.path.join(RESULTS_DIR, "1_Dijkstra_Exploration.png")
    )

    print(" -> Chart: Dijkstra Solution")
    plot_solution(
        maze, dijkstra_path,
        f"Dijkstra: Optimal Solution (Length {len(dijkstra_path)})",
        os.path.join(RESULTS_DIR, "2_Dijkstra_Solution.png")
    )

    # 3. GENETIC ALGORITHM
    print("\n[3/3] Running Genetic Algorithm...")
    t0 = time.time()

    best_ind, history, ga_path, found, all_paths = genetic_algorithm(
        maze, start, goal, N_POP, L_PROG, TS, TM, NG, use_pheromones=True
    )

    print(f" -> Done in {time.time() - t0:.4f} s")
    print(f" -> Result: {'FOUND' if found else 'FAILED'}")

    print(" -> Chart: GA Fitness")
    plot_fitness(
        history,
        "Genetic Algorithm: Fitness Convergence",
        os.path.join(RESULTS_DIR, "3_GA_Fitness.png")
    )

    # GA exploration heatmap
    ga_map = np.zeros((MAZE_SIZE, MAZE_SIZE))
    for path in all_paths:
        for (r, c) in path:
            ga_map[r][c] += 1

    print(" -> Chart: GA Exploration")
    plot_exploration(
        ga_map,
        f"Genetic Algorithm: Exploration Heatmap\n(Population: {N_POP}, Generations: {NG})",
        os.path.join(RESULTS_DIR, "4_GA_Exploration.png")
    )

    print(" -> Chart: GA Solution")
    plot_solution(
        maze, ga_path,
        f"Genetic Algorithm: Best Path (Length {len(ga_path)})",
        os.path.join(RESULTS_DIR, "5_GA_Solution.png")
    )

    print("\n==========================================")
    print(f"DONE. Open folder: {RESULTS_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()
