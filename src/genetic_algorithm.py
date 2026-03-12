"""
Module 3: Maze solving using a Genetic Algorithm with pheromones
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from maze_generation import generate_maze, DIRECTIONS
from dijkstra_solver import dijkstra, initialize_map


# Default constants
POPULATION_SIZE = 1000    # N
PROGRAM_LENGTH  = 100     # L
SELECTION_RATE  = 0.3     # ts
MUTATION_RATE   = 0.1     # tm
MAX_GENERATIONS = 500     # nG


def create_random_individual(length):
    """
    Creates a random individual (program).
    """
    individual = []
    i = 0
    while i < length:
        gene = random.randint(0, 7)
        individual.append(gene)
        i = i + 1
    return individual


def initialize_population(population_size, program_length):
    """
    Creates the initial population.
    """
    population = []
    i = 0
    while i < population_size:
        individual = create_random_individual(program_length)
        population.append(individual)
        i = i + 1
    return population


def execute_program(maze, program, start, goal=None):
    """
    Executes a movement program on the maze from `start`.
    Returns the path followed and whether a collision occurred.
    """
    n = len(maze)
    i, j = start
    path = [(i, j)]
    collision = False
    visited = {(i, j)}

    deltas = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    idx = 0
    max_steps = len(program)

    while idx < max_steps:
        move = program[idx]
        di, dj = deltas[move]

        ni, nj = i + di, j + dj

        if 0 <= ni < n and 0 <= nj < n and maze[ni][nj] == 1:
            i, j = ni, nj
            path.append((i, j))

            if goal and i == goal[0] and j == goal[1]:
                return path, collision
        else:
            collision = True

        idx += 1

    return path, collision


def calculate_fitness(individual, maze, start, goal, pheromones=None):
    """
    Fitness: penalizes walls lightly to encourage exploration.
    """
    path, collision = execute_program(maze, individual, start, goal)
    last_pos = path[-1]

    dist_goal = ((last_pos[0] - goal[0])**2 + (last_pos[1] - goal[1])**2)**0.5

    penalty = 0
    if collision:
        penalty += 0.1

    if pheromones is not None:
        for pos in path:
            if pheromones[pos[0]][pos[1]] > 0:
                penalty += 0.1 * pheromones[pos[0]][pos[1]]

    return dist_goal + penalty


def select_best(population, scores, selection_rate):
    """
    Selects elite individuals.
    """
    population_score = []
    i = 0
    while i < len(population):
        population_score.append((scores[i], population[i]))
        i = i + 1

    population_score.sort(key=lambda x: x[0])

    num_to_keep = int(len(population) * selection_rate)
    if num_to_keep < 2:
        num_to_keep = 2

    elites = []
    i = 0
    while i < num_to_keep:
        elites.append(population_score[i][1])
        i = i + 1

    return elites


def crossover(parent1, parent2):
    """
    Single-point crossover with cut point in [L/3, 2L/3].
    """
    L = len(parent1)
    min_cut = L // 3
    max_cut = 2 * L // 3
    cut = random.randint(min_cut, max_cut)
    child = parent1[:cut] + parent2[cut:]
    return child


def mutation(individual, mutation_rate):
    """
    Random mutation.
    """
    mutated = list(individual)
    num_mutations = int(len(individual) * mutation_rate)
    if num_mutations < 1 and random.random() < mutation_rate:
        num_mutations = 1

    k = 0
    while k < num_mutations:
        pos = random.randint(0, len(individual) - 1)
        mutated[pos] = random.randint(0, 7)
        k = k + 1
    return mutated


def genetic_algorithm(maze, start, goal, N, L, ts, tm, nG, use_pheromones=True):
    """
    Main genetic algorithm loop.

    Parameters:
        maze            : n x n maze
        start           : starting position (i, j)
        goal            : goal position (i, j)
        N               : population size
        L               : program length (number of genes)
        ts              : selection rate
        tm              : mutation rate
        nG              : max generations
        use_pheromones  : whether to use pheromone penalties
    """
    population = initialize_population(N, L)
    n = len(maze)
    pheromones = [[0.0] * n for _ in range(n)]

    fitness_history = {'best': [], 'avg': []}
    best_overall_score = float('inf')
    best_overall_individual = None

    generation = 0
    solution_found = False

    while generation < nG and not solution_found:
        scores = []
        paths = []
        score_sum = 0
        best_gen_score = float('inf')

        for ind in population:
            path, collision = execute_program(maze, ind, start, goal)
            paths.append((path, collision))

            last_pos = path[-1]
            dist = ((last_pos[0] - goal[0])**2 + (last_pos[1] - goal[1])**2)**0.5

            penalty = 0
            if collision:
                penalty += 0.5

            if use_pheromones:
                for pos in path:
                    penalty += 2.0 * pheromones[pos[0]][pos[1]]

            score = dist + penalty
            scores.append(score)
            score_sum += score

            if score < best_gen_score:
                best_gen_score = score

            if score < best_overall_score:
                best_overall_score = score
                best_overall_individual = list(ind)

            if last_pos == goal:
                solution_found = True

        avg_score = score_sum / N
        fitness_history['best'].append(best_gen_score)
        fitness_history['avg'].append(avg_score)

        if solution_found:
            break

        # Pheromone update: bad individuals deposit on their final position
        if use_pheromones:
            i = 0
            while i < N:
                if scores[i] > avg_score:
                    bad_path = paths[i][0]
                    end_pos = bad_path[-1]
                    pheromones[end_pos[0]][end_pos[1]] += 1.0
                i = i + 1

            # Evaporation
            for i in range(n):
                for j in range(n):
                    pheromones[i][j] *= 0.95

        # Reproduction
        elites = select_best(population, scores, ts)
        new_population = list(elites)
        num_children = N - len(elites)

        k = 0
        while k < num_children:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = crossover(p1, p2)
            child = mutation(child, tm)
            new_population.append(child)
            k = k + 1

        population = new_population
        generation += 1

    all_paths = [c[0] for c in paths]

    best_path, _ = execute_program(maze, best_overall_individual, start, goal)
    return best_overall_individual, fitness_history, best_path, solution_found, all_paths


def visualize_fitness(history, filename):
    """Plots the fitness evolution over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['best'], label='Best Fitness')
    plt.plot(history['avg'], label='Average Fitness', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_exploration(explored_paths, size, start, goal, filename):
    """
    Generates a heatmap of cells visited by the entire population.
    """
    exploration_map = np.zeros((size, size))

    for path in explored_paths:
        for (x, y) in path:
            exploration_map[x][y] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(exploration_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit frequency')

    plt.scatter(start[1], start[0], c='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], c='blue', s=100, label='Goal')

    plt.title('Exploration Map (Heatmap)')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_calculate_fitness():
    """Test: verifies fitness calculation."""
    print('Test: calculate_fitness')
    maze = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    # Program that moves South twice: from (0,1) to (2,1) = goal
    winning_program = [4, 4]

    score = calculate_fitness(winning_program, maze, (0, 1), (2, 1), None)

    if score < 1.0:
        print('  OK')
    else:
        print('  Test failed (score too high: ' + str(score) + ')')


if __name__ == "__main__":
    print('MODULE 3: Genetic Algorithm')
    print('=' * 60)

    plots_dir = os.path.join('..', 'results', 'plots')
    solutions_dir = os.path.join('..', 'results', 'solutions')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(solutions_dir):
        os.makedirs(solutions_dir)

    print('TESTS')
    test_calculate_fitness()
    print('Tests passed\n')

    n = 20
    print('Generating maze ' + str(n) + 'x' + str(n))
    maze = generate_maze(n)

    valid_points = []
    i = 0
    while i < n:
        j = 0
        while j < n:
            if maze[i][j] == 1:
                valid_points.append((i, j))
            j = j + 1
        i = i + 1

    if len(valid_points) > 2:
        start = random.choice(valid_points)

        goal = start
        max_dist = -1

        for pt in valid_points:
            d = abs(pt[0] - start[0]) + abs(pt[1] - start[1])
            if d > max_dist:
                max_dist = d
                goal = pt

        print('Start:', start, 'Goal:', goal, '(Distance:', max_dist, ')')

        N_pop = 1000
        L_prog = 400
        ts = 0.1
        tm = 0.2
        nG = 500

        print('Running GA...')
        t0 = time.time()
        best_ind, hist, best_path, found, _ = genetic_algorithm(
            maze, start, goal, N_pop, L_prog, ts, tm, nG
        )
        t1 = time.time()

        print('Time:', round(t1 - t0, 2), 's')
        print('Solution found:', found)
        print('Path length:', len(best_path))

        visualize_fitness(hist, os.path.join(plots_dir, 'fitness_ga.png'))

        from dijkstra_solver import visualize_solution
        visualize_solution(maze, best_path, goal, os.path.join(solutions_dir, 'solution_ga.png'))

    print('END')
