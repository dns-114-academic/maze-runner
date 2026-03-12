"""
Module 2: Maze solving using Dijkstra's algorithm

Optimized Dijkstra with BFS queue for O(N^2) complexity.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from maze_generation import generate_maze, DIRECTIONS
import os
from collections import deque


def initialize_map(maze, goal_pos):
    """
    Initializes the distance map for Dijkstra's algorithm.

    Convention:
    - None : cell not yet visited
    - -1   : wall (maze[i][j] == 0)
    - 0    : goal position
    - n>0  : distance to goal

    Complexity: O(n^2)
    """
    n = len(maze)
    distance_map = []

    i = 0
    while i < n:
        row = []
        j = 0
        while j < n:
            row.append(None)
            j = j + 1
        distance_map.append(row)
        i = i + 1

    # Mark walls
    i = 0
    while i < n:
        j = 0
        while j < n:
            if maze[i][j] == 0:
                distance_map[i][j] = -1
            j = j + 1
        i = i + 1

    # Mark goal
    i_goal = goal_pos[0]
    j_goal = goal_pos[1]
    distance_map[i_goal][j_goal] = 0

    return distance_map


def get_adjacent_neighbors(i, j, n):
    """
    Returns adjacent neighbors of cell (i, j).

    Complexity: O(1) - at most 8 neighbors
    """
    neighbors = []
    k = 0

    while k < 8:
        di = DIRECTIONS[k][0]
        dj = DIRECTIONS[k][1]
        vi = i + di
        vj = j + dj

        if vi >= 0 and vi < n and vj >= 0 and vj < n:
            neighbors.append((vi, vj))

        k = k + 1

    return neighbors


def dijkstra(maze, goal_pos):
    """
    Optimized Dijkstra (BFS) to compute distances from all cells to the goal.

    Optimization: uses a deque for O(N^2) complexity instead of O(N^4).

    Inputs :
        maze (list)     - n x n maze
        goal_pos (tuple) - (i, j) coordinates of the goal

    Output : distance_map (list) - matrix of distances to goal

    Complexity: O(N^2) - each cell processed exactly once
    """
    n = len(maze)
    distance_map = initialize_map(maze, goal_pos)

    queue = deque()
    queue.append(goal_pos)

    while len(queue) > 0:
        current_cell = queue.popleft()
        i = current_cell[0]
        j = current_cell[1]

        current_dist = distance_map[i][j]

        neighbors = get_adjacent_neighbors(i, j, n)

        k = 0
        while k < len(neighbors):
            vi = neighbors[k][0]
            vj = neighbors[k][1]

            if distance_map[vi][vj] is None:
                distance_map[vi][vj] = current_dist + 1
                queue.append((vi, vj))

            k = k + 1

    return distance_map


def build_directional_map(distance_map):
    """
    Builds a directional map from Dijkstra's distance map.
    Each cell stores the direction index pointing to the neighbor
    with the smallest distance to the goal.

    Complexity: O(n^2)
    """
    n = len(distance_map)
    dir_map = []

    i = 0
    while i < n:
        row = []
        j = 0
        while j < n:
            row.append(None)
            j = j + 1
        dir_map.append(row)
        i = i + 1

    i = 0
    while i < n:
        j = 0
        while j < n:
            current_val = distance_map[i][j]

            if current_val == -1:
                dir_map[i][j] = -1
            elif current_val == 0:
                dir_map[i][j] = None
            elif current_val is not None:
                min_dist = current_val
                min_dir = None

                neighbors = get_adjacent_neighbors(i, j, n)
                k = 0
                while k < len(neighbors):
                    vi = neighbors[k][0]
                    vj = neighbors[k][1]
                    neighbor_dist = distance_map[vi][vj]

                    if neighbor_dist is not None and neighbor_dist != -1:
                        if neighbor_dist < min_dist:
                            min_dist = neighbor_dist
                            di = vi - i
                            dj = vj - j
                            m = 0
                            while m < 8:
                                if DIRECTIONS[m][0] == di and DIRECTIONS[m][1] == dj:
                                    min_dir = m
                                    break
                                m = m + 1

                    k = k + 1

                dir_map[i][j] = min_dir

            j = j + 1
        i = i + 1

    return dir_map


def solve_maze(maze, dir_map, start_pos, goal_pos):
    """
    Solves the maze by following the directional map.

    Complexity: O(path_length)
    """
    path = []
    i_current = start_pos[0]
    j_current = start_pos[1]

    path.append((i_current, j_current))

    iterations = 0
    n = len(maze)
    max_iterations = n * n

    while iterations < max_iterations:
        if i_current == goal_pos[0] and j_current == goal_pos[1]:
            break

        direction = dir_map[i_current][j_current]

        if direction is None or direction == -1:
            break

        di = DIRECTIONS[direction][0]
        dj = DIRECTIONS[direction][1]
        i_current = i_current + di
        j_current = j_current + dj

        path.append((i_current, j_current))

        iterations = iterations + 1

    return path


def visualize_distances(maze, distance_map, goal_pos, filename=None):
    """
    Visualizes the distance map using colors (dark blue → yellow).

    Complexity: O(n^2)
    """
    n = len(maze)

    vis_matrix = []
    max_dist = 0

    i = 0
    while i < n:
        row = []
        j = 0
        while j < n:
            val = distance_map[i][j]
            if val is None:
                row.append(-1)
            elif val == -1:
                row.append(-1)
            else:
                row.append(val)
                if val > max_dist:
                    max_dist = val
            j = j + 1
        vis_matrix.append(row)
        i = i + 1

    image = np.zeros((n, n, 3), dtype=np.uint8)

    i = 0
    while i < n:
        j = 0
        while j < n:
            val = vis_matrix[i][j]
            if val == -1:
                image[i][j] = [0, 0, 0]  # wall: black
            else:
                if max_dist > 0:
                    ratio = val / max_dist
                else:
                    ratio = 0

                r = int(ratio * 255)
                g = int(ratio * 255)
                b = int((1 - ratio) * 255)
                image[i][j] = [r, g, b]

            j = j + 1
        i = i + 1

    i_goal = goal_pos[0]
    j_goal = goal_pos[1]
    image[i_goal][j_goal] = [255, 0, 0]  # goal: red

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('Distance map - Maze ' + str(n) + 'x' + str(n))
    plt.axis('off')

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print('Image saved: ' + filename)

    plt.show()


def visualize_solution(maze, path, goal_pos, filename=None):
    """
    Visualizes the maze with the solution path highlighted in red.

    Complexity: O(n^2)
    """
    n = len(maze)

    image = np.zeros((n, n, 3), dtype=np.uint8)

    i = 0
    while i < n:
        j = 0
        while j < n:
            if maze[i][j] == 1:
                image[i][j] = [255, 255, 255]  # path: white
            else:
                image[i][j] = [0, 0, 0]        # wall: black
            j = j + 1
        i = i + 1

    k = 0
    while k < len(path):
        i_path = path[k][0]
        j_path = path[k][1]
        image[i_path][j_path] = [255, 0, 0]  # solution: red
        k = k + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('Solution - Length: ' + str(len(path)))
    plt.axis('off')

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print('Image saved: ' + filename)

    plt.show()


def analyze_average_length(sizes, num_mazes):
    """
    Analyzes the evolution of average solution length by maze size.

    Complexity: O(len(sizes) * num_mazes * n^2)
    """
    results = {}

    i = 0
    while i < len(sizes):
        n = sizes[i]
        print('Analysis for N = ' + str(n))

        lengths = []
        j = 0

        while j < num_mazes:
            maze = generate_maze(n)

            start_found = False
            attempts = 0
            while not start_found and attempts < 100:
                i_start = random.randint(0, n - 1)
                j_start = random.randint(0, n - 1)
                if maze[i_start][j_start] == 1:
                    start_found = True
                attempts = attempts + 1

            goal_found = False
            attempts = 0
            while not goal_found and attempts < 100:
                i_goal = random.randint(0, n - 1)
                j_goal = random.randint(0, n - 1)
                if maze[i_goal][j_goal] == 1:
                    goal_found = True
                attempts = attempts + 1

            if start_found and goal_found:
                start_pos = (i_start, j_start)
                goal_pos = (i_goal, j_goal)

                distance_map = dijkstra(maze, goal_pos)
                dir_map = build_directional_map(distance_map)
                path = solve_maze(maze, dir_map, start_pos, goal_pos)

                if len(path) > 1:
                    lengths.append(len(path))

            j = j + 1

        if len(lengths) > 0:
            total = 0
            k = 0
            while k < len(lengths):
                total = total + lengths[k]
                k = k + 1

            average = total / len(lengths)

            min_len = lengths[0]
            max_len = lengths[0]
            k = 1
            while k < len(lengths):
                if lengths[k] < min_len:
                    min_len = lengths[k]
                if lengths[k] > max_len:
                    max_len = lengths[k]
                k = k + 1

            results[n] = {
                'average': average,
                'min': min_len,
                'max': max_len,
                'num_samples': len(lengths)
            }

            print('  Average length: ' + str(round(average, 2)))
            print('  Min: ' + str(min_len) + ', Max: ' + str(max_len))
        else:
            print('  No valid paths found')

        print('')
        i = i + 1

    return results


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_initialize_map():
    """Test: verifies distance map initialization."""
    print('Test: initialize_map')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    goal_pos = (1, 1)

    distance_map = initialize_map(maze, goal_pos)

    assert distance_map[1][1] == 0
    assert distance_map[0][0] == -1
    assert distance_map[0][2] == -1
    assert distance_map[0][1] is None
    assert distance_map[1][0] is None

    print('  OK')
    print('')


def test_dijkstra():
    """Test: verifies Dijkstra's algorithm."""
    print('Test: dijkstra')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    goal_pos = (1, 1)

    distance_map = dijkstra(maze, goal_pos)

    assert distance_map[1][1] == 0
    assert distance_map[0][1] == 1
    assert distance_map[1][0] == 1
    assert distance_map[1][2] == 1
    assert distance_map[2][1] == 1

    print('  OK')
    print('')


def test_build_directional_map():
    """Test: verifies directional map construction."""
    print('Test: build_directional_map')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    goal_pos = (1, 1)

    distance_map = dijkstra(maze, goal_pos)
    dir_map = build_directional_map(distance_map)

    assert dir_map[1][1] is None
    assert dir_map[0][0] == -1
    assert dir_map[0][1] >= 0 and dir_map[0][1] <= 7

    print('  OK')
    print('')


def test_solve_maze():
    """Test: verifies maze solving."""
    print('Test: solve_maze')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    start_pos = (0, 1)
    goal_pos = (2, 1)

    distance_map = dijkstra(maze, goal_pos)
    dir_map = build_directional_map(distance_map)
    path = solve_maze(maze, dir_map, start_pos, goal_pos)

    assert path[0] == start_pos
    assert path[len(path) - 1] == goal_pos
    assert len(path) >= 2

    print('  OK - Path length: ' + str(len(path)))
    print('')


if __name__ == "__main__":
    print('MODULE 2: Maze Solving by Dijkstra (Optimized)')
    print('=' * 60)
    print('')

    solutions_dir = os.path.join('..', 'results', 'solutions')
    plots_dir = os.path.join('..', 'results', 'plots')

    if not os.path.exists(solutions_dir):
        os.makedirs(solutions_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print('TESTS')
    print('-' * 60)
    test_initialize_map()
    test_dijkstra()
    test_build_directional_map()
    test_solve_maze()
    print('All tests passed')
    print('')

    print('DEMO')
    print('-' * 60)

    demo_sizes = [5, 50, 500]

    i = 0
    while i < len(demo_sizes):
        n = demo_sizes[i]
        print('Generating and solving maze ' + str(n) + 'x' + str(n))

        start = time.perf_counter()
        maze = generate_maze(n)
        end = time.perf_counter()
        print('  Generation time: ' + str(round(end - start, 6)) + ' seconds')

        start_found = False
        attempts = 0
        while not start_found and attempts < 100:
            i_start = random.randint(0, n - 1)
            j_start = random.randint(0, n - 1)
            if maze[i_start][j_start] == 1:
                start_found = True
            attempts = attempts + 1

        goal_found = False
        attempts = 0
        while not goal_found and attempts < 100:
            i_goal = random.randint(0, n - 1)
            j_goal = random.randint(0, n - 1)
            if maze[i_goal][j_goal] == 1:
                goal_found = True
            attempts = attempts + 1

        if start_found and goal_found:
            start_pos = (i_start, j_start)
            goal_pos = (i_goal, j_goal)

            print('  Start: ' + str(start_pos))
            print('  Goal : ' + str(goal_pos))

            t0 = time.perf_counter()
            distance_map = dijkstra(maze, goal_pos)
            t1 = time.perf_counter()
            print('  Dijkstra time: ' + str(round(t1 - t0, 6)) + ' seconds')

            dir_map = build_directional_map(distance_map)
            path = solve_maze(maze, dir_map, start_pos, goal_pos)
            print('  Path length: ' + str(len(path)))

            dist_file = os.path.join(plots_dir, 'distances_' + str(n) + 'x' + str(n) + '.png')
            visualize_distances(maze, distance_map, goal_pos, dist_file)

            sol_file = os.path.join(solutions_dir, 'solution_' + str(n) + 'x' + str(n) + '.png')
            visualize_solution(maze, path, goal_pos, sol_file)
        else:
            print('  Error: could not find valid start and/or goal')

        print('')
        i = i + 1

    print('STATISTICAL ANALYSIS')
    print('-' * 60)
    print('Average path length vs maze size')
    print('')

    analysis_sizes = [8, 16, 32, 64, 128, 256, 512]
    results = analyze_average_length(analysis_sizes, 10)

    sizes_list = []
    averages = []

    for size in sorted(results.keys()):
        sizes_list.append(size)
        averages.append(results[size]['average'])

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_list, averages, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Maze size (N)')
    plt.ylabel('Average path length')
    plt.title('Average path length vs maze size')
    plt.grid(True)
    chart_file = os.path.join(plots_dir, 'average_length_analysis.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print('Chart saved: ' + chart_file)
    plt.show()

    print('=' * 60)
    print('END')
