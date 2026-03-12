"""
Module 1: Maze generation using iterative Depth-First Search (DFS)

Algorithm: Iterative DFS with backtracking.
Convention: 0 = wall (unvisited), 1 = path (visited)
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import os


DIRECTIONS = [
    (0, 1),   # 0: East
    (-1, 1),  # 1: North-East
    (-1, 0),  # 2: North
    (-1, -1), # 3: North-West
    (0, -1),  # 4: West
    (1, -1),  # 5: South-West
    (1, 0),   # 6: South
    (1, 1),   # 7: South-East
]


def create_empty_matrix(n):
    """
    Creates an n x n matrix initialized to 0.

    Input  : n (int) - matrix size
    Output : matrix (list of lists) - n x n matrix filled with 0

    Complexity: O(n^2)
    """
    matrix = []
    i = 0
    while i < n:
        row = []
        j = 0
        while j < n:
            row.append(0)
            j = j + 1
        matrix.append(row)
        i = i + 1
    return matrix


def count_visited_neighbors(maze, x, y, n):
    """
    Counts the number of already-visited neighbors (value = 1) around (x, y).

    Inputs :
        maze (list) : maze under construction
        x, y (int) : cell coordinates
        n (int)    : maze size

    Output : counter (int) - number of neighbors with value 1

    Complexity: O(1) - at most 8 neighbors
    """
    counter = 0
    i = 0
    while i < 8:
        dx = DIRECTIONS[i][0]
        dy = DIRECTIONS[i][1]
        vx = x + dx
        vy = y + dy

        if vx >= 0 and vx < n and vy >= 0 and vy < n:
            if maze[vx][vy] == 1:
                counter = counter + 1

        i = i + 1

    return counter


def get_eligible_neighbors(maze, x, y, n):
    """
    Finds all eligible neighbors of cell (x, y).

    A neighbor is eligible if:
    - It is within maze bounds
    - It has not been visited yet (value = 0)
    - It has exactly one visited neighbor (to prevent cycles)

    Inputs :
        maze (list) : maze under construction
        x, y (int) : current cell coordinates
        n (int)    : maze size

    Output : neighbors (list) - list of (nx, ny) tuples of eligible neighbors

    Complexity: O(1) - 8 neighbors x 8 sub-neighbors = 64 ops max
    """
    neighbors = []
    i = 0

    while i < 8:
        dx = DIRECTIONS[i][0]
        dy = DIRECTIONS[i][1]
        nx = x + dx
        ny = y + dy

        if nx >= 0 and nx < n and ny >= 0 and ny < n:
            if maze[nx][ny] == 0:
                num_visited = count_visited_neighbors(maze, nx, ny, n)
                if num_visited == 1:
                    neighbors.append((nx, ny))

        i = i + 1

    return neighbors


def generate_maze(n):
    """
    Generates an n x n maze using iterative DFS.

    Input  : n (int) - maze size
    Output : maze (list) - n x n matrix with 0 (wall) and 1 (path)

    Complexity: O(n^2) - each cell visited at most once
    """
    maze = create_empty_matrix(n)

    x_start = random.randint(0, n - 1)
    y_start = random.randint(0, n - 1)
    maze[x_start][y_start] = 1

    stack = []
    stack.append((x_start, y_start))

    while len(stack) > 0:
        current_cell = stack[len(stack) - 1]
        x_current = current_cell[0]
        y_current = current_cell[1]

        neighbors = get_eligible_neighbors(maze, x_current, y_current, n)

        if len(neighbors) > 0:
            index = random.randint(0, len(neighbors) - 1)
            chosen = neighbors[index]
            nx = chosen[0]
            ny = chosen[1]

            maze[nx][ny] = 1
            stack.append((nx, ny))
        else:
            stack.pop()

    return maze


def measure_generation_time(n):
    """
    Generates a maze and measures execution time.

    Input  : n (int) - maze size
    Output : (maze, elapsed) - tuple of maze and time in seconds

    Complexity: O(n^2)
    """
    start = time.perf_counter()
    maze = generate_maze(n)
    end = time.perf_counter()
    elapsed = end - start

    return maze, elapsed


def calculate_statistics(maze):
    """
    Computes statistics on a generated maze.

    Input  : maze (list) - n x n maze
    Output : dict - dictionary with statistics

    Complexity: O(n^2)
    """
    n = len(maze)
    path_cells = 0

    i = 0
    while i < n:
        j = 0
        while j < n:
            if maze[i][j] == 1:
                path_cells = path_cells + 1
            j = j + 1
        i = i + 1

    total_cells = n * n
    fill_rate = path_cells / total_cells

    stats = {
        'size': n,
        'path_cells': path_cells,
        'total_cells': total_cells,
        'fill_rate': fill_rate
    }

    return stats


def display_maze(maze, goal=None, filename=None):
    """
    Displays the maze as an image.
    - Walls (0) in black
    - Paths (1) in white
    - Goal in red

    Inputs :
        maze (list)     : n x n matrix
        goal (tuple)    : coordinates (x, y)
        filename (str)  : output file name

    Output : None (displays and optionally saves the image)

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

    if goal is not None:
        gx = goal[0]
        gy = goal[1]
        image[gx][gy] = [255, 0, 0]  # goal: red

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title('Maze ' + str(n) + 'x' + str(n))
    plt.axis('off')

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print('Image saved: ' + filename)

    plt.show()


def display_statistics(stats, elapsed):
    """
    Prints maze generation statistics.

    Inputs :
        stats (dict)   : maze statistics
        elapsed (float): generation time in seconds
    """
    print('Size: ' + str(stats['size']) + 'x' + str(stats['size']))
    print('Path cells: ' + str(stats['path_cells']) + ' / ' + str(stats['total_cells']))
    print('Fill rate: ' + str(round(stats['fill_rate'] * 100, 2)) + '%')
    print('Execution time: ' + str(round(elapsed, 6)) + ' seconds')
    print('')


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_create_empty_matrix():
    """Test: verifies the created matrix is n x n and filled with 0."""
    print('Test: create_empty_matrix')

    n = 5
    matrix = create_empty_matrix(n)

    assert len(matrix) == n

    i = 0
    while i < n:
        assert len(matrix[i]) == n
        j = 0
        while j < n:
            assert matrix[i][j] == 0
            j = j + 1
        i = i + 1

    print('  OK')
    print('')


def test_count_visited_neighbors():
    """Test: verifies visited neighbor counting."""
    print('Test: count_visited_neighbors')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    n = 3

    # Center cell (1,1): 4 visited neighbors (top, bottom, left, right)
    counter = count_visited_neighbors(maze, 1, 1, n)
    assert counter == 4

    # Corner cell (0,0): 3 visited neighbors (right, bottom, bottom-right)
    counter = count_visited_neighbors(maze, 0, 0, n)
    assert counter == 3

    print('  OK')
    print('')


def test_get_eligible_neighbors():
    """Test: verifies that only eligible neighbors are returned."""
    print('Test: get_eligible_neighbors')

    maze = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    n = 5

    neighbors = get_eligible_neighbors(maze, 1, 1, n)
    assert len(neighbors) == 8

    print('  OK')
    print('')


def test_generate_maze():
    """Test: verifies the generated maze is valid."""
    print('Test: generate_maze')

    n = 10
    maze = generate_maze(n)

    assert len(maze) == n
    assert len(maze[0]) == n

    counter = 0
    i = 0
    while i < n:
        j = 0
        while j < n:
            if maze[i][j] == 1:
                counter = counter + 1
            j = j + 1
        i = i + 1

    assert counter > 0

    print('  OK')
    print('')


def test_calculate_statistics():
    """Test: verifies statistics computation."""
    print('Test: calculate_statistics')

    maze = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]

    stats = calculate_statistics(maze)

    assert stats['size'] == 3
    assert stats['path_cells'] == 5
    assert stats['total_cells'] == 9

    print('  OK')
    print('')


if __name__ == "__main__":
    print('MODULE 1: Maze Generation by DFS')
    print('=' * 60)
    print('')

    mazes_dir = os.path.join('..', 'results', 'mazes')
    if not os.path.exists(mazes_dir):
        os.makedirs(mazes_dir)

    print('TESTS')
    print('-' * 60)
    test_create_empty_matrix()
    test_count_visited_neighbors()
    test_get_eligible_neighbors()
    test_generate_maze()
    test_calculate_statistics()
    print('All tests passed')
    print('')

    print('DEMO')
    print('-' * 60)
    sizes = [5, 50, 500]

    for size in sizes:
        print('Generating maze ' + str(size) + 'x' + str(size))

        maze, elapsed = measure_generation_time(size)
        stats = calculate_statistics(maze)
        display_statistics(stats, elapsed)

        goal_found = False
        attempts = 0
        while not goal_found and attempts < 100:
            x_goal = random.randint(0, size - 1)
            y_goal = random.randint(0, size - 1)
            if maze[x_goal][y_goal] == 1:
                goal_found = True
                goal = (x_goal, y_goal)
            attempts = attempts + 1

        if goal_found:
            filename = os.path.join(mazes_dir, 'maze_' + str(size) + 'x' + str(size) + '.png')
            display_maze(maze, goal, filename)

    print('=' * 60)
    print('END')
