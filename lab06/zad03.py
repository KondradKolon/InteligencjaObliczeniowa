import pygad
import numpy as np

# Define the maze (S=2, E=3, walls=0)
maze = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

maze = np.array(maze)
start_pos = (1, 1)  # S position
end_pos = (10, 10)  # E position

# Genetic Algorithm Parameters
gene_space = [0, 1, 2, 3]  # 0=up, 1=right, 2=down, 3=left
chromosome_length = 50
population_size = 1500
num_parents_mating = 400
num_generations = 100
keep_parents = 400


def fitness_func(ga_instance, solution, solution_idx):
    current_pos = start_pos
    path_length = 0
    visited = set([current_pos])  # Track visited positions
    reached_exit = False

    for step in solution:
        new_pos = list(current_pos)
        if step == 0:
            new_pos[0] -= 1
        elif step == 1:
            new_pos[1] += 1
        elif step == 2:
            new_pos[0] += 1
        elif step == 3:
            new_pos[1] -= 1

        new_pos = tuple(new_pos)

        # Check boundaries, walls, and visited squares
        if (new_pos[0] < 0 or new_pos[0] >= 12 or
            new_pos[1] < 0 or new_pos[1] >= 12 or
            maze[new_pos] == 0 or
                new_pos in visited):
            break  # Illegal move - terminate path evaluation

        if new_pos == end_pos:
            reached_exit = True
            break  # Exit found - terminate path evaluation

        visited.add(new_pos)
        current_pos = new_pos
        path_length += 1

    # Fitness calculation
    if reached_exit:
        # Base score for reaching exit + bonus for longer paths
        fitness = 100 + (path_length * 500)
    else:
        # Score based on path length and proximity to exit
        distance = abs(current_pos[0] - end_pos[0]) + \
            abs(current_pos[1] - end_pos[1])
        fitness = path_length * 40 + ((30 - distance)*10)

    return fitness


# Run GA with mutation_probability
ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=population_size,
    num_genes=chromosome_length,
    parent_selection_type="sss",
    keep_parents=keep_parents,
    crossover_type="single_point",
    mutation_type="random",
    mutation_probability=0.05,
    stop_criteria="saturate_100"
)

ga_instance.run()

# Process best solution
solution, solution_fitness, _ = ga_instance.best_solution()
current_pos = start_pos
path = [current_pos]
steps = 0
move_names = ['Up', 'Right', 'Down', 'Left']
moves = []
visited = set([current_pos])

for step in solution:
    step = int(step)
    new_pos = list(current_pos)
    if step == 0:
        new_pos[0] -= 1
    elif step == 1:
        new_pos[1] += 1
    elif step == 2:
        new_pos[0] += 1
    elif step == 3:
        new_pos[1] -= 1

    new_pos = tuple(new_pos)

    # Check for illegal moves
    if (new_pos[0] < 0 or new_pos[0] >= 12 or
        new_pos[1] < 0 or new_pos[1] >= 12 or
        maze[new_pos] == 0 or
            new_pos in visited):
        break

    moves.append(move_names[step])
    path.append(new_pos)
    visited.add(new_pos)
    steps += 1
    current_pos = new_pos

    if new_pos == end_pos:
        break

# Print results
print("\n--- Maze Solution ---")
print(f"Path found with {steps} steps")
print("Move sequence:", ', '.join(moves))

# Visualize maze
print("\nMaze Visualization:")
for i in range(12):
    for j in range(12):
        if (i, j) == start_pos:
            print("S", end=" ")
        elif (i, j) == end_pos:
            print("E", end=" ")
        elif (i, j) in path:
            print("·", end=" ")
        elif maze[i, j] == 0:
            print("█", end=" ")
        else:
            print(" ", end=" ")
    print()
