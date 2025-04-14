import matplotlib.pyplot as plt
import random
import time
import numpy as np
from aco import AntColony

plt.style.use("dark_background")

# Coordinates for our TSP problem
COORDS = (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90),
    (87, 83), (23, 23), (13, 76), (53, 32), (63, 12),
    (99, 33), (1, 12), (2, 23), (31, 99)
)


def calculate_path_length(path):
    """Calculate the total length of a path"""
    total_length = 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        distance = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        total_length += distance
    # Add distance from last to first node to complete the cycle
    a, b = path[-1], path[0]
    total_length += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return total_length


def plot_path(path, title="Ant Colony Optimization Path", w=10, h=8):
    """Plot the path found by the ACO algorithm"""
    plt.figure(figsize=(w, h))

    # Plot all nodes
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)

    # Plot the path
    for i in range(len(path) - 1):
        plt.plot(
            (path[i][0], path[i + 1][0]),
            (path[i][1], path[i + 1][1]),
            'c-', linewidth=2
        )

    # Connect the last point to the first
    plt.plot(
        (path[-1][0], path[0][0]),
        (path[-1][1], path[0][1]),
        'c-', linewidth=2
    )

    plt.title(title)
    plt.axis("off")
    return plt.gcf()


def test_parameter_combinations():
    """Test different parameter combinations and record results"""

    # Parameter ranges to test
    ant_counts = [10, 50, 200, 400]
    alphas = [0.5, 1.0, 2.0, 5.0]  # Pheromone importance
    betas = [0.5, 1.0, 2.0, 5.0]   # Distance importance
    evaporation_rates = [0.1, 0.3, 0.5, 0.8]
    pheromone_constants = [100, 500, 1000, 5000]
    iterations_list = [50, 100, 200, 400]

    # Lists to store results
    results = []

    # Default parameters for reference
    default_params = {
        'ant_count': 300,
        'alpha': 0.7,
        'beta': 1.2,
        'pheromone_evaporation_rate': 0.40,
        'pheromone_constant': 1000.0,
        'iterations': 300
    }

    # Test default parameters
    colony = AntColony(COORDS, **default_params)
    default_path = colony.get_path()
    default_length = calculate_path_length(default_path)

    results.append({
        'params': default_params,
        'path_length': default_length,
        'path': default_path,
        'name': 'Default'
    })

    print(f"Default parameters - Path length: {default_length:.2f}")

    # Test parameter variations - one parameter at a time

    # 1. Vary ant_count
    for ant_count in ant_counts:
        params = default_params.copy()
        params['ant_count'] = ant_count

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Ant count: {ant_count}'
        })

        print(f"Ant count {ant_count} - Path length: {path_length:.2f}")

    # 2. Vary alpha
    for alpha in alphas:
        params = default_params.copy()
        params['alpha'] = alpha

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Alpha: {alpha}'
        })

        print(f"Alpha {alpha} - Path length: {path_length:.2f}")

    # 3. Vary beta
    for beta in betas:
        params = default_params.copy()
        params['beta'] = beta

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Beta: {beta}'
        })

        print(f"Beta {beta} - Path length: {path_length:.2f}")

    # 4. Vary evaporation rate
    for rate in evaporation_rates:
        params = default_params.copy()
        params['pheromone_evaporation_rate'] = rate

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Evaporation rate: {rate}'
        })

        print(f"Evaporation rate {rate} - Path length: {path_length:.2f}")

    # 5. Vary pheromone constant
    for constant in pheromone_constants:
        params = default_params.copy()
        params['pheromone_constant'] = constant

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Pheromone constant: {constant}'
        })

        print(
            f"Pheromone constant {constant} - Path length: {path_length:.2f}")

    # 6. Vary iterations
    for iterations in iterations_list:
        params = default_params.copy()
        params['iterations'] = iterations

        colony = AntColony(COORDS, **params)
        path = colony.get_path()
        path_length = calculate_path_length(path)

        results.append({
            'params': params,
            'path_length': path_length,
            'path': path,
            'name': f'Iterations: {iterations}'
        })

        print(f"Iterations {iterations} - Path length: {path_length:.2f}")

    # Find best and worst solutions
    results.sort(key=lambda x: x['path_length'])
    best_result = results[0]
    worst_result = results[-1]

    print("\n=== Results Summary ===")
    print(
        f"Best parameters: {best_result['name']} - Length: {best_result['path_length']:.2f}")
    print(
        f"Worst parameters: {worst_result['name']} - Length: {worst_result['path_length']:.2f}")

    # Plot best, default, and worst paths
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    for i in range(len(best_result['path']) - 1):
        path = best_result['path']
        plt.plot((path[i][0], path[i+1][0]),
                 (path[i][1], path[i+1][1]), 'g-', linewidth=2)
    plt.plot((path[-1][0], path[0][0]),
             (path[-1][1], path[0][1]), 'g-', linewidth=2)
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.title(
        f"Best: {best_result['name']}\nLength: {best_result['path_length']:.2f}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    for i in range(len(default_path) - 1):
        plt.plot((default_path[i][0], default_path[i+1][0]),
                 (default_path[i][1], default_path[i+1][1]), 'c-', linewidth=2)
    plt.plot((default_path[-1][0], default_path[0][0]),
             (default_path[-1][1], default_path[0][1]), 'c-', linewidth=2)
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.title(f"Default Parameters\nLength: {default_length:.2f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    for i in range(len(worst_result['path']) - 1):
        path = worst_result['path']
        plt.plot((path[i][0], path[i+1][0]),
                 (path[i][1], path[i+1][1]), 'r-', linewidth=2)
    plt.plot((path[-1][0], path[0][0]),
             (path[-1][1], path[0][1]), 'r-', linewidth=2)
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.title(
        f"Worst: {worst_result['name']}\nLength: {worst_result['path_length']:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('aco_parameter_comparison.png')
    plt.show()

    # Plot parameter influence charts
    create_parameter_charts(results, default_params)

    return results


def create_parameter_charts(results, default_params):
    """Create charts showing how each parameter affects solution quality"""

    # Group results by parameter type
    ant_counts = [r for r in results if 'Ant count' in r['name']]
    alphas = [r for r in results if 'Alpha' in r['name']]
    betas = [r for r in results if 'Beta' in r['name']]
    evap_rates = [r for r in results if 'Evaporation rate' in r['name']]
    pheromone_constants = [
        r for r in results if 'Pheromone constant' in r['name']]
    iterations = [r for r in results if 'Iterations' in r['name']]

    # Default result
    default_result = next(r for r in results if r['name'] == 'Default')

    # Set up the figure
    plt.figure(figsize=(18, 12))

    # 1. Ant Count
    plt.subplot(2, 3, 1)
    x_vals = [r['params']['ant_count'] for r in ant_counts]
    y_vals = [r['path_length'] for r in ant_counts]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Ant Count')
    plt.ylabel('Path Length')
    plt.title('Effect of Ant Count on Solution Quality')
    plt.grid(True, alpha=0.3)

    # 2. Alpha
    plt.subplot(2, 3, 2)
    x_vals = [r['params']['alpha'] for r in alphas]
    y_vals = [r['path_length'] for r in alphas]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Alpha (Pheromone Importance)')
    plt.ylabel('Path Length')
    plt.title('Effect of Alpha on Solution Quality')
    plt.grid(True, alpha=0.3)

    # 3. Beta
    plt.subplot(2, 3, 3)
    x_vals = [r['params']['beta'] for r in betas]
    y_vals = [r['path_length'] for r in betas]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Beta (Distance Importance)')
    plt.ylabel('Path Length')
    plt.title('Effect of Beta on Solution Quality')
    plt.grid(True, alpha=0.3)

    # 4. Evaporation Rate
    plt.subplot(2, 3, 4)
    x_vals = [r['params']['pheromone_evaporation_rate'] for r in evap_rates]
    y_vals = [r['path_length'] for r in evap_rates]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Pheromone Evaporation Rate')
    plt.ylabel('Path Length')
    plt.title('Effect of Evaporation Rate on Solution Quality')
    plt.grid(True, alpha=0.3)

    # 5. Pheromone Constant
    plt.subplot(2, 3, 5)
    x_vals = [r['params']['pheromone_constant'] for r in pheromone_constants]
    y_vals = [r['path_length'] for r in pheromone_constants]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Pheromone Constant')
    plt.ylabel('Path Length')
    plt.title('Effect of Pheromone Constant on Solution Quality')
    plt.grid(True, alpha=0.3)

    # 6. Iterations
    plt.subplot(2, 3, 6)
    x_vals = [r['params']['iterations'] for r in iterations]
    y_vals = [r['path_length'] for r in iterations]
    plt.plot(x_vals, y_vals, 'o-', linewidth=2)
    plt.axhline(y=default_result['path_length'],
                color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Path Length')
    plt.title('Effect of Iteration Count on Solution Quality')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aco_parameter_effects.png')
    plt.show()


if __name__ == "__main__":
    print("Starting parameter testing for Ant Colony Optimization...")
    results = test_parameter_combinations()
    print("Testing complete!")
