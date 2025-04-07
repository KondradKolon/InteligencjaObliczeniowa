import pygad
import numpy as np
import time

# Dane wejściowe
items = [
    {'nazwa': 'zegar', 'wartosc': 100, 'waga': 7},
    {'nazwa': 'obraz-pejzaż', 'wartosc': 300, 'waga': 7},
    {'nazwa': 'obraz-portret', 'wartosc': 200, 'waga': 6},
    {'nazwa': 'radio', 'wartosc': 40, 'waga': 2},
    {'nazwa': 'laptop', 'wartosc': 500, 'waga': 5},
    {'nazwa': 'lampka noona', 'wartosc': 70, 'waga': 6},
    {'nazwa': 'srebrne sztućce', 'wartosc': 100, 'waga': 1},
    {'nazwa': 'porcelana', 'wartosc': 250, 'waga': 3},
    {'nazwa': 'figura z brązu', 'wartosc': 300, 'waga': 10},
    {'nazwa': 'skórzana torebka', 'wartosc': 280, 'waga': 3},
    {'nazwa': 'odkurzacz', 'wartosc': 300, 'waga': 15}
]

max_weight = 25  # kg
best_possible_value = 1630  # Najlepsze możliwe rozwiązanie

# Przygotowanie danych dla PyGAD
values = np.array([item['wartosc'] for item in items])
weights = np.array([item['waga'] for item in items])
num_items = len(items)

# Definicja funkcji fitness


def fitness_func(model, solution, solution_idx):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)

    # Kara za przekroczenie limitu wagi
    if total_weight > max_weight:
        # Im większe przekroczenie, tym większa kara
        penalty = -10 * (total_weight - max_weight)
        return penalty
    else:
        # Jeśli waga jest OK, zwracamy całkowitą wartość
        return total_value


# Parametry algorytmu genetycznego - optymalne ustawienia
sol_per_pop = 50  # Liczba rozwiązań w populacji
num_generations = 200  # Liczba pokoleń
num_parents_mating = 20  # Liczba rodziców do krzyżowania
keep_parents = 5  # Liczba rodziców do zachowania
mutation_percent_genes = 10  # Procent mutowanych genów
mutation_type = "swap"  # Lepszy typ mutacji dla problemów plecakowych

# Funkcja do uruchamiania i mierzenia algorytmu


def run_ga_with_stats():
    start_time = time.time()

    # Inicjalizacja algorytmu
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_items,
        gene_space=[0, 1],
        parent_selection_type="sss",
        keep_parents=keep_parents,
        crossover_type="single_point",
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        # Zatrzymaj jeśli najlepsze rozwiązanie nie poprawia się przez 20 pokoleń
        stop_criteria=f"reach_1630"
    )

    ga_instance.run()

    end_time = time.time()
    runtime = end_time - start_time

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    total_weight = np.sum(solution * weights)

    return solution_fitness, runtime, solution, total_weight


# Testowanie skuteczności (10 prób)
success_count = 0
successful_runtimes = []

for i in range(10):
    print(f"\n--- Próba {i+1} ---")
    fitness, runtime, solution, weight = run_ga_with_stats()

    if fitness == best_possible_value:
        success_count += 1
        successful_runtimes.append(runtime)
        print(
            f"SUKCES! Znaleziono optymalne rozwiązanie (1630 zł) w czasie {runtime:.2f}s")
    else:
        print(
            f"Nie znaleziono optymalnego rozwiązania. Najlepsza wartość: {fitness} zł")

    # Wyświetlanie szczegółów rozwiązania
    selected_items = [items[i] for i in range(num_items) if solution[i] == 1]
    print("Wybrane przedmioty:")
    for item in selected_items:
        print(
            f"- {item['nazwa']} (wartość: {item['wartosc']}, waga: {item['waga']}kg")
    print(f"Łączna wartość: {fitness} zł")
    print(f"Łączna waga: {weight} kg")

# Obliczanie statystyk
success_rate = (success_count / 10) * 100
average_runtime = np.mean(successful_runtimes) if successful_runtimes else 0

print("\n--- Podsumowanie ---")
print(f"Liczba udanych prób (znaleziono 1630 zł): {success_count}/10")
print(f"Skuteczność: {success_rate:.0f}%")
print(f"Średni czas działania dla udanych prób: {average_runtime:.2f} sekund")

# Wykres ewolucji rozwiązania z ostatniej próby
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_items,
    gene_space=[0, 1],
    parent_selection_type="sss",
    keep_parents=keep_parents,
    crossover_type="single_point",
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    stop_criteria=f"reach_1630"
)
ga_instance.run()
ga_instance.plot_fitness(
    title="Ewolucja wartości fitness w kolejnych pokoleniach")
