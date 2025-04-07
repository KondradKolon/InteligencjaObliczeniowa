import pygad
import numpy as np
import math

# Definiujemy funkcję fitness dostosowaną do PyGAD


def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)


# Parametry chromosomu - każdy gen w przedziale [0, 1]
gene_space = {'low': 0, 'high': 1}

# Parametry algorytmu genetycznego
sol_per_pop = 50  # Wielkość populacji
num_genes = 6     # Liczba genów (zmiennych x,y,z,u,v,w)
num_generations = 100  # Liczba pokoleń
num_parents_mating = 25  # Liczba rodziców do krzyżowania
keep_parents = 10  # Liczba rodziców do zachowania
parent_selection_type = "sss"  # Selekcja steady-state
crossover_type = "two_points"  # Krzyżowanie dwupunktowe
mutation_type = "random"  # Mutacja adaptacyjna
mutation_percent_genes = 10  # Początkowy procent mutacji
# teorytyczne maksimum 2.8415
# Inicjalizacja algorytmu
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria="saturate_25")  # Zatrzymaj jeśli brak poprawy przez 25 pokoleń

# Uruchomienie algorytmu
ga_instance.run()

# Podsumowanie wyników
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie:", solution)
print("Wartość funkcji fitness dla najlepszego rozwiązania:", solution_fitness)

# Wykres ewolucji fitness
ga_instance.plot_fitness(
    title="Ewolucja wartości fitness w kolejnych pokoleniach",
    save_dir="./wykreszad2.png"
)
