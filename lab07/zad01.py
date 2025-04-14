import numpy as np
import math
from matplotlib import pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

# Funkcja dla pojedynczej cząstki - przyjmuje tablicę 6 wartości


def endurance_single(variables):
    x, y, z, u, v, w = variables
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

# Funkcja dla całego roju - przyjmuje tablicę cząstek


def endurance_for_swarm(variables):
    # ile czastek w swarm
    n_particles = variables.shape[0]
    #tworzymy tablice dlugosci n wypelniona wynkiami ( dla kazdej czastki potem bedzie wynik )
    j = np.zeros(n_particles)

    # Obliczamy wartość funkcji dla każdej cząstki
    for i in range(n_particles):
        j[i] = -endurance_single(variables[i])

    return j


# ograniczenia
x_min = np.zeros(6)  # [0, 0, 0, 0, 0, 0]
x_max = np.ones(6)   # [1, 1, 1, 1, 1, 1]
my_bounds = (x_min, x_max)


# parametry optimazera
options = {'c1': 0.5, 'c2': 0.4, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=6,
    options=options,
    bounds=my_bounds
)

# Wykonanie optymalizacji
cost, pos = optimizer.optimize(endurance_for_swarm, iters=1000)

# Wyświetlenie wyników
print(f"Najlepszy koszt: {-cost}")  # Zmieniamy znak z powrotem
print(f"Najlepsza pozycja: {pos}")

# Wizualizacja historii kosztu
cost_history = optimizer.cost_history
plt.figure(figsize=(10, 6))
plot_cost_history(cost_history)
plt.title("Historia kosztu w procesie optymalizacji")
plt.xlabel("Iteracja")
plt.ylabel("Koszt (odwrócony znak)")
plt.grid(True)
plt.show()
