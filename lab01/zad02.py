import math
import random
import matplotlib.pyplot as plt

# Twoje zmienne
wysokosc = 100
predkosc = 50
kat = int(input("podaj kat strzalu (od 90 do 0): "))
cel = random.randint(50, 340)
print(f"Cel znajduje sie na {cel} metrze")
Czy_kontuować = True
counter = 0

# Stałe
g = 9.81  # przyspieszenie ziemskie w m/s^2
v0 = 50   # początkowa prędkość w m/s
h = 100   # wysokość w metrach

# Funkcja do obliczania zasięgu pocisku


def calculate_range(angle_deg):
    # Konwersja kąta na radiany
    angle_rad = math.radians(angle_deg)

    # Składowe prędkości
    v0x = v0 * math.cos(angle_rad)
    v0y = v0 * math.sin(angle_rad)

    # Czas lotu
    t = (v0y + math.sqrt(v0y**2 + 2 * g * h)) / g

    # Zasięg pocisku
    d = v0x * t
    return d, v0x, v0y, t


def plot_trajectory(v0x, v0y, time_of_flight):
    # Generowanie punktów czasu
    time_points = [t * 0.1 for t in range(int(time_of_flight * 10) + 1)]

    # Obliczenie pozycji x i y dla każdego punktu czasu
    x_points = [v0x * t for t in time_points]
    y_points = [h + v0y * t - 0.5 * g * t**2 for t in time_points]

    # Rysowanie wykresu
    plt.figure(figsize=(10, 5))
    plt.plot(x_points, y_points, color='blue')  # Niebieska linia trajektorii
    plt.title("Trajektoria pocisku Warwolf")
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.grid(True)  # Linie siatki
    plt.savefig("trajektoria.png")  # Zapis wykresu do pliku
    plt.show()


# Główna pętla strzelania
while Czy_kontuować:
    shot_distance, v0x, v0y, time_of_flight = calculate_range(kat)
    print(f"Kamień poleciał na odległość: {shot_distance:.2f} metrów.")

    if (cel - 5) <= shot_distance <= (cel + 5):
        print(f"Cel Trafiony! Ilosc strzalow: {counter + 1}")
        print(shot_distance)
        Czy_kontuować = False

        # wykres
        plot_trajectory(v0x, v0y, time_of_flight)
    else:
        print(shot_distance)
        kat = int(input("podaj nowy kat strzalu (od 90 do 0): "))
        counter += 1
