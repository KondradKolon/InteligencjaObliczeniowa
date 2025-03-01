import math
import random
import matplotlib.pyplot as plt

# Stałe
GRAVITY = 9.81  # przyspieszenie ziemskie w m/s^2
INITIAL_VELOCITY = 50  # początkowa prędkość w m/s
INITIAL_HEIGHT = 100  # wysokość w metrach

# Funkcja do obliczania zasięgu i trajektorii pocisku
def calculate_trajectory(angle_deg):
    angle_rad = math.radians(angle_deg)  # Konwersja kąta na radiany
    v0x = INITIAL_VELOCITY * math.cos(angle_rad)  # Składowa pozioma prędkości
    v0y = INITIAL_VELOCITY * math.sin(angle_rad)  # Składowa pionowa prędkości
    time_of_flight = (v0y + math.sqrt(v0y**2 + 2 * GRAVITY * INITIAL_HEIGHT)) / GRAVITY  # Czas lotu
    distance = v0x * time_of_flight  # Zasięg pocisku
    return distance, v0x, v0y, time_of_flight

# Funkcja do rysowania trajektorii
def plot_trajectory(v0x, v0y, time_of_flight):
    time_points = [t * 0.1 for t in range(int(time_of_flight * 10) + 1)]  # Punkty czasu
    x_points = [v0x * t for t in time_points]  # Pozycje x
    y_points = [INITIAL_HEIGHT + v0y * t - 0.5 * GRAVITY * t**2 for t in time_points]  # Pozycje y

    plt.figure(figsize=(10, 5))
    plt.plot(x_points, y_points, color='blue')  # Niebieska linia trajektorii
    plt.title("Trajektoria pocisku Warwolf")
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.grid(True)  # Linie siatki
    plt.savefig("trajektoria.png")  # Zapis wykresu do pliku
    plt.show()

# Główna funkcja programu
def main():
    target_distance = random.randint(50, 340)  # Losowanie pozycji celu
    print(f"Cel znajduje się na {target_distance} metrze.")

    shot_counter = 0
    while True:
        angle = int(input("Podaj kąt strzału (od 90 do 0): "))  # Pobranie kąta od użytkownika
        distance, v0x, v0y, time_of_flight = calculate_trajectory(angle)  # Obliczenie trajektorii
        print(f"Kamień poleciał na odległość: {distance:.2f} metrów.")

        if abs(distance - target_distance) <= 5:  # Sprawdzenie, czy cel został trafiony
            print(f"Cel trafiony! Ilość strzałów: {shot_counter + 1}")
            plot_trajectory(v0x, v0y, time_of_flight)  # Rysowanie trajektorii
            break
        else:
            print("Nie trafiono. Spróbuj ponownie.")
            shot_counter += 1

# Uruchomienie programu
if __name__ == "__main__":
    main()