import math
from datetime import datetime

# Funkcja do obliczania biorytmu
def calculate_biorhythm(t, cycle):
    return math.sin(2 * math.pi * t / cycle)

# Funkcja do obliczania liczby dni od urodzenia
def calculate_days_since_birth(birth_date):
    today = datetime.today()
    delta = today - birth_date
    return delta.days

# Funkcja do sprawdzania, czy wynik biorytmu jest wysoki, niski czy neutralny
def check_biorhythm_status(value):
    if value > 0.5:
        return "wysoki", "Gratulacje! Masz dziś świetny wynik!"
    elif value < -0.5:
        return "niski", "Nie martw się, jutro będzie lepiej!"
    else:
        return "neutralny", "Twój wynik jest w normie."

# Główna część programu
def main():
    # Pytanie użytkownika o imię i datę urodzenia
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia (YYYY): "))
    month = int(input("Podaj miesiąc urodzenia (MM): "))
    day = int(input("Podaj dzień urodzenia (DD): "))

    # Obliczenie liczby dni od urodzenia
    birth_date = datetime(year, month, day)
    days_since_birth = calculate_days_since_birth(birth_date)

    # Obliczenie biorytmów
    physical = calculate_biorhythm(days_since_birth, 23)
    emotional = calculate_biorhythm(days_since_birth, 28)
    intellectual = calculate_biorhythm(days_since_birth, 33)

    # Wyświetlenie wyników
    print(f"\nWitaj, {name}!")
    print(f"Dziś jest twój {days_since_birth} dzień życia.")
    print(f"Twój fizyczny biorytm: {physical:.2f}")
    print(f"Twój emocjonalny biorytm: {emotional:.2f}")
    print(f"Twój intelektualny biorytm: {intellectual:.2f}")

    # Sprawdzenie statusu biorytmów
    physical_status, physical_message = check_biorhythm_status(physical)
    emotional_status, emotional_message = check_biorhythm_status(emotional)
    intellectual_status, intellectual_message = check_biorhythm_status(intellectual)

    print(f"\nTwój fizyczny biorytm jest {physical_status}. {physical_message}")
    print(f"Twój emocjonalny biorytm jest {emotional_status}. {emotional_message}")
    print(f"Twój intelektualny biorytm jest {intellectual_status}. {intellectual_message}")

if __name__ == "__main__":
    main()