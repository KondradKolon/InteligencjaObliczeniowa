from datetime import date
import math

# Pobranie danych od użytkownika
imie = input("Podaj imię: ")
rok = int(input("Podaj rok urodzenia: "))
miesiac = int(input("Podaj miesiąc urodzenia: "))
dzien = int(input("Podaj dzień urodzenia: "))

# Obliczenie liczby dni od daty urodzenia do dziś
dzisiaj = date.today()
data_urodzenia = date(rok, miesiac, dzien)
ilosc_dni = (dzisiaj - data_urodzenia).days
print(f"Liczba dni od urodzenia: {ilosc_dni}")

# Funkcje obliczające wartości fal biorytmicznych
def fala_fizyczna(t):
    return math.sin((2 * math.pi / 23) * t)

def fala_emocjonalna(t):
    return math.sin((2 * math.pi / 28) * t)

def fala_intelektualna(t):
    return math.sin((2 * math.pi / 33) * t)

# Wyświetlenie wartości fal biorytmicznych
print(f"Twoja fala fizyczna: {fala_fizyczna(ilosc_dni):.2f}")
print(f"Twoja fala emocjonalna: {fala_emocjonalna(ilosc_dni):.2f}")
print(f"Twoja fala intelektualna: {fala_intelektualna(ilosc_dni):.2f}")

# Funkcja sprawdzająca, czy jutro będzie lepiej
def czy_jutro_bedzie_lepiej(t, typ_fali, nazwa_fali):
    if typ_fali(t + 1) > typ_fali(t):
        print(f"Jutro będzie lepsza fala {nazwa_fali}.")
    else:
        print(f"Jutro fala {nazwa_fali} nie będzie lepsza.")

# Sprawdzenie wyników fal biorytmicznych i wyświetlenie odpowiednich komunikatów
if fala_fizyczna(ilosc_dni) > 0.5:
    print("Gratulacje, dobrego wyniku fizycznego!")
elif fala_fizyczna(ilosc_dni) < -0.5:
    print("Nie przejmuj się wynikiem fizycznej fali, to głupoty.")
    czy_jutro_bedzie_lepiej(ilosc_dni, fala_fizyczna, "fizyczna")

if fala_emocjonalna(ilosc_dni) > 0.5:
    print("Gratulacje, dobrego wyniku emocjonalnego!")
elif fala_emocjonalna(ilosc_dni) < -0.5:
    print("Nie przejmuj się wynikiem emocjonalnej fali, to głupoty.")
    czy_jutro_bedzie_lepiej(ilosc_dni, fala_emocjonalna, "emocjonalna")

if fala_intelektualna(ilosc_dni) > 0.5:
    print("Gratulacje, dobrego wyniku intelektualnego!")
elif fala_intelektualna(ilosc_dni) < -0.5:
    print("Nie przejmuj się wynikiem intelektualnej fali, to głupoty.")
    czy_jutro_bedzie_lepiej(ilosc_dni, fala_intelektualna, "intelektualna")
    
    
    '''
    Główne zmiany:
Nazwy zmiennych i funkcji: Zmieniono na snake_case.

Komunikaty: Dodano więcej informacji dla użytkownika.

Unikanie powtórzeń: Wprowadzono funkcję czy_jutro_bedzie_lepiej, aby uniknąć powtarzania kodu.

Formatowanie wyników: Wyniki fal są teraz wyświetlane z dokładnością do dwóch miejsc po przecinku.
    '''