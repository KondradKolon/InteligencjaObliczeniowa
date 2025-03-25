# Import niezbędnych bibliotek
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# a) Wczytanie i podział danych
data = pd.read_csv('dane.csv')

# Konwersja etykiet na wartości liczbowe
data['class'] = data['class'].map({'tested_positive': 1, 'tested_negative': 0})

# Podział na cechy i etykiety
X = data.drop('class', axis=1)
y = data['class']

# Podział na zbiór treningowy (70%) i testowy (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123321)

# Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# b) Budowa modelu MLP z dwiema warstwami (6 i 3 neurony)
mlp = MLPClassifier(
    hidden_layer_sizes=(6, 3),  # Dwie warstwy: 6 i 3 neurony
    activation='relu',          # Funkcja aktywacji ReLU
    solver='adam',
    max_iter=500,               # c) Maksymalnie 500 iteracji
    random_state=123321,

)

# Trenowanie modelu
print("\nTrenowanie modelu...")
mlp.fit(X_train_scaled, y_train)

# d) Ewaluacja modelu
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print("\nWyniki ewaluacji:")
print(f"Dokładność modelu: {accuracy:.4f}")
print("\nMacierz błędu:")
print(conf_matrix)
print("\n[TN FP]")
print("[FN TP]")


# f) Analiza błędów
tn, fp, fn, tp = conf_matrix.ravel()
print("\nAnaliza błędów:")
print(f"False Positives (FP): {fp} - Zdrowi zdiagnozowani jako chorzy")
print(f"False Negatives (FN): {fn} - Chorzy zdiagnozowani jako zdrowi")


# Wizualizacja macierzy błędów
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['tested_negative (0)', 'tested_positive (1)'],
            yticklabels=['tested_negative (0)', 'tested_positive (1)'])
plt.xlabel('Przewidywane klasy')
plt.ylabel('Prawdziwe klasy')
plt.title('Macierz błędów')
plt.show()


plt.savefig("macierz_bledow.png")
