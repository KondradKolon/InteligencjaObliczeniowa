import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

# Ignoruj ostrzeżenia
warnings.filterwarnings("ignore")

# Wczytanie danych
data = pd.read_csv("iris1.csv")

# Konwersja nazw gatunków na liczby
target = data[['variety']].replace(
    ['setosa', 'versicolor', 'virginica'], [0, 1, 2])

# Wydzielenie cech i etykiet
features = data.drop('variety', axis=1)
labels = target['variety']

# Skalowanie danych
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Podział na zbiór treningowy i testowy
trainX, testX, trainY, testY = train_test_split(
    features_scaled, labels, test_size=0.3, random_state=295982)

# Funkcja do testowania modeli


def test_model(hidden_layers, max_iter=1000):
    model = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=295982
    )
    model.fit(trainX, trainY)
    return accuracy_score(testY, model.predict(testX))


# Testowanie różnych architektur i przechowywanie wyników w liście
results = [
    ["1 warstwa (2 neurony)", test_model((2,))],
    ["1 warstwa (3 neurony)", test_model((3,))],
    ["2 warstwy (3,3 neurony)", test_model((3, 3))]
]


print("Porównanie dokładności modeli:")
for items in results:
    print(f"{items[0]}: {items[1]:.4f}")


best_model = max(results, key=lambda x: x[1])
print(f"\nNajlepszy model: {best_model[0]} (dokładność: {best_model[1]:.4f})")

# print("\nMapowanie gatunków na liczby:")
# print("setosa -> 0")
# print("versicolor -> 1")
# print("virginica -> 2")
