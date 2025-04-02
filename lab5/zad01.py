import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -----------------------------------------------------------------------------
# a) STANDARD SCALER vs MINMAX SCALER
# -----------------------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Domyślnie: StandardScaler (centruje i skaluje do jednostkowego odchylenia)
scaler = StandardScaler()
# Test: MinMaxScaler (skalowanie do zakresu [0,1])
# scaler = MinMaxScaler()  # Lepszy gdy wszystkie cechy są w podobnym zakresie
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# b) ONEHOT ENCODER vs LABELBINARIZER
# -----------------------------------------------------------------------------
# Domyślnie: OneHotEncoder (lepszy dla wieloklasowej klasyfikacji)
encoder = OneHotEncoder(sparse_output=False)
# Test: LabelBinarizer (prostsza alternatywa dla problemów binarnych)
# encoder = LabelBinarizer()  # Dla 2 klas działa identycznie
# y_encoded = encoder.fit_transform(y)  # Uwaga: inna składnia dla LabelBinarizer
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# -----------------------------------------------------------------------------
# PODZIAŁ DANYCH
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.3,
    random_state=42  # Ziarno dla powtarzalności wyników
)

# -----------------------------------------------------------------------------
# c) ARCHITEKTURA MODELU
# -----------------------------------------------------------------------------
model = Sequential([
    # Warstwa wejściowa (4 neurony = 4 cechy)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),

    # Warstwy ukryte - testuj różne konfiguracje:
    # Domyślnie: 64 neurony z relu (dobry kompromis)
    Dense(64, activation='relu'),
    # Test 1: Więcej neuronów (128) - ryzyko przeuczenia
    # Dense(128, activation='relu'),
    # Test 2: Zmiana funkcji aktywacji na tanh
    # Dense(64, activation='tanh'),

    # Warstwa wyjściowa (3 neurony = 3 klasy)
    Dense(y_encoded.shape[1], activation='softmax')
])

# -----------------------------------------------------------------------------
# d) KOMPILACJA MODELU
# -----------------------------------------------------------------------------
# Domyślnie: Adam z learning_rate=0.001 (najlepszy dla większości przypadków)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    # Test 1: Wersja bez ręcznego ustawiania learning_rate
    # optimizer='adam',  # Domyślny learning_rate=0.001
    # Test 2: SGD z większym learning_rate
    # optimizer=SGD(learning_rate=0.01),  # Wolniejszy ale bardziej stabilny
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------------------------
# e) TRENING MODELU (BATCH SIZE)
# -----------------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    # Domyślnie: batch_size=32 (dobry kompromis)
    batch_size=32,
    # Test 1: Mniejszy batch (8) - dokładniejszy ale wolniejszy
    # batch_size=8,
    # Test 2: Większy batch (64) - szybszy ale mniej dokładny
    # batch_size=64,
    validation_split=0.2  # 20% danych do walidacji
)

# -----------------------------------------------------------------------------
# WYKRESY Z ANALIZĄ
# -----------------------------------------------------------------------------
# Zapis wykresu dokładności
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('accuracy_plot.png')  # Zapis do pliku
plt.close()  # Zamknięcie wykresu

# Zapis wykresu funkcji straty
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], 'b--', label='Train Loss')
plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('loss_plot.png')  # Zapis do pliku
plt.close()  # Zamknięcie wykresu

# Zapis wykresu łączonego
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], 'b-', label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r-', label='Val Acc')
plt.plot(history.history['loss'], 'b--', label='Train Loss')
plt.plot(history.history['val_loss'], 'r--', label='Val Loss')
plt.title('Training Metrics')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('combined_plot.png')  # Zapis do pliku
plt.close()  # Zamknięcie wykresu


# Analiza wyników:
# - Dobre dopasowanie: val_accuracy ~= train_accuracy
# - Przeuczenie: val_accuracy << train_accuracy
# - Niedouczenie: obie krzywe niskie

# -----------------------------------------------------------------------------
# EWALUACJA I ZAPIS
# -----------------------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nDokładność na zbiorze testowym: {test_accuracy*100:.2f}%")
print("WNIOSEK: Im wyższa wartość, tym lepiej (max 100%)")

model.save('iris_model.keras')
plot_model(model, to_file='model_arch.png', show_shapes=True)
print("test")