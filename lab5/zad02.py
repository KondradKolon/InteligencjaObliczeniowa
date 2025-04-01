import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History

# -----------------------------------------------------------------------------
# a) PREPROCESSING - WYJAŚNIENIE
# -----------------------------------------------------------------------------
# 1. reshape - zmienia kształt obrazów z (28,28) na (28,28,1) - dodaje kanał dla koloru (1=szary)
# 2. to_categorical - zamienia etykiety na postać one-hot (np. 5 → [0,0,0,0,0,1,0,0,0,0])
# 3. np.argmax - odwrotna operacja do to_categorical (zamienia one-hot z powrotem na liczby)

# Wczytanie danych MNIST (cyfry 0-9, obrazy 28x28 pikseli)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Przetwarzanie obrazów
train_images = train_images.reshape(
    (train_images.shape[0], 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape(
    (test_images.shape[0], 28, 28, 1)).astype("float32") / 255
# Dzielimy przez 255 aby znormalizować piksele do zakresu [0,1]

# Przetwarzanie etykiet
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(
    test_labels, axis=1)  # Zapis oryginalnych etykiet

# -----------------------------------------------------------------------------
# b) ARCHITEKTURA SIECI - PRZEPŁYW DANYCH
# -----------------------------------------------------------------------------
model = Sequential([
    # Warstwa konwolucyjna 1:
    # - Wejście: (28, 28, 1)
    # - Wyjście: (26, 26, 32) - 32 filtry 3x3
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Warstwa pooling:
    # - Wejście: (26, 26, 32)
    # - Wyjście: (13, 13, 32) - zmniejsza wymiary 2x
    MaxPooling2D((2, 2)),

    # Warstwa spłaszczająca:
    # - Wejście: (13, 13, 32)
    # - Wyjście: (5408,) - zamienia na wektor
    Flatten(),

    # Warstwa gęsta:
    # - Wejście: (5408,)
    # - Wyjście: (64,) - 64 neurony
    Dense(64, activation='relu'),

    # Warstwa wyjściowa:
    # - Wejście: (64,)
    # - Wyjście: (10,) - 10 neuronów (po 1 na każdą cyfrę)
    Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------------------------------------------------------
# e) ZAPIS MODELU CO EPOKĘ (JEŚLI LEPszy WYNIK)
# -----------------------------------------------------------------------------
# Krok 1: Stwórz callback ModelCheckpoint
checkpoint = ModelCheckpoint(
    'best_modelzad2.h5',          # Nazwa pliku do zapisu
    monitor='val_accuracy',   # Monitoruj dokładność walidacji
    save_best_only=True,      # Zapisz tylko lepsze modele
    mode='max',               # Tryb maksymalizacji metryki
    verbose=1                 # Pokazuj komunikaty
)

# Krok 2: Dodaj callback do funkcji fit
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint]    # Dodajemy nasz callback
)

# -----------------------------------------------------------------------------
# c) MACIERZ BŁĘDÓW - ANALIZA
# -----------------------------------------------------------------------------
# Ewaluacja modelu
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predykcje
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Macierz błędów
cm = confusion_matrix(original_test_labels, predicted_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# WNIOSKI:
# Najczęstsze błędy to zwykle mylenie:
# - 4 ↔ 9
# - 5 ↔ 6
# - 3 ↔ 8
# - 7 ↔ 1

# -----------------------------------------------------------------------------
# d) ANALIZA KRZYWYCH UCZENIA
# -----------------------------------------------------------------------------
# c) MACIERZ BŁĘDÓW - ANALIZA Z ZAPISEM
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Zapis macierzy błędów
plt.close()  # Zamknięcie wykresu

# -----------------------------------------------------------------------------
# d) ANALIZA KRZYWYCH UCZENIA Z ZAPISEM
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
print("test")
# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Wykres straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')  # Zapis krzywych uczenia
plt.close()  # Zamknięcie wykresu

# -----------------------------------------------------------------------------
# WIZUALIZACJA PREDYKCJI Z ZAPISEM
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 12))  # Zwiększony rozmiar całego wykresu

# Ustawienie odstępów między podwykresami
plt.subplots_adjust(
    wspace=0.4,  # Pozioma odległość między obrazkami
    hspace=0.6,  # Pionowa odległość między obrazkami
    left=0.1,    # Margines lewy
    right=0.9,   # Margines prawy
    top=0.9,     # Margines górny
    bottom=0.1   # Margines dolny
)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)

    # Lepsze formatowanie etykiet
    plt.xlabel(
        f"True: {original_test_labels[i]}\nPred: {predicted_labels[i]}",
        fontsize=8,  # Mniejsza czcionka
        labelpad=2   # Mniejszy odstęp od obrazka
    )

    # Zwiększenie odstępów wokół każdego podwykresu
    plt.tight_layout(pad=0.5)

# Zapis z wysoką rozdzielczością
plt.savefig('predictions_samples.png', dpi=300, bbox_inches='tight')
plt.close()

print("Wszystkie wykresy zostały zapisane do plików:")
print("- confusion_matrix.png")
print("- training_curves.png")
print("- predictions_samples.png")
