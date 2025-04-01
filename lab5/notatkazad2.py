# Oto podsumowanie zadania i wykonanych poleceń krok po kroku:

# ZADANIE 2: Klasyfikacja cyfr MNIST w Keras
# Cel: Stworzenie i analiza modelu CNN do rozpoznawania ręcznie pisanych cyfr (0-9)

# Polecenia i ich realizacja:
# a) Preprocessing danych
# Polecenie: Wyjaśnij funkcje reshape, to_categorical i np.argmax
# Realizacja:

# python
# Copy
# # 1. reshape - dodaje wymiar kanału (1=skala szarości)
# train_images = train_images.reshape((60000, 28, 28, 1)) 

# # 2. to_categorical - zamienia etykiety na postać one-hot
# train_labels = to_categorical(train_labels) 

# # 3. np.argmax - odwraca one-hot na liczby
# original_labels = np.argmax(test_labels, axis=1) 
# Wyjaśnienie:

# Obrazy 28x28 px przekształcamy do kształtu (28,28,1) - ostatni wymiar to kanał koloru

# Etykiety (np. "5") stają się wektorami [0,0,0,0,0,1,0,0,0,0]

# argmax służy później do wizualizacji wyników

# b) Architektura modelu
# Polecenie: Opisz przepływ danych przez sieć
# Realizacja:

# python
# Copy
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), # Warstwa konwolucyjna
#     MaxPooling2D((2,2)), # Pooling zmniejszający wymiary
#     Flatten(), # Spłaszczenie do wektora
#     Dense(64, activation='relu'), # Warstwa gęsta
#     Dense(10, activation='softmax') # Warstwa wyjściowa
# ])
# Przepływ danych:

# Obraz 28x28x1 → konwolucja → 26x26x32

# Pooling → 13x13x32

# Spłaszczenie → 5408 elementów

# Warstwa gęsta → 64 neurony

# Wyjście → 10 neuronów (procentowe prawdopodobieństwa cyfr)

# c) Macierz błędów
# Polecenie: Zidentyfikuj najczęstsze błędy klasyfikacji
# Realizacja:

# python
# Copy
# cm = confusion_matrix(original_labels, predictions)
# sns.heatmap(cm, annot=True) # Wizualizacja macierzy
# Wynik:
# Confusion Matrix
# Typowe pomyłki:

# 4 ↔ 9 (podobny kształt)

# 5 ↔ 6 (częściowe podobieństwo)

# 3 ↔ 8

# d) Analiza krzywych uczenia
# Polecenie: Oceń czy model się przeucza
# Realizacja:

# python
# Copy
# plt.plot(history.history['accuracy'], label='Training')
# plt.plot(history.history['val_accuracy'], label='Validation')
# Interpretacja:

# Jeśli val_accuracy << train_accuracy → przeuczenie

# Jeśli obie krzywe niskie → niedouczenie

# W naszym przypadku: dobre dopasowanie (krzywe blisko siebie)

# e) Zapisywanie modelu
# Polecenie: Zapisz model gdy poprawia się dokładność walidacji
# Realizacja:

# python
# Copy
# checkpoint = ModelCheckpoint(
#     'best_model.h5',
#     monitor='val_accuracy',
#     save_best_only=True,
#     mode='max'
# )
# model.fit(..., callbacks=[checkpoint])
# Efekt:
# Plik best_model.h5 zawiera tylko najlepszą wersję modelu

# Dodatkowe elementy:
# Wizualizacja predykcji:

# python
# Copy
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(test_images[i].reshape(28,28), cmap='gray')
#     plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}", fontsize=8)
# Zapis wykresów:

# python
# Copy
# plt.savefig('nazwa_pliku.png', dpi=300, bbox_inches='tight')
# Podsumowanie:
# Wczytaliśmy zbiór MNIST (60k obrazów 28x28 px)

# Przetworzyliśmy dane (normalizacja, one-hot encoding)

# Zbudowaliśmy model CNN z:

# Warstwą konwolucyjną

# Poolingiem

# Warstwami gęstymi

# Wytrenowaliśmy z zapisem najlepszej wersji

# Przeanalizowaliśmy wyniki (macierz błędów, krzywe uczenia)

# Zwizualizowaliśmy przykładowe predykcje