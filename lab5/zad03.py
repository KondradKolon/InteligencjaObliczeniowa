import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and prepare data from dogs-cats-mini folder
data_dir = "dogs-cats-mini"
filenames = os.listdir(data_dir)
categories = []
for filename in filenames:
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Skip non-image files
        continue
    category = filename.split('.')[0]  # 'cat' or 'dog'
    categories.append(category)  # Store as string

df = pd.DataFrame({
    'filename': filenames,
    'category': categories  # Now contains strings 'cat'/'dog'
})

# Split data
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

# Image generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=(150, 150),
    class_mode='binary',  # Will automatically encode 'cat' as 0 and 'dog' as 1
    batch_size=32
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback to save best model
checkpoint = ModelCheckpoint('best_model.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

# Training
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')

plt.savefig('training_curves3xD.png', bbox_inches='tight')
plt.close()

# Evaluation
best_model = load_model('best_model.h5')
loss, accuracy = best_model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

# Sample predictions
plt.figure(figsize=(15, 15))
sample_files = validate_df['filename'].sample(9).values
print("test")
for i, filename in enumerate(sample_files):
    img_path = os.path.join(data_dir, filename)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)/255.0
    pred_prob = best_model.predict(np.expand_dims(img_array, axis=0))[0][0]
    pred = "DOG" if pred_prob > 0.5 else "CAT"
    confidence = max(pred_prob, 1-pred_prob)  # Get the confidence score

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {pred}\n({confidence:.2f})")
    plt.axis('off')

plt.savefig('sample_predictions.png', bbox_inches='tight')
plt.close()
