import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load only 1000 images (500 cats, 500 dogs)
data_dir = "dogs-cats-mini"
cat_files = [f for f in os.listdir(data_dir) if f.startswith('cat.')][:500]
dog_files = [f for f in os.listdir(data_dir) if f.startswith('dog.')][:500]
filenames = cat_files + dog_files
categories = ['cat']*500 + ['dog']*500

df = pd.DataFrame({'filename': filenames, 'category': categories})

# Split data (80% train, 20% validation)
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

# Simple image generator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
print("test")
batch_size = 32
target_size = (150, 150)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False
)

# Lightweight model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*target_size, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,  # Fewer epochs for quick testing
    validation_data=validation_generator,
    callbacks=[
        ModelCheckpoint('best_model_small.h5',
                        monitor='val_accuracy', save_best_only=True)
    ]
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

plt.savefig('training_curves.png', bbox_inches='tight')
plt.close()

# Evaluation
best_model = load_model('best_model.h5')
loss, accuracy = best_model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

# Sample predictions
plt.figure(figsize=(15, 15))
sample_files = validate_df['filename'].sample(9).values

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
