import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Check dataset
print("YES folder images:", len(os.listdir('dataset/yes')))
print("NO folder images:", len(os.listdir('dataset/no')))

# Settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Load Training Data
train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    classes=['no', 'yes']
)

# Load Validation Data
val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    classes=['no', 'yes']
)

print("Class labels:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save Model
os.makedirs('model', exist_ok=True)
model.save('model/cancer_model.keras')
print("Model saved successfully!")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('model/accuracy_plot.png')
plt.show()
print("Done!")