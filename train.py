import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_PATH = 'dataset/yes'  
NO_TUMOR_PATH = 'dataset/no'

# Check dataset
print("YES folder images:", len(os.listdir('dataset/yes')))
print("NO folder images:", len(os.listdir('dataset/no')))

# Settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixels 0-1
    validation_split=0.2,    # 20% for validation
    rotation_range=15,       # Rotate images slightly
    zoom_range=0.2,          # Zoom in/out
    horizontal_flip=True,    # Flip images
    width_shift_range=0.1,   # Shift left/right
    height_shift_range=0.1   # Shift up/down
)

# Load Training Data
train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    classes=['no', 'yes']    # no=0, yes=1
)

# Load Validation Data
val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    classes=['no', 'yes']    # no=0, yes=1
)

print("Class labels:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)


