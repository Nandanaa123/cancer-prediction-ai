import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load saved model
model = tf.keras.models.load_model('model/cancer_model.keras')

def predict_tumor(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Result
    if prediction > 0.5:
        result = "TUMOR DETECTED ⚠️"
        confidence = prediction * 100
    else:
        result = "NO TUMOR DETECTED ✅"
        confidence = (1 - prediction) * 100

    # Show image with result
    img_display = image.load_img(img_path)
    plt.imshow(img_display)
    plt.title(f"{result}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2f}%")

# Test with an image
img_path = input("Enter image path: ")
predict_tumor(img_path)