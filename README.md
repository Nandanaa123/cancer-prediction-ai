# Brain Tumor Detection using CNN

A deep learning project that detects brain tumors from MRI images
using Convolutional Neural Networks (CNN).

## Project Overview

This project uses a CNN model built with TensorFlow and Keras to
classify brain MRI scans as either tumor or no tumor.
It was built as part of my AIML college project to explore
how deep learning can assist in medical image analysis.

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

cancer_prediction/
├── dataset/
│   ├── yes/    - MRI images with tumor
│   └── no/     - MRI images without tumor
├── model/      - Saved trained model
├── train.py    - CNN model training code
├── app.py      - Prediction script
└── README.md

## How to Run

1. Clone the repository
2. Install dependencies:
   pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow
3. Train the model:
   python train.py
4. Run prediction:
   python app.py

## Model Performance

- Training Accuracy: 73%
- Validation Accuracy: 78%

## About

This project was built by Nandana K T as part of an AIML
college project to learn real-world deep learning workflows
using VS Code and GitHub.