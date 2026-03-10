# Facial Emotion Recognition using CNN (FER-2013)

This project implements a **Convolutional Neural Network (CNN)** to classify human facial emotions using the **FER-2013 dataset**.  
The model learns patterns from grayscale facial images and predicts emotional states such as happiness, sadness, anger, surprise, and more.

The goal of this project is to explore **computer vision and deep learning techniques for emotion detection** using TensorFlow/Keras.

---

## Project Overview

Emotion recognition is a common task in computer vision used in:

- Human–computer interaction
- Mental health monitoring
- Smart classrooms
- Customer sentiment analysis
- AI assistants and robotics

This project trains a CNN model capable of predicting emotions from facial images.

---

## Dataset

The model uses the **FER-2013 dataset**, which contains facial images categorized into emotional classes.

Characteristics:

- Image size: **48 × 48 pixels**
- Format: **Grayscale**
- Multiple emotion classes
- Organized into **train and test directories**

Example classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Model Architecture

The model uses a **Convolutional Neural Network (CNN)** architecture.

Structure:

Input Layer  
→ Rescaling (Normalization)  
→ Conv2D (32 filters) + MaxPooling  
→ Conv2D (64 filters) + MaxPooling  
→ Conv2D (128 filters) + MaxPooling  
→ Flatten  
→ Dense (128 neurons)  
→ Dropout (0.3)  
→ Output Layer (Softmax)

Key features:

- Feature extraction through convolution layers
- Downsampling using pooling layers
- Regularization using dropout
- Softmax output for emotion classification

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

---

## Training Process

Steps followed in the pipeline:

1. Load dataset using `image_dataset_from_directory`
2. Resize images to **48×48**
3. Convert images to **grayscale**
4. Normalize pixel values
5. Train CNN model
6. Evaluate on test dataset
7. Visualize accuracy and loss
8. Save trained model

Training parameters:

- Batch Size: **32**
- Epochs: **15**
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**

---

## Model Evaluation

The model is evaluated using:

- Test Accuracy
- Test Loss
- Training vs Validation accuracy plots
- Training vs Validation loss plots

These visualizations help identify:

- Overfitting
- Underfitting
- Model convergence

---

## Emotion Prediction

The trained model can predict emotions from new images.

Example workflow:

1. Load image
2. Convert to grayscale
3. Resize to 48×48
4. Convert to array
5. Run model prediction
6. Return predicted emotion

---

## Model Export

The trained model is saved for reuse:


It can later be loaded using:

```python
from tensorflow.keras.models import load_model
model = load_model("fer2013_emotion_cnn.h5")
