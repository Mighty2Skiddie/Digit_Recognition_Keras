# Digit_Recognition_Keras
# 🧠 Digit Recognition using MNIST (with Keras & TensorFlow)

This project builds a **neural network classifier** that can recognize **handwritten digits (0–9)** using the famous **MNIST dataset**. It uses **TensorFlow Keras**, **NumPy**, **pandas**, **matplotlib**, and **scikit-learn** to preprocess the data, build, train, evaluate, and test the model.

## 🚀 Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mighty2Skiddie/Digit_Recognition_Keras/blob/main/Digit_Recognition_MNIST.ipynb)

---

## 📁 Dataset

- **`train.csv`**: Contains 42,000+ images (28x28 pixels flattened into 784 columns) with a `label` (digit from 0–9).
- **`test.csv`**: Contains similar image data **without labels** for prediction.

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

