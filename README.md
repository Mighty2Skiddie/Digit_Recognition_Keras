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
🧹 Step-by-Step Workflow
1. Load the Data
python
Copy
Edit
data = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
2. Separate Features and Labels
python
Copy
Edit
X = data.drop('label', axis=1)
y = data['label'].values
3. Visualize a Sample Digit
python
Copy
Edit
img = X.iloc[41].values.reshape((28, 28))
plt.imshow(img, cmap='gray')
4. Split the Dataset
python
Copy
Edit
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
🧠 Model Architecture
python
Copy
Edit
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
🔧 Compile the Model
python
Copy
Edit
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
📈 Train the Model
python
Copy
Edit
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))
🧪 Make Predictions
python
Copy
Edit
y_pred = model.predict(X_val).argmax(axis=1)
🔍 Test with an Image
python
Copy
Edit
img = X_val.iloc[2].values.reshape((28, 28))
plt.imshow(img, cmap='gray')
print('Predicted label:', model.predict(X_val).argmax(axis=1)[2])
📊 Evaluation Metrics
python
Copy
Edit
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))
🏁 Future Improvements
Switch to CNNs for better image analysis.

Use data augmentation.

Add model checkpointing and early stopping.

📌 Project Summary
Feature	Description
Framework	TensorFlow Keras
Dataset	MNIST (via CSV files)
Model Type	Feed-forward Neural Network (MLP)
Activation Functions	ReLU, Softmax
Optimizer	Adam
Accuracy Achieved	~96–98% (approx.)
