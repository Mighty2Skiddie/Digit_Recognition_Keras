# Digit_Recognition_Keras
# 🧠 Digit Recognition using Deep Neural Networks (MNIST Dataset)

This project aims to classify handwritten digits (0–9) from the **MNIST dataset** using a **Deep Neural Network (DNN)** built with **TensorFlow** and **Keras**. The goal is to demonstrate the power of dense layers in accurately identifying digits without using Convolutional Neural Networks (CNNs).

---

## 📂 Dataset Description

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a benchmark dataset in the machine learning community. It contains:

- **60,000** training images and **10,000** test images  
- Each image is a **28x28 grayscale** pixel image representing digits from **0 to 9**
- Pixel values range from 0 (black) to 255 (white), which are **flattened to 784-dimensional vectors**

---

## 🚀 Project Workflow

1. **Importing Libraries:**  
   - `TensorFlow`, `Keras` for model building  
   - `NumPy`, `Pandas` for data handling  
   - `Matplotlib`, `Seaborn` for visualization  
   - `Scikit-learn` for evaluation and preprocessing

2. **Data Loading and Preprocessing:**  
   - Read CSV files using `pandas`
   - Separate features and labels
   - Flatten images from 28x28 to 784 features
   - Split into training and test sets using `train_test_split`

3. **Model Design:**
   Built using **Keras Sequential API** with the following layers:

   | Layer Type       | Neurons | Activation | Description |
   |------------------|---------|------------|-------------|
   | Dense (Hidden 1) | 128     | ReLU       | Learns initial features from pixel intensities |
   | Dense (Hidden 2) | 128     | ReLU       | Captures complex patterns and relationships |
   | Dense (Hidden 3) | 128     | ReLU       | Deepens feature extraction |
   | Dense (Output)   | 10      | Softmax    | Outputs probabilities for each of the 10 classes |

4. **Model Compilation:**  
   - **Optimizer:** `Adam` — adaptive learning rate algorithm  
   - **Loss Function:** `Sparse Categorical Crossentropy` — suitable for integer-labeled multiclass classification  
   - **Metrics:** Accuracy  

5. **Model Training:**  
   - Trained over multiple epochs (e.g., 10) with batch size (e.g., 32 or 64)  
   - Model summary printed before training  

6. **Evaluation:**  
   - Visualized predictions and evaluated using:  
     - **Confusion Matrix**
     - **Classification Report**
     - **Accuracy Score**

---

## 🧠 Deep Neural Network Explanation

### 1. **Dense Layers (Fully Connected Layers):**
These layers are the core of DNNs where each neuron in one layer connects to every neuron in the next. This allows complex relationships to be learned between inputs and outputs.

### 2. **ReLU Activation Function:**
ReLU stands for Rectified Linear Unit and is defined as:


- **Purpose:** Introduces non-linearity into the network
- **Benefits:** Prevents vanishing gradient, faster training, simple computation
- **Used In:** All hidden layers

### 3. **Softmax Activation (Output Layer):**
Softmax converts raw logits to probabilities across 10 classes:


- **Purpose:** Outputs a probability distribution over 10 digit classes (0–9)
- **Used In:** Final output layer

### 4. **Adam Optimizer:**
Combines **AdaGrad** and **RMSProp** optimizers. It adjusts learning rates during training:

- **Advantages:** Adaptive, efficient on large datasets, requires minimal tuning

### 5. **Loss Function - Sparse Categorical Crossentropy:**
Used for integer-encoded targets (labels like 0, 1, …, 9). It measures the difference between actual labels and predicted probabilities.

---

## 📊 **Results**

| Metric             | Value     |
|--------------------|-----------|
| Training Accuracy  | ~98%      |
| Validation Accuracy| ~96.4%    |
| Loss (Crossentropy)| Very Low  |
| Test Size Used     | 10% split from training set |

- High accuracy proves dense layers alone (without CNNs) can classify handwritten digits effectively when data is well-preprocessed.

---

## 📌 **Installation & Setup**

### 1. Clone this repository:
```bash
git clone https://github.com/yourusername/Digit-Recognition-DNN.git
cd Digit-Recognition-DN
```
### 2. Install dependencies:
```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```
### 3. Run the notebook:
```
jupyter notebook Digit_Recognition_MNIST.ipynb
```
## **Technologies Used**
- Programming Language: Python

- Deep Learning Framework: TensorFlow 2.x, Keras

- Data Manipulation: NumPy, Pandas

- Visualization: Matplotlib, Seaborn

- Model Evaluation: Scikit-learn

## **Future Improvements**

- Implement Convolutional Neural Networks (CNNs) for better accuracy

- Apply Dropout layers for regularization and preventing overfitting

- Perform Hyperparameter tuning using GridSearchCV or Keras Tuner

- Use TensorBoard for advanced visualization of training metrics

- Convert model to TensorFlow Lite for deployment on mobile devices


