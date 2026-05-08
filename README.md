# Neural Network Project - MNIST Classification

## Problem Description

This project aims to build a Multilayer Perceptron (MLP) model to recognize handwritten digits using the MNIST dataset. The model takes grayscale images as input and predicts the corresponding digit (0–9).

---

## Dataset

The MNIST dataset is used in this project. It contains:

* 60,000 training images
* 10,000 testing images
* Each image has a size of 28×28 pixels (grayscale)

The dataset is available directly through TensorFlow/Keras.

---

## Data Preprocessing

Before training the model, the following steps were applied:

* Pixel values were normalized to the range [0, 1]
* Images were flattened from 28×28 into a 1D vector of size 784
* Data was split into 80% training and 20% validation sets

---

## Model Architecture

### Experiment 1 (Baseline)

* Input layer: Flatten (784 units)
* Hidden layer 1: 256 units (ReLU) + Batch Normalization + Dropout (0.3)
* Hidden layer 2: 128 units (ReLU) + Batch Normalization + Dropout (0.2)
* Output layer: 10 units (Softmax)

### Experiment 2 (Modified)

* Input layer: Flatten (784 units)
* Hidden layer 1: 128 units (Tanh) + Batch Normalization + Dropout (0.2)
* Hidden layer 2: 64 units (Tanh) + Batch Normalization + Dropout (0.1)
* Output layer: 10 units (Softmax)

---

## Training Setup

The models were trained using the following configuration:

* Optimizer: Adam (LR=0.001 for Exp 1, LR=0.005 for Exp 2)
* Loss function: Sparse Categorical Crossentropy
* Number of epochs: 10
* Batch size: 64
* Validation split: 20%

---

## Results

| Experiment | Architecture & Settings | Accuracy | Loss   |
| ---------- | ----------------------- | -------- | ------ |
| Exp 1      | 256 → 128 (ReLU)        | 97.94%   | 0.0695 |
| Exp 2      | 128 → 64 (Tanh)         | 97.42%   | 0.0933 |

---

## Experiments

Two experiments were performed by changing:

* The number of neurons (256 vs 128)
* The activation function (ReLU vs Tanh)
* The learning rate (0.001 vs 0.005)

The first model (Experiment 1) achieved better results in both accuracy and loss.

---

## Visualization

The following plots were generated during training and saved in the repository:

* Training and validation loss
* Training and validation accuracy

These plots indicate that both models learned effectively without overfitting, thanks to the usage of Dropout and Batch Normalization.

---

## Conclusion

An MLP model was implemented successfully for digit classification. Both experiments achieved high accuracy (above 97%).

Experiment 1 showed the best performance. The combination of the **ReLU** activation function, a higher number of neurons, and a stable learning rate proved to be more effective and accurate for this dataset.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py