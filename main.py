import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. Data Requirements (Loading & Preprocessing)
# ==========================================
print("Loading and Preprocessing MNIST Data...")
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalization: Scaling pixel values to be between 0 and 1
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Partitioning: Splitting training set into Training (80%) and Validation (20%)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# ==========================================
# 2. Model, Training & Evaluation Setup
# ==========================================
def run_experiment(exp_name, num_neurons, activation_fn, lr, dropout_rate):
    print(f"\n{'-'*50}")
    print(f"Running {exp_name}")
    print(f"Hyperparameters -> Neurons: {num_neurons}, Activation: '{activation_fn}', LR: {lr}")
    print(f"{'-'*50}")
    
    # MLP Architecture with proper Regularization ordering
    model = Sequential([
        # Input Layer
        Flatten(input_shape=(28, 28)), 
        
        # Hidden Layer 1
        Dense(num_neurons),
        BatchNormalization(),
        Activation(activation_fn),
        Dropout(dropout_rate),
        
        # Hidden Layer 2
        Dense(num_neurons // 2),
        BatchNormalization(),
        Activation(activation_fn),
        Dropout(dropout_rate - 0.1 if dropout_rate > 0.1 else dropout_rate), # تقليل نسبة الإسقاط في الطبقة الأعمق
        
        # Output Layer (10 classes for digits 0-9)
        Dense(10, activation='softmax')
    ])
    
    # Selecting Loss and Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Training the model
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        verbose=1)
    
    # Evaluating on Testing Dataset
    print(f"\nEvaluating {exp_name} on unseen Test Data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")
    
    return history, test_acc, test_loss

# ==========================================
# 3. Experimentation Requirements
# ==========================================
# Experiment 1: Baseline settings
hist1, acc1, loss1 = run_experiment("Experiment 1", 
                                    num_neurons=256, 
                                    activation_fn='relu', 
                                    lr=0.001, 
                                    dropout_rate=0.3)

# Experiment 2: Modifying Activation, Neurons, and Learning Rate
hist2, acc2, loss2 = run_experiment("Experiment 2", 
                                    num_neurons=128, 
                                    activation_fn='tanh', 
                                    lr=0.005, 
                                    dropout_rate=0.2)

# ==========================================
# 4. Visualization Requirements
# ==========================================
def plot_experiment_curves(hist1, hist2):
    plt.figure(figsize=(16, 10))
    
    # Experiment 1 Loss
    plt.subplot(2, 2, 1)
    plt.plot(hist1.history['loss'], label='Train Loss', color='blue')
    plt.plot(hist1.history['val_loss'], label='Validation Loss', color='cyan', linestyle='dashed')
    plt.title('Experiment 1: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Experiment 2 Loss
    plt.subplot(2, 2, 2)
    plt.plot(hist2.history['loss'], label='Train Loss', color='red')
    plt.plot(hist2.history['val_loss'], label='Validation Loss', color='orange', linestyle='dashed')
    plt.title('Experiment 2: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Experiment 1 Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(hist1.history['accuracy'], label='Train Acc', color='blue')
    plt.plot(hist1.history['val_accuracy'], label='Validation Acc', color='cyan', linestyle='dashed')
    plt.title('Experiment 1: Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Experiment 2 Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(hist2.history['accuracy'], label='Train Acc', color='red')
    plt.plot(hist2.history['val_accuracy'], label='Validation Acc', color='orange', linestyle='dashed')
    plt.title('Experiment 2: Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\nGenerating Visualizations...")
plot_experiment_curves(hist1, hist2)

# ==========================================
# 5. Final Comparison
# ==========================================
results_df = pd.DataFrame({
    "Experiment": [
        "Exp 1 (256 Neurons, ReLU, LR=0.001)", 
        "Exp 2 (128 Neurons, Tanh, LR=0.005)"
    ],
    "Test Accuracy": [f"{acc1:.4f}", f"{acc2:.4f}"],
    "Test Loss": [f"{loss1:.4f}", f"{loss2:.4f}"]
})

print("\n" + "="*50)
print("FINAL RESULTS COMPARISON")
print("="*50)
print(results_df.to_string(index=False))