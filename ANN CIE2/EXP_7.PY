import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR truth table)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected output (XOR)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Fix seed for reproducibility
np.random.seed(42)

# Network architecture
input_neurons = X.shape[1]   # 2 features
hidden_neurons = 4           # hidden layer
output_neurons = 1           # single output

# Initialize weights and biases
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Learning rate
lr = 0.5

# Training loop
for epoch in range(10000):
    # -------- Forward Propagation --------
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # -------- Backpropagation --------
    error = y - final_output                     # output error
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)            # hidden layer error
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # -------- Weight & Bias Updates --------
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Print error every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error)):.4f}")

# -------- Final Predictions --------
print("\nPredictions after training:")
print(final_output.round(3))   # rounded for clarity