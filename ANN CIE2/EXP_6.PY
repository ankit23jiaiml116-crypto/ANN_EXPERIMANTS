import numpy as np

# Hebbian learning function
def hebbian_learning(X, Y):
    # Add bias input of 1 to each sample
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Initialize weights with zeros
    weights = np.zeros(X.shape[1])
    
    # Apply Hebb’s rule
    for i in range(X.shape[0]):
        weights += X[i] * Y[i]
    
    return weights

# Predict function
def predict(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias
    outputs = np.dot(X, weights)
    return np.where(outputs >= 0, 1, 0)

# Input patterns for logical functions
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# -------- AND Function --------
Y_and = np.array([0, 0, 0, 1])
weights_and = hebbian_learning(X, Y_and)
print("Hebbian Weights for AND:", weights_and)
print("Predictions for AND:", predict(X, weights_and))

# -------- OR Function --------
Y_or = np.array([0, 1, 1, 1])
weights_or = hebbian_learning(X, Y_or)
print("\nHebbian Weights for OR:", weights_or)
print("Predictions for OR:", predict(X, weights_or))