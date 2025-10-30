import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ----- LVQ Class -----
class LVQ:
    def init(self, n_codebooks, learning_rate, data_dim, n_epochs):
        self.n_codebooks = n_codebooks
        self.learning_rate = learning_rate
        self.data_dim = data_dim
        self.n_epochs = n_epochs
        self.codebooks = None
        self.codebook_labels = None

    def train(self, X_train, y_train):
        # Initialize codebooks from random training samples
        idx = np.random.choice(len(X_train), self.n_codebooks, replace=False)
        self.codebooks = X_train[idx].copy()
        self.codebook_labels = y_train[idx].copy()

        for epoch in range(self.n_epochs):
            for i in range(X_train.shape[0]):
                data_point = X_train[i]
                data_label = y_train[i]

                distances = np.linalg.norm(self.codebooks - data_point, axis=1)
                bmu_index = np.argmin(distances)
                bmu_label = self.codebook_labels[bmu_index]

                # LVQ update rule
                if bmu_label == data_label:
                    self.codebooks[bmu_index] += self.learning_rate * (data_point - self.codebooks[bmu_index])
                else:
                    self.codebooks[bmu_index] -= self.learning_rate * (data_point - self.codebooks[bmu_index])

            # Optionally decay learning rate
            self.learning_rate *= 0.99

    def predict(self, X_test):
        predictions = []
        for data_point in X_test:
            distances = np.linalg.norm(self.codebooks - data_point, axis=1)
            bmu_index = np.argmin(distances)
            predictions.append(self.codebook_labels[bmu_index])
        return np.array(predictions)


# ----- Load Dataset -----
iris = load_iris()
X, y = iris.data, iris.target

# Normalize data for better LVQ performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ----- Train LVQ -----
lvq = LVQ(n_codebooks=6, learning_rate=0.3, data_dim=X_train.shape[1], n_epochs=20)
lvq.train(X_train, y_train)

# ----- Predict -----
y_pred = lvq.predict(X_test)

# ----- Output -----
print("Predicted labels:", y_pred)
print("True labels     :", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))