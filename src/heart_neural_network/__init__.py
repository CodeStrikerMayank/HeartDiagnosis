import sys
import pathlib as pt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


"""Extracting data"""

class extraction_process_by_CS:

    def __init__(self, filepath="heart.csv"):
        try:
            self.address = pt.Path(filepath)
            if not self.address.exists():
                # Check relative to package root if running from a different working directory
                alt_path = pt.Path(__file__).resolve().parent.parent.parent / "heart.csv"
                if alt_path.exists():
                    self.address = alt_path

            if self.address.exists():
                print("File found:", self.address)
            else:
                print("File does not exist:", self.address)

        except Exception as e:
            print("Error finding file:", e)

    def data_extract(self):
        try:
            self.data = pd.read_csv(self.address)
            self.x = self.data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]].to_numpy()
            self.y = self.data["target"].to_numpy().reshape(-1, 1)
            
        except FileNotFoundError as err:
            print("CSV file not found:", err)
            sys.exit(1)
        except Exception as err:
            print("Something went wrong during extraction:", err)
            sys.exit(1)

    def spiliting_data_sets(self):
        try:
            x_train, x_temp, y_train, y_temp = train_test_split(self.x, self.y, test_size=0.30, random_state=42, shuffle=True)
            x_test, x_valid, y_test, y_valid = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42, shuffle=True)
            return x_train, y_train, x_test, y_test, x_valid, y_valid
        except ValueError as err:
            print("Failed extraction and data split:", err)
            sys.exit(1)


class Neural_activation_process(extraction_process_by_CS):

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def relu(z):
        return np.maximum(z * 0.01, z)

    @staticmethod
    def relu_der(z):
        return np.where(z > 0, 1.0, 0.01)

    @staticmethod
    def bce(pred, target):
        """Binary cross entropy: -[y*log(p) + (1-y)*log(1-p)]"""
        eps = 1e-15
        p = np.clip(pred, eps, 1.0 - eps)
        loss = -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
        return np.mean(loss)

    def __init__(self, filepath="heart.csv", auto_train=True):
        super().__init__(filepath=filepath)
        self.data_extract()
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_valid, self.y_valid = self.spiliting_data_sets()

        # Feature normalization (Standardization)
        self.mean = self.x_train.mean(axis=0)
        self.std = self.x_train.std(axis=0)
        self.std[self.std == 0] = 1.0

        self.x_train = (self.x_train - self.mean) / self.std
        self.x_valid = (self.x_valid - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        # He initialization for LeakyReLU layers: std = sqrt(2 / fan_in)
        np.random.seed(42)
        self.w1 = np.random.randn(13, 32) * np.sqrt(2.0 / 13)
        self.b1 = np.zeros((1, 32))

        self.w2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, 16))

        self.w3 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1, 8))

        self.w4 = np.random.randn(8, 4) * np.sqrt(2.0 / 8)
        self.b4 = np.zeros((1, 4))

        self.w5 = np.random.randn(4, 2) * np.sqrt(2.0 / 4)
        self.b5 = np.zeros((1, 2))

        self.w6 = np.random.randn(2, 1) * np.sqrt(2.0 / 2)
        self.b6 = np.zeros((1, 1))

        if auto_train:
            self.train(epochs=300, batch_size=16, lr=0.01)

    def forward(self, x):
        """Forward pass through the neural network"""
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2, self.w3) + self.b3
        a3 = self.relu(z3)

        z4 = np.dot(a3, self.w4) + self.b4
        a4 = self.relu(z4)

        z5 = np.dot(a4, self.w5) + self.b5
        a5 = self.relu(z5)

        z6 = np.dot(a5, self.w6) + self.b6
        a6 = self.sigmoid(z6)

        cache = (z1, a1, z2, a2, z3, a3, z4, a4, z5, a5, z6, a6)
        return a6, cache

    def predict(self, x):
        """Predict binary class labels (0 or 1)"""
        pred, _ = self.forward(x)
        return (pred > 0.5).astype(int)

    def evaluate(self, x, y):
        """Calculate loss and accuracy"""
        pred, _ = self.forward(x)
        loss = self.bce(pred, y)
        accuracy = np.mean((pred > 0.5).astype(int) == y)
        return loss, accuracy

    def train(self, epochs=300, batch_size=16, lr=0.01):
        """Train the neural network using mini-batch gradient descent"""
        m = len(self.x_train)

        print(f"\nStarting training: {epochs} epochs, batch size {batch_size}, learning rate {lr}...")

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(m)
            x_shuffle = self.x_train[indices]
            y_shuffle = self.y_train[indices]

            for i in range(0, m, batch_size):
                xb = x_shuffle[i:i + batch_size]
                yb = y_shuffle[i:i + batch_size]
                mb = len(xb)

                # Forward propagation
                z1 = np.dot(xb, self.w1) + self.b1
                a1 = self.relu(z1)

                z2 = np.dot(a1, self.w2) + self.b2
                a2 = self.relu(z2)

                z3 = np.dot(a2, self.w3) + self.b3
                a3 = self.relu(z3)

                z4 = np.dot(a3, self.w4) + self.b4
                a4 = self.relu(z4)

                z5 = np.dot(a4, self.w5) + self.b5
                a5 = self.relu(z5)

                z6 = np.dot(a5, self.w6) + self.b6
                a6 = self.sigmoid(z6)

                # Backward propagation
                dz6 = a6 - yb
                dw6 = (1.0 / mb) * np.dot(a5.T, dz6)
                db6 = (1.0 / mb) * np.sum(dz6, axis=0, keepdims=True)

                dz5 = np.dot(dz6, self.w6.T) * self.relu_der(z5)
                dw5 = (1.0 / mb) * np.dot(a4.T, dz5)
                db5 = (1.0 / mb) * np.sum(dz5, axis=0, keepdims=True)

                dz4 = np.dot(dz5, self.w5.T) * self.relu_der(z4)
                dw4 = (1.0 / mb) * np.dot(a3.T, dz4)
                db4 = (1.0 / mb) * np.sum(dz4, axis=0, keepdims=True)

                dz3 = np.dot(dz4, self.w4.T) * self.relu_der(z3)
                dw3 = (1.0 / mb) * np.dot(a2.T, dz3)
                db3 = (1.0 / mb) * np.sum(dz3, axis=0, keepdims=True)

                dz2 = np.dot(dz3, self.w3.T) * self.relu_der(z2)
                dw2 = (1.0 / mb) * np.dot(a1.T, dz2)
                db2 = (1.0 / mb) * np.sum(dz2, axis=0, keepdims=True)

                dz1 = np.dot(dz2, self.w2.T) * self.relu_der(z1)
                dw1 = (1.0 / mb) * np.dot(xb.T, dz1)
                db1 = (1.0 / mb) * np.sum(dz1, axis=0, keepdims=True)

                # Gradient descent parameter updates
                self.w1 -= lr * dw1
                self.w2 -= lr * dw2
                self.w3 -= lr * dw3
                self.w4 -= lr * dw4
                self.w5 -= lr * dw5
                self.w6 -= lr * dw6

                self.b1 -= lr * db1
                self.b2 -= lr * db2
                self.b3 -= lr * db3
                self.b4 -= lr * db4
                self.b5 -= lr * db5
                self.b6 -= lr * db6

            if epoch % 50 == 0 or epoch == epochs:
                val_loss, val_acc = self.evaluate(self.x_valid, self.y_valid)
                train_loss, train_acc = self.evaluate(self.x_train, self.y_train)
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        # Final evaluation on test set
        test_loss, test_acc = self.evaluate(self.x_test, self.y_test)
        print(f"\nTraining Complete!")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%\n")


def main():
    model = Neural_activation_process()
    return 0


if __name__ == "__main__":
    main()
