import csv
import math
import random
import matplotlib.pyplot as plt

filepath = "train.csv"

def get_data(filepath):
    X = []
    y = []
    with open(filepath) as f:
        for i, row in enumerate(csv.reader(f)):
            goal_index = i // 7
            row = row[0]
            if len(row) == 6:
                if len(X) <= goal_index:
                    X.append([])
                X[goal_index].extend(map(int, row))
            elif len(row) == 3:
                if len(y) <= goal_index:
                    y.append([])
                y[goal_index].extend(map(int, row))
    return X, y

import math
import random

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, activation="sigmoid", loss="mse"):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss
        
        # Ініціалізація ваг і зсувів
        self.weights = [
            [
                [random.uniform(-1, 1) for _ in range(layer_sizes[l])]
                for _ in range(layer_sizes[l + 1])
            ] for l in range(len(layer_sizes) - 1)
        ]
        self.biases = [
            [random.uniform(-1, 1) for _ in range(layer_sizes[l + 1])]
            for l in range(len(layer_sizes) - 1)
        ]

        # Збереження виходів
        self.outputs = [[0.0 for _ in range(size)] for size in layer_sizes]

        # Мапа для виводу результатів
        self.results = {
            (0, 0 ,0): "коло", (1, 0, 0): "квадрат", (0, 1, 0): "ромб",
            (0, 0, 1): "еліпс", (1, 1, 1): "трикутник"
        }

    def activate(self, x):
        if self.activation == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif self.activation == "relu":
            return max(0, x)
        elif self.activation == "tanh":
            return math.tanh(x)
        return x

    def activate_derivative(self, activated_x):
        if self.activation == "sigmoid":
            return activated_x * (1 - activated_x)
        elif self.activation == "relu":
            return 1 if activated_x > 0 else 0
        elif self.activation == "tanh":
            return 1 - activated_x ** 2
        return 1

    def forward(self, x):
        self.outputs[0] = x[:]
        for l in range(1, len(self.layer_sizes)):
            for j in range(self.layer_sizes[l]):
                weighted_sum = sum(
                    self.weights[l - 1][j][k] * self.outputs[l - 1][k]
                    for k in range(self.layer_sizes[l - 1])
                ) + self.biases[l - 1][j]
                self.outputs[l][j] = self.activate(weighted_sum)
        return self.outputs

    def mse_loss(self, y_true, y_pred):
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def mae_loss(self, y_true, y_pred):
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def backward(self, y_true):
        num_layers = len(self.layer_sizes)
        local_gradients = [[0.0 for _ in range(size)] for size in self.layer_sizes[1:]]

        # Градієнти вихідного шару
        for i in range(self.layer_sizes[-1]):
            output = self.outputs[-1][i]
            error = y_true[i] - output
            local_gradients[-1][i] = error * self.activate_derivative(output)

        # Зворотне поширення градієнтів
        for l in reversed(range(len(self.layer_sizes) - 1)):
            for i in range(self.layer_sizes[l + 1]):
                for j in range(self.layer_sizes[l]):
                    delta = self.learning_rate * local_gradients[l][i] * self.outputs[l][j]
                    self.weights[l][i][j] += delta
                self.biases[l][i] += self.learning_rate * local_gradients[l][i]

            if l > 0:
                for j in range(self.layer_sizes[l]):
                    gradient_sum = sum(
                        self.weights[l][i][j] * local_gradients[l][i]
                        for i in range(self.layer_sizes[l + 1])
                    )
                    local_gradients[l - 1][j] = gradient_sum * self.activate_derivative(self.outputs[l][j])

        # Повернення втрати
        if self.loss == 'mse':
            return self.mse_loss(y_true, self.outputs[-1])
        elif self.loss == 'mae':
            return self.mae_loss(y_true, self.outputs[-1])
        return 0

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                self.forward(x)
                loss = self.backward(y)
                total_loss += loss
            if epoch % 100 == 0:
                print(f"Епоха {epoch}, середня втрата: {total_loss / len(X_train)}")

    def evaluate(self, X_test, y_test):
        correct = 0
        for x, y in zip(X_test, y_test):
            raw_pred = self.forward(x)[-1]  # беремо тільки вихідний шар
            pred = [1 if p >= 0.5 else 0 for p in raw_pred]
            pred_fig = self.results.get(tuple(pred), '?')
            true_fig = self.results.get(tuple(y), '?')
            if pred_fig == true_fig:
                correct += 1
        accuracy = correct / len(X_test)
        print(f"Точність: {accuracy * 100}%")

    def visualize_predictions(self, X_test, y_test, num_samples=5):
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            x, y = X_test[i], y_test[i]
            raw_pred = self.forward(x)[-1]  # беремо лише вихід нейромережі
            pred = [1 if p >= 0.5 else 0 for p in raw_pred]  # округлення до 0 або 1
            # Перетворення списку x у 2D-масив 6x6
            image = [x[j*6:(j+1)*6] for j in range(6)]

            axes[i].imshow(image, cmap="Blues")
            axes[i].set_title(f"Прогноз: {self.results.get(tuple(pred), '?')}\nПравильна: {self.results.get(tuple(y), '?')}")
            axes[i].axis("off")
        plt.show()
    
    
X_train, y_train = get_data("train.csv")
X_test, y_test = get_data("test.csv")

nn = NeuralNetwork(layer_sizes=[36, 8, 3], learning_rate=0.01, activation="sigmoid", loss="mse")
nn.train(X_train, y_train, epochs=1000)
nn.evaluate(X_test, y_test)
nn.visualize_predictions(X_test, y_test, num_samples=5)
