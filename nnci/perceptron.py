import numpy as np

import numpy
from dataclasses import dataclass, field
from matplotlib import pyplot as plt


@dataclass
class Perceptron:
    """
    This class represents a Perceptron, a simple binary classification algorithm based on a linear predictor function.

    Attributes:
    - data: The input data for the Perceptron, including features and labels.
    - epochs: The number of epochs for which the Perceptron will be trained.
    - c: The threshold value for the activation function.
    - weights: The weights of the Perceptron, initialized to zero.
    - features: The feature vectors from the input data.
    - labels: The labels from the input data.
    - energy: The energy of the Perceptron, computed as the dot product of the Hebbian vectors and the weights.
    - hebbians: The Hebbian vectors, computed as the product of the feature vectors and the labels.
    """

    data: np.ndarray
    epochs: int = 10
    c: float = 0
    weights: np.ndarray = field(init=False)
    features: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    energy: np.ndarray = field(init=False)
    hebbians: np.ndarray = field(init=False)

    def __post_init__(self):
        self.shuffle_data()
        self.weights = np.zeros((1, self.features.shape[1]))
        self.hebbians = self.features * self.labels
        self.energy = self.compute_energy()
        self.embedding_strenght = np.zeros_like(self.theta_fun())

    def compute_energy(self):
        return np.dot(self.hebbians, self.weights.T)

    def theta_fun(self):
        return (self.compute_energy() <= self.c).astype(int)

    def update(self):
        theta_fun = self.theta_fun()
        self.embedding_strenght += theta_fun
        self.weights += ((1 / self.features.shape[1]) * theta_fun * self.hebbians).sum(
            axis=0
        )

    def shuffle_data(self):
        np.random.shuffle(self.data)
        self.labels = self.data[:, -1, None]
        self.features = self.data[:, :-1]

    def train(self):
        for _ in range(self.epochs):
            if (self.compute_energy() > 0).all():
                break
            self.shuffle_data()
            self.update()
            self.stats()
        energy = self.compute_energy()
        return len(energy[energy < 0]) / len(self.features)

    def plot_solution(self):
        if self.features.shape[1] != 2:
            return
        fig, ax = plt.subplots()
        ax.scatter(self.features[:, 0], self.features[:, 1], c=self.labels)
        slope = -self.weights[:, 1] / self.weights[:, 0]
        ax.axline((0, 0), slope=slope[0])
        fig.show()
