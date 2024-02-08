import numpy as np
from itertools import product
from tqdm import tqdm
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import csv
import os

def gen_data(n_points: int, n_features: int):
    """
    Generates a dataset with P input feature vectors (ξ), each of length N, with values drawn
    from a standard normal distribution (mean 0, standard deviation 1).
    The output labels S are randomly assigned as -1 or 1 with equal probability 1/2.

    Returns:
    array([[ξ_1^1, ξ_2^1,..., ξ_N^1, S(ξ^1)],
           [ξ_1^2, ξ_2^2,..., ξ_N^2, S(ξ^2)],
           ...
           [ξ_1^P, ξ_2^P,..., ξ_N^P, S(ξ^P)]]
    """
    features = np.random.normal(loc=0, scale=1, size=(n_points, n_features))
    labels = np.array([x or -1 for x in np.random.randint(0, 2, features.shape[0])])
    return np.c_[features, labels]


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

    def compute_energy(self):
        return np.dot(self.hebbians, self.weights.T)

    def update(self):
        theta_fun = (self.compute_energy() <= self.c).astype(int)
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

    def stats(self):
        pass


def run_experiment(args):
    alpha, N, n_dataset, max_iter, c = args
    args = locals()
    del args["args"]
    P = int(alpha * N)
    solutions = 0
    errs = []
    for _ in range(n_dataset):
        data = gen_data(P, N)
        p = Perceptron(data, epochs=max_iter, c=c)
        error = p.train()
        solutions += int(error == 0)
        errs.append(error)
    return {
        "solutions": solutions,
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        **args,
    }


def run_experiments():
    alpha = np.linspace(0.5, 3, 11)
    # N = [20, 40, 100, 200, 500, 1000]
    # N = [20, 40, 100, 200]
    N = [20, 40]
    n_datasets = [50, 100]
    # n_datasets = [50, 100 , 200]
    max_iters = [100]
    # max_iters = [100,1000]
    c = np.linspace(0.1, 3, 15)
    # c = [0.1, 0.2, 0.5, 1]
    args = list(product(alpha, N, n_datasets, max_iters, c))
    first = True
    res = []
    for x in tqdm(args):
        res.append(run_experiment(x))
        if len(res) > 50:
            if first:
                if os.path.exists("res.csv"):
                    print("Warning: res.csv already exists and will be overwritten.")
                with open("res.csv", "w") as fil:
                    writer = csv.DictWriter(fil, fieldnames=res[0].keys())
                    writer.writeheader()
                first = False
            with open("res.csv", "a") as fil:
                writer = csv.DictWriter(fil, fieldnames=res[0].keys())
                writer.writerows(res)
            res = []
    if res:
        with open("res.csv", "a") as fil:
            writer = csv.DictWriter(fil, fieldnames=res[0].keys())
            writer.writerows(res)


if __name__ == "__main__":
    run_experiments()
