import numpy as np
from typing import Tuple
from dataclasses import dataclass, field


@np.vectorize
def sign(x, noise: float):
    return (1 if x > 0 else -1) * (2 * (np.random.rand() > noise) - 1)


def gen_data(
    n_points: int,
    n_features: int,
    noise: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    teacher = np.random.normal(0, 1, n_features)
    teacher /= np.linalg.norm(teacher) * n_features

    features = np.random.normal(loc=0, scale=1, size=(n_points, n_features))
    labels = sign(np.matmul(features, teacher), noise=noise)
    return np.c_[features, labels], np.expand_dims(teacher, 0)


def stability(weights, features, labels, min_idx: bool = False):
    if min_idx:
        return np.argmin(np.dot(features * labels, weights.T))
    return np.dot(features * labels, weights.T) / np.linalg.norm(weights)


def gen_error(weights, teacher):
    return (1 / np.pi) * np.arccos(
        (
            np.dot(weights, teacher.T)
            / (np.linalg.norm(weights) * np.linalg.norm(teacher))
        ).squeeze()
    )


@dataclass
class EarlyStopping:
    patience: int = 4
    delta: float = 0.0
    cnt: int = field(init=False, default_factory=lambda: 0)
    min_val: float = field(init=False, default_factory=lambda: float("inf"))

    @property
    def removable(self):
        return -self.cnt if self.cnt >= 1 else None

    def __call__(self, weights, teacher):
        metric = gen_error(weights, teacher)
        if metric < self.min_val:
            self.min_val = metric
            self.cnt = 0
            print("reset")
            return False, metric

        if metric >= (self.min_val + self.delta):
            print("increase")
            self.cnt += 1

        return self.cnt >= self.patience, metric


def minover(data, teacher, max_iter=10, patience=4, delta=0):
    features = data[:, :-1]
    labels = data[:, -1, None]
    weights = np.zeros((1, features.shape[1]))
    errors = []
    stop_criterion = EarlyStopping(patience, delta)
    for i in range(max_iter):
        min_idx = stability(weights, features, labels, min_idx=True)
        weights += (1 / features.shape[1]) * (features[min_idx] * labels[min_idx])
        stop, error = stop_criterion(weights, teacher)
        errors.append(error)
        if stop:
            break
    return weights, errors[: stop_criterion.removable]


def corr_mat(features, labels):
    return (
        (1 / features.shape[1])
        * np.outer(labels, labels)
        * np.dot(features, features.T)
    )


def adatron(
    data,
    teacher,
    lr=1,
    max_iter=10,
    max_strenght=np.inf,
    patience=4,
    delta=0,
):
    features = data[:, :-1]
    labels = data[:, -1, None]
    embedding_strenghts = np.zeros((features.shape[0],))
    weights = np.zeros((1, features.shape[1]))
    C = corr_mat(features, labels)
    hebbians = features * labels
    errors = []
    stop_criterion = EarlyStopping(patience, delta)
    for i in range(max_iter):
        for j in range(features.shape[0]):
            energy = np.dot(C, embedding_strenghts)[j]
            embedding_strenghts[j] = np.clip(
                embedding_strenghts[j] + lr * (1 - energy), 0, max_strenght
            )
        weights = np.dot(embedding_strenghts.T, hebbians)
        stop, error = stop_criterion(weights, teacher)
        errors.append(error)
        if stop:
            break
    return weights, errors[: stop_criterion.removable]


# def run_base_experiment(alpha,N,n_dataset,max_iter)
