import numpy as np
from typing import Tuple


@np.vectorize
def sign(x):
    return 1 if x > 0 else -1


def gen_data(
    n_points: int,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    teacher = np.random.normal(0, 1, n_features)
    teacher /= np.linalg.norm(teacher) * n_features

    features = np.random.normal(loc=0, scale=1, size=(n_points, n_features))
    labels = sign(np.matmul(features, teacher))
    return np.c_[features, labels], np.expand_dims(teacher, 0)


def stability(weights, features, labels, min_idx: bool = False):
    stabs = np.dot(features * labels, weights.T) / np.linalg.norm(weights)
    return np.argmin(stabs) if min_idx else stabs


def gen_error(weights, teacher):
    return (1 / np.pi) * np.arccos(
        (
            np.dot(weights, teacher.T)
            / (np.linalg.norm(weights) * np.linalg.norm(teacher))
        ).squeeze()
    )


def minover(data, teacher, max_iter=10):
    features = data[:, :-1]
    labels = data[:, -1, None]
    weights = np.zeros((1, features.shape[1]))
    gen_errors = []
    for i in range(max_iter):
        min_idx = stability(weights, features, labels, min_idx=True)
        weights += (1 / features.shape[1]) * (features[min_idx] * labels[min_idx])
        gen_errors.append(gen_error(weights, teacher))

    return weights, gen_errors
