import numpy as np
from typing import Tuple
import csv


@np.vectorize
def sign(x, noise: float):
    return (1 if x > 0 else -1) * (2 * (np.random.rand() > noise) - 1)


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


def gen_data_with_teacher(
    n_points: int,
    n_features: int,
    noise: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    teacher = np.random.normal(0, 1, n_features)
    teacher /= np.linalg.norm(teacher) * n_features

    features = np.random.normal(loc=0, scale=1, size=(n_points, n_features))
    labels = sign(np.matmul(features, teacher), noise)
    return np.c_[features, labels], np.expand_dims(teacher, 0)


def iris_data(lin_sep: bool):
    file_name = "class12centered" if lin_sep else "class23centered"
    with open(f"data/{file_name}.csv", "r") as f:
        reader = csv.reader(f)
        data = np.array(list(reader)).astype(np.float32)
    return np.c_[data, np.concatenate((np.ones(50), -np.ones(50)))]
