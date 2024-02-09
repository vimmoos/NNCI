# try:
#     import cupy as np
# except ImportError:
#     import numpy as np
import numpy as np
import numpy
from typing import Tuple
from dataclasses import dataclass, field
from itertools import product
import csv
import os
from tqdm import tqdm
import inspect as i


@np.vectorize
def sign(x, noise: float):
    return (1 if x > 0 else -1) * (2 * (numpy.random.rand() > noise) - 1)


def gen_data(
    n_points: int,
    n_features: int,
    noise: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    teacher = np.random.normal(0, 1, n_features)
    teacher /= np.linalg.norm(teacher) * n_features

    features = np.random.normal(loc=0, scale=1, size=(n_points, n_features))
    labels = sign(np.matmul(features, teacher), noise)
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
            return False, metric

        if metric >= self.min_val:
            self.cnt += 1

        return self.cnt >= self.patience, metric


def minover(data, teacher, max_iter=10, patience=4):
    features = data[:, :-1]
    labels = data[:, -1, None]
    weights = np.zeros((1, features.shape[1]))
    errors = []
    stop_criterion = EarlyStopping(patience)
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
):
    features = data[:, :-1]
    labels = data[:, -1, None]
    embedding_strenghts = np.zeros((features.shape[0],))
    weights = np.zeros((1, features.shape[1]))
    C = corr_mat(features, labels)
    hebbians = features * labels
    errors = []
    stop_criterion = EarlyStopping(patience)
    for _ in range(max_iter):
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


def runner(fun, list_args, file_name: str):
    first = True
    res = []
    root, _ = os.path.splitext(file_name)
    file_path = root + ".csv"
    args = list(product(*list_args))
    for x in tqdm(args):
        res.append(fun(*x))
        if first:
            while os.path.exists(file_path):
                print(f"Warning: {file_path} already exists and changing file name.")
                root, ext = os.path.splitext(file_path)
                file_path = root + "0" + ext
                print(f"new file_name {file_path}")
            with open(file_path, "w") as fil:
                writer = csv.DictWriter(fil, fieldnames=res[0].keys())
                writer.writeheader()
            first = False
        if len(res) > 50:
            with open(file_path, "a") as fil:
                writer = csv.DictWriter(fil, fieldnames=res[0].keys())
                writer.writerows(res)
            res = []
    if res:
        with open(file_path, "a") as fil:
            writer = csv.DictWriter(fil, fieldnames=res[0].keys())
            writer.writerows(res)


def run_base_experiment(alpha, N, n_dataset, max_iter, noise):
    args = locals()
    P = int(alpha * N)
    errs, lengths = [], []
    for _ in range(n_dataset):
        data, teacher = gen_data(P, N, noise=noise)
        w, err = minover(data, teacher, max_iter=max_iter, patience=100)
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": numpy.mean(errs),
        "error_std": numpy.std(errs),
        "length_mean": numpy.mean(lengths),
        "length_std": numpy.std(lengths),
        **args,
    }


def run_adatron_exp(alpha, N, n_dataset, max_iter, noise, lr, max_strenght):
    args = locals()
    P = int(alpha * N)
    errs, lengths = [], []
    for _ in range(n_dataset):
        data, teacher = gen_data(P, N, noise=noise)
        w, err = adatron(
            data,
            teacher,
            lr=lr,
            max_iter=max_iter,
            patience=100,
            max_strenght=max_strenght,
        )
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": numpy.mean(errs),
        "error_std": numpy.std(errs),
        "length_mean": numpy.mean(lengths),
        "length_std": numpy.std(lengths),
        **args,
    }


def run_base_experiments(file_name: str):
    base_alpha = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    more_alpha = np.linspace(0.1, 10, 15)
    alpha = np.unique(np.concatenate((base_alpha, more_alpha)))
    N = [20, 40, 100, 200]
    n_dataset = [100]
    max_iters = [500]
    noise = [0]
    runner(run_base_experiment, (alpha, N, n_dataset, max_iters, noise), file_name)


def run_adatron_experiments(file_name: str):
    base_alpha = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    more_alpha = np.linspace(0.1, 10, 15)
    alpha = np.unique(np.concatenate((base_alpha, more_alpha)))
    lr = np.linspace(0.05, 2, 10)
    N = [20, 40, 100, 200]
    n_dataset = [100]
    max_iters = [500]
    noise = [0]
    max_strenght = [np.inf]
    runner(
        run_adatron_exp,
        (alpha, N, n_dataset, max_iters, noise, lr, max_strenght),
        file_name,
    )


def run_base_noise_experiment(file_name: str):
    alpha = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    N = [20, 40, 100, 200]
    n_dataset = [100]
    max_iters = [500]
    noise = np.linspace(0.01, 0.5, 10)
    runner(run_base_experiment, (alpha, N, n_dataset, max_iters, noise), file_name)


def run_adatron_noise_experiments(file_name: str):
    alpha = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    lr = np.linspace(0.05, 2, 5)
    N = [20, 40, 100]
    n_dataset = [100]
    max_iters = [500]
    noise = np.linspace(0.01, 0.5, 5)
    max_strenght = np.linspace(0, 200, 10)
    runner(
        run_adatron_exp,
        (alpha, N, n_dataset, max_iters, noise, lr, max_strenght),
        file_name,
    )


if __name__ == "__main__":
    # run_base_experiments("first_base_run")
    run_base_noise_experiment("base_noise_run")
    # run_adatron_noise_experiments("adatron_noise_run")
    # run_adatron_experiments("adatron_run")
