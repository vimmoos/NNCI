import numpy as np
import numpy
from nnci.early_stop import EarlyStopping


def stability(weights, data, min_idx: bool = False):
    features = data[:, :-1]
    labels = data[:, -1, None]
    if min_idx:
        return np.argmin(np.dot(features * labels, weights.T))
    return np.min(np.dot(features * labels, weights.T) / np.linalg.norm(weights))


def gen_error(weights, teacher):
    return (1 / np.pi) * np.arccos(
        (
            np.dot(weights, teacher.T)
            / (np.linalg.norm(weights) * np.linalg.norm(teacher))
        ).squeeze()
    )


def corr_mat(features, labels):
    return (
        (1 / features.shape[1])
        * np.outer(labels, labels)
        * np.dot(features, features.T)
    )


def minover(
    data,
    teacher,
    max_iter=10,
    patience=4,
    metric=gen_error,
    inverted=False,
):
    features = data[:, :-1]
    labels = data[:, -1, None]
    weights = np.zeros((1, features.shape[1]))
    errors = []
    stop_criterion = EarlyStopping(
        metric=metric,
        patience=patience,
        inverted=inverted,
    )
    for _ in range(max_iter):
        min_idx = stability(weights, data, min_idx=True)
        weights += (1 / features.shape[1]) * (features[min_idx] * labels[min_idx])
        stop, error = stop_criterion(weights, teacher)
        errors.append(error)
        if stop:
            break
    return weights, errors[: stop_criterion.removable]


def adatron(
    data,
    teacher,
    lr=1,
    max_iter=10,
    max_strenght=np.inf,
    patience=4,
    metric=gen_error,
    inverted=False,
):
    features = data[:, :-1]
    labels = data[:, -1, None]
    embedding_strenghts = np.zeros((features.shape[0],))
    weights = np.zeros((1, features.shape[1]))
    C = corr_mat(features, labels)
    hebbians = features * labels
    errors = []
    stop_criterion = EarlyStopping(
        metric=metric,
        patience=patience,
        inverted=inverted,
    )
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
