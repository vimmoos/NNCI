import nnci.perceptron as per
import nnci.stability as stab
import nnci.data as d
import numpy as np


def run_perceptron(alpha, N, n_dataset, max_iter, c):
    args = locals()
    P = int(alpha * N)
    solutions = 0
    errs = []
    for _ in range(n_dataset):
        data = d.gen_data(P, N)
        p = per.Perceptron(data, epochs=max_iter, c=c)
        error = p.train()
        solutions += int(error == 0)
        errs.append(error)
    return {
        "solutions": solutions,
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        **args,
    }


def run_minover(alpha, N, n_dataset, max_iter, noise, patience=100):
    args = locals()
    P = int(alpha * N)
    errs, lengths = [], []
    for _ in range(n_dataset):
        data, teacher = d.gen_data_with_teacher(P, N, noise=noise)
        w, err = stab.minover(
            data,
            teacher,
            max_iter=max_iter,
            patience=patience,
        )
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        **args,
    }


def run_adatron(alpha, N, n_dataset, max_iter, noise, lr, max_strenght, patience=100):
    args = locals()
    P = int(alpha * N)
    errs, lengths = [], []
    for _ in range(n_dataset):
        data, teacher = d.gen_data_with_teacher(P, N, noise=noise)
        w, err = stab.adatron(
            data,
            teacher,
            lr=lr,
            max_iter=max_iter,
            patience=patience,
            max_strenght=max_strenght,
        )
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        **args,
    }


def run_iris_adatron(n_dataset, max_iter, lr, max_strenght, patience=100):
    args = locals()
    errs, lengths = [], []
    for _ in range(n_dataset):
        data = d.iris_data(False)
        w, err = stab.adatron(
            data,
            data,
            lr=lr,
            max_iter=max_iter,
            patience=patience,
            max_strenght=max_strenght,
            metric=stab.stability,
            inverted=True,
        )
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        **args,
    }


def run_iris_minover(n_dataset, max_iter, patience=100):
    args = locals()
    errs, lengths, ws = [], [], []
    for _ in range(n_dataset):
        data = d.iris_data(True)
        w, err = stab.minover(
            data,
            data,
            max_iter=max_iter,
            patience=patience,
            metric=stab.stability,
            inverted=True,
        )
        lengths.append(len(err))
        errs.append(err[-1])
    return {
        "error_mean": np.mean(errs),
        "error_std": np.std(errs),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        **args,
    }
