from nnci.exp import (
    run_perceptron,
    run_adatron,
    run_minover,
    run_iris_adatron,
    run_iris_minover,
)
from nnci.runner import runner
import numpy as np
import argparse

perceptron = {
    "alpha": np.unique(
        np.concatenate((np.linspace(0.5, 3, 11), np.linspace(1.5, 2.5, 10)))
    ),
    "N": [20, 40, 100, 200, 500, 1000],
    "n_datasets": [50, 100, 200],
    "max_iters": [100, 1000],
    "c": np.linspace(0, 3, 15),
    "FUNCTION": run_perceptron,
    "ORDER": [
        "alpha",
        "N",
        "n_datasets",
        "max_iters",
        "c",
    ],
}

base_minover = {
    "alpha": np.unique(
        np.concatenate(
            (
                np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
                np.linspace(0.1, 10, 15),
            )
        )
    ),
    "N": [20, 40, 100, 200],
    "n_dataset": [100],
    "max_iters": [500],
    "noise": [0],
    "FUNCTION": run_minover,
    "ORDER": [
        "alpha",
        "N",
        "n_dataset",
        "max_iters",
        "noise",
    ],
}


base_adatron = {
    "alpha": np.unique(
        np.concatenate(
            (
                np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
                np.linspace(0.1, 10, 15),
            )
        )
    ),
    "lr": np.linspace(0.05, 2, 10),
    "N": [20, 40],
    "n_dataset": [100],
    "max_iters": [500],
    "noise": [0],
    "max_strenght": [np.inf],
    "FUNCTION": run_adatron,
    "ORDER": [
        "alpha",
        "N",
        "n_dataset",
        "max_iters",
        "noise",
        "lr",
        "max_strenght",
    ],
}


noise_minover = {
    "alpha": np.unique(
        np.concatenate(
            (
                np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
                np.linspace(0.1, 10, 15),
            )
        )
    ),
    "N": [20, 40, 100, 200],
    "n_dataset": [100],
    "max_iters": [500],
    "noise": np.linspace(0.01, 0.5, 10),
    "FUNCTION": run_minover,
    "ORDER": [
        "alpha",
        "N",
        "n_dataset",
        "max_iters",
        "noise",
    ],
}


noise_adatron = {
    "alpha": np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "lr": np.linspace(0.05, 2, 5),
    "N": [20, 40],
    "n_dataset": [100],
    "max_iters": [500],
    "noise": np.linspace(0.01, 0.5, 5),
    "max_strenght": np.linspace(0, 200, 10),
    "FUNCTION": run_adatron,
    "ORDER": [
        "alpha",
        "N",
        "n_dataset",
        "max_iters",
        "noise",
        "lr",
        "max_strenght",
    ],
}

iris_minover = {
    "n_dataset": [100],
    "max_iters": [1000],
    "FUNCTION": run_iris_minover,
    "ORDER": [
        "n_dataset",
        "max_iters",
    ],
}

iris_adatron = {
    "n_dataset": [100],
    "max_iters": [1000],
    "lr": np.linspace(0.05, 2, 15),
    "max_strenght": np.linspace(0, 200, 10),
    "FUNCTION": run_iris_adatron,
    "ORDER": [
        "n_dataset",
        "max_iters",
        "lr",
        "max_strenght",
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        choices=[
            "perceptron",
            "base_minover",
            "base_adatron",
            "noise_minover",
            "noise_adatron",
            "iris_adatron",
            "iris_minover",
        ],
        help="Possible experiment choice",
        required=True,
    )
    parser.add_argument(
        "--file",
        default=None,
        type=str,
        required=False,
        help="the file name were to dump the results",
    )
    args = parser.parse_args()
    conf = eval(args.exp)
    runner(
        conf["FUNCTION"], tuple([conf[x] for x in conf["ORDER"]]), args.file or args.exp
    )
