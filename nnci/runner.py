import os
import csv
from tqdm import tqdm
from itertools import product


def writer(data, file_path, hearder: bool = False):
    if not data:
        return
    mode = "w" if hearder else "a"
    with open(file_path, mode) as fil:
        w = csv.DictWriter(fil, fieldnames=data[0].keys())
        if hearder:
            w.writeheader()
        else:
            w.writerows(data)


def check_file(file_path):
    while os.path.exists(file_path):
        print(f"Warning: {file_path} already exists and changing file name.")
        root, ext = os.path.splitext(file_path)
        file_path = root + "0" + ext
        print(f"new file_name {file_path}")
    return file_path


def runner(fun, list_args, file_name: str, buffer_size: int = 50):
    first = True
    res = []
    root, _ = os.path.splitext(file_name)
    file_path = check_file("results/" + root + ".csv")
    args = list(product(*list_args))
    for x in tqdm(args):
        res.append(fun(*x))
        if first:
            writer(res, file_path, hearder=True)
            first = False
        if len(res) > buffer_size:
            writer(res, file_path)
            res = []
    writer(res, file_path)
