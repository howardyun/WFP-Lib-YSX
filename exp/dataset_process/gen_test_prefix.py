import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm

from WFlib.tools import data_processor


def write_npz_from_npy(out_file, X_path, y_path):
    import zipfile

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(X_path, "X.npy")
        zf.write(y_path, "y.npy")


def truncate_sequence(sequence, percent):
    nonzero = sequence[sequence != 0]
    if nonzero.size == 0:
        return np.zeros_like(sequence)
    keep = int(np.ceil(nonzero.shape[0] * percent / 100.0))
    keep = max(1, min(keep, nonzero.shape[0]))
    out = np.zeros_like(sequence)
    out[:keep] = nonzero[:keep]
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate Holmes prefix test sets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--percent", type=int, required=True)
    parser.add_argument("--in_file", type=str, default="test")
    args = parser.parse_args()

    in_path = os.path.join("./datasets", args.dataset)
    src_file = os.path.join(in_path, f"{args.in_file}.npz")
    if not os.path.exists(src_file):
        raise FileNotFoundError(src_file)

    out_file = os.path.join(in_path, f"{args.in_file}_p{args.percent}.npz")
    if os.path.exists(out_file):
        print(f"{out_file} has been generated.")
        return

    X, y = data_processor.open_npz_as_mmap(src_file)
    tmp_dir = f"{out_file}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    X_path = os.path.join(tmp_dir, "X.npy")
    y_path = os.path.join(tmp_dir, "y.npy")
    X_out = np.lib.format.open_memmap(X_path, mode="w+", dtype=X.dtype, shape=X.shape)
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype=y.dtype, shape=y.shape)
    y_out[:] = y[:]

    for index in tqdm(range(X.shape[0]), desc=f"test_p{args.percent}"):
        X_out[index] = truncate_sequence(X[index], args.percent)
    X_out.flush()
    y_out.flush()
    del X_out, y_out

    write_npz_from_npy(out_file, X_path, y_path)
    shutil.rmtree(tmp_dir)
    print(f"Generate {out_file} done.")


if __name__ == "__main__":
    main()
