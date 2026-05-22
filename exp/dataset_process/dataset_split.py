import numpy as np
import os
import random
import argparse
import shutil
import zipfile
from sklearn.model_selection import train_test_split

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# Set up argument parser to get dataset name from command line arguments
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--use_stratify", type=str, default="True", help="Whether to use stratify")
parser.add_argument("--chunk_size", type=int, default=2048, help="Rows written per chunk when splitting large datasets")

# Parse arguments
args = parser.parse_args()
infile = os.path.join("./datasets", f"{args.dataset}.npz")
dataset_path = os.path.join("./datasets", args.dataset)
os.makedirs(dataset_path, exist_ok=True)

assert os.path.exists(infile), f"{infile} does not exist!"

def extract_npz_member(npz_file, member, out_path):
    # Extract one .npy member from the .npz archive so numpy can mmap it later.
    # np.load(..., mmap_mode=...) does not mmap arrays while they are still inside a zip file.
    with zipfile.ZipFile(npz_file) as zf:
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)


def write_split(out_file, X, y, indices, chunk_size):
    # Write one split through temporary mmap-backed .npy files. This avoids
    # materializing the full train/valid/test arrays in RAM for large datasets.
    tmp_dir = f"{out_file}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    X_path = os.path.join(tmp_dir, "X.npy")
    y_path = os.path.join(tmp_dir, "y.npy")

    X_out = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=X.dtype,
        shape=(len(indices),) + X.shape[1:],
    )
    y_out = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=y.dtype,
        shape=(len(indices),) + y.shape[1:],
    )

    for start in range(0, len(indices), chunk_size):
        # Fancy indexing creates a copy, so keep each indexed batch small.
        end = min(start + chunk_size, len(indices))
        batch_idx = indices[start:end]
        X_out[start:end] = X[batch_idx]
        y_out[start:end] = y[batch_idx]

    X_out.flush()
    y_out.flush()
    del X_out, y_out

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        # Store .npy files without compression to match numpy savez behavior
        # while keeping the operation streaming and memory-light.
        zf.write(X_path, "X.npy")
        zf.write(y_path, "y.npy")
    shutil.rmtree(tmp_dir)


# Load dataset from the specified .npz file using mmap-backed .npy files.
print("loading...", infile)
tmp_path = os.path.join(dataset_path, "_split_tmp")
os.makedirs(tmp_path, exist_ok=True)
X_file = os.path.join(tmp_path, "X.npy")
y_file = os.path.join(tmp_path, "y.npy")
if not os.path.exists(X_file):
    print("extracting X.npy for mmap...")
    extract_npz_member(infile, "X.npy", X_file)
if not os.path.exists(y_file):
    print("extracting y.npy for mmap...")
    extract_npz_member(infile, "y.npy", y_file)

X = np.load(X_file, mmap_mode="r")
y = np.load(y_file, mmap_mode="r")

# Ensure labels are continuous
num_classes = len(np.unique(y))
assert num_classes == y.max() + 1, "Labels are not continuous"


if args.use_stratify == "True":
    # Split indices instead of arrays; the actual data is copied later in chunks.
    train_idx, test_idx = train_test_split(np.arange(len(y)), train_size=0.9, random_state=fix_seed, stratify=y)
    train_idx, valid_idx = train_test_split(train_idx, train_size=0.9, random_state=fix_seed, stratify=y[train_idx])
else:
    # Split indices instead of arrays; the actual data is copied later in chunks.
    train_idx, test_idx = train_test_split(np.arange(len(y)), train_size=0.9, random_state=fix_seed)
    train_idx, valid_idx = train_test_split(train_idx, train_size=0.9, random_state=fix_seed)

# Print dataset information
print(f"Train: X = {(len(train_idx),) + X.shape[1:]}, y = {(len(train_idx),) + y.shape[1:]}")
print(f"Valid: X = {(len(valid_idx),) + X.shape[1:]}, y = {(len(valid_idx),) + y.shape[1:]}")
print(f"Test: X = {(len(test_idx),) + X.shape[1:]}, y = {(len(test_idx),) + y.shape[1:]}")

# Save the split datasets into separate .npz files
print("writing train split...")
write_split(os.path.join(dataset_path, "train.npz"), X, y, train_idx, args.chunk_size)
print("writing valid split...")
write_split(os.path.join(dataset_path, "valid.npz"), X, y, valid_idx, args.chunk_size)
print("writing test split...")
write_split(os.path.join(dataset_path, "test.npz"), X, y, test_idx, args.chunk_size)
