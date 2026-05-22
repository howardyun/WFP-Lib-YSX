# Generates the traffic aggregation features. More details can be found in the paper: 
# Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis. CCS 2024.
import numpy as np
import os
import argparse
import shutil
from typing import List
import time
import random
import torch
import zipfile
from tqdm import tqdm
from multiprocessing import Process
from WFlib.tools import data_processor

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument("--in_file", type=str, default="train", help="input file")
parser.add_argument("--chunk_size", type=int, default=1024, help="Rows flushed per chunk when writing large TAF files")

# Parse arguments
args = parser.parse_args()
in_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

# Define output file path
out_file = os.path.join(in_path, f"taf_{args.in_file}.npz")

def write_npz_from_npy(out_file, X_path, y_path):
    # Store the generated .npy files inside an .npz archive without loading
    # them into RAM. This keeps compatibility with np.load(...)[\"X\"/\"y\"].
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(X_path, "X.npy")
        zf.write(y_path, "y.npy")

# If the output file does not exist, process the input file
if not os.path.exists(out_file):
    # Load the source split through mmap-backed .npy files. Large .npz files
    # cannot be memory-mapped directly, so data_processor extracts a reusable
    # .mmap cache next to the source file.
    in_file = os.path.join(in_path, f"{args.in_file}.npz")
    X, y = data_processor.open_npz_as_mmap(in_file)

    # TAF has a fixed internal length of 2000 in data_processor.extract_TAF.
    # Use a temporary mmap output so we never hold the full TAF tensor in RAM.
    tmp_dir = f"{out_file}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    X_path = os.path.join(tmp_dir, "X.npy")
    y_path = os.path.join(tmp_dir, "y.npy")
    X_out = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(X.shape[0], 3, 2, 2000),
    )
    y_out = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=y.dtype,
        shape=y.shape,
    )
    y_out[:] = y[:]

    for start in tqdm(range(0, X.shape[0], args.chunk_size)):
        end = min(start + args.chunk_size, X.shape[0])
        for index in range(start, end):
            sequence = data_processor.align_sequence(X[index], args.seq_len)
            _, taf = data_processor.process_TAF(index, sequence * 1000, 40, 2000)
            X_out[index] = taf.astype(np.float32, copy=False)
        X_out.flush()

    y_out.flush()
    del X_out, y_out
    write_npz_from_npy(out_file, X_path, y_path)
    shutil.rmtree(tmp_dir)
    print(f"{args.in_file} process done: X = {(X.shape[0], 3, 2, 2000)}, y = {y.shape}")
else:
    # Print a message if the output file already exists
    print(f"{out_file} has been generated.")
