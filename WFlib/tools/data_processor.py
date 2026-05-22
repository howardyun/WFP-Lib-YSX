import os
import shutil
import torch
import zipfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

LAZY_LOAD_THRESHOLD = 512 * 1024 * 1024

def length_align(X, seq_len):
    """
    Align the length of the sequences to the specified sequence length.
    
    Parameters:
    X (ndarray): Input sequences.
    seq_len (int): Desired sequence length.

    Returns:
    ndarray: Aligned sequences with the specified length.
    """
    if seq_len < X.shape[-1]:
        X = X[...,:seq_len]  # Truncate the sequence if seq_len is shorter than the sequence length
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]  # Calculate padding length
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)  # Pad the sequence with zeros
    return X

def extract_npz_member(npz_file, member, out_path):
    # Extract one .npy member from .npz so numpy can mmap it instead of
    # loading the full array into memory.
    if os.path.exists(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(npz_file) as zf:
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)

def open_npz_as_mmap(data_path):
    cache_dir = f"{data_path}.mmap"
    X_file = os.path.join(cache_dir, "X.npy")
    y_file = os.path.join(cache_dir, "y.npy")
    if os.path.isdir(data_path):
        X_file = os.path.join(data_path, "X.npy")
        y_file = os.path.join(data_path, "y.npy")
        return np.load(X_file, mmap_mode="r"), np.load(y_file, mmap_mode="r")

    if os.path.isdir(cache_dir) and os.path.exists(data_path):
        cache_mtime = max(
            os.path.getmtime(X_file) if os.path.exists(X_file) else 0,
            os.path.getmtime(y_file) if os.path.exists(y_file) else 0,
        )
        if os.path.getmtime(data_path) > cache_mtime:
            shutil.rmtree(cache_dir, ignore_errors=True)

    try:
        extract_npz_member(data_path, "X.npy", X_file)
        extract_npz_member(data_path, "y.npy", y_file)
    except zipfile.BadZipFile:
        tmp_dir = f"{data_path}.tmp"
        X_file = os.path.join(tmp_dir, "X.npy")
        y_file = os.path.join(tmp_dir, "y.npy")
        if not os.path.exists(X_file) or not os.path.exists(y_file):
            raise
    return np.load(X_file, mmap_mode="r"), np.load(y_file, mmap_mode="r")

def align_sequence(sequence, seq_len):
    sequence = sequence[..., :seq_len]
    if seq_len > sequence.shape[-1]:
        pad_width = [(0, 0) for _ in range(len(sequence.shape) - 1)]
        pad_width.append((0, seq_len - sequence.shape[-1]))
        sequence = np.pad(sequence, pad_width=pad_width, mode="constant", constant_values=0)
    return sequence

class LazyFeatureArray:
    def __init__(self, data_path, feature_type, seq_len):
        self.X, self.y = open_npz_as_mmap(data_path)
        self.feature_type = feature_type
        self.seq_len = seq_len
        self.shape = self._feature_shape()

    def _feature_shape(self):
        base_shape = self.X.shape[:-1] + (min(self.seq_len, self.X.shape[-1]),)
        if self.seq_len > self.X.shape[-1]:
            base_shape = self.X.shape[:-1] + (self.seq_len,)
        if self.feature_type in ["DIR", "DT", "TAM"]:
            return (self.X.shape[0], 1) + base_shape[1:]
        if self.feature_type == "DT2":
            return (self.X.shape[0], 2, self.seq_len)
        return base_shape

    def __len__(self):
        return self.X.shape[0]

    def get_feature(self, index):
        row = self.X[index]
        if self.feature_type == "DIR":
            row = np.sign(align_sequence(row, self.seq_len))
            return row[np.newaxis].astype(np.float32, copy=True)
        if self.feature_type == "DT":
            row = align_sequence(row, self.seq_len)
            return row[np.newaxis].astype(np.float32, copy=True)
        if self.feature_type == "DT2":
            row_dir = np.sign(row)
            row_time = np.abs(row)
            row_time = np.diff(row_time)
            row_time[row_time < 0] = 0
            row_dir = align_sequence(row_dir, self.seq_len)
            row_time = align_sequence(row_time, self.seq_len)
            return np.stack([row_dir, row_time]).astype(np.float32, copy=True)
        if self.feature_type == "TAM":
            row = align_sequence(row, self.seq_len)
            return row[np.newaxis].astype(np.float32, copy=True)
        if self.feature_type in ["TAF", "MTAF"]:
            row = align_sequence(row, self.seq_len)
            return row.astype(np.float32, copy=True)
        if self.feature_type == "Origin":
            return align_sequence(row, self.seq_len)
        raise ValueError(f"Feature type {self.feature_type} is not matched.")

class LazyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X.get_feature(index)), self.y[index]

def load_data(data_path, feature_type, seq_len, num_tab=1):
    """
    Load and process data from a specified path.

    Parameters:
    data_path (str): Path to the data file.
    feature_type (str): Type of feature to extract.
    seq_len (int): Desired sequence length.

    Returns:
    tuple: Processed feature tensor and label tensor.
    """
    if os.path.getsize(data_path) >= LAZY_LOAD_THRESHOLD:
        X = LazyFeatureArray(data_path, feature_type, seq_len)
        if num_tab == 1:
            y = torch.tensor(X.y, dtype=torch.int64)
        else:
            y = torch.tensor(X.y, dtype=torch.float32)
        return X, y

    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if feature_type == "DIR":
        X = np.sign(X)  # Directional feature
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
    elif feature_type == "DT":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
    elif feature_type == "DT2":
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0  # Ensure no negative values
        X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
        X_time = length_align(X_time, seq_len)[:, np.newaxis]
        X = np.concatenate([X_dir, X_time], axis=1)
        X = torch.tensor(X, dtype=torch.float32)
    elif feature_type == "TAM":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
    elif feature_type in ["TAF", "MTAF"]:
        X = length_align(X, seq_len)
        X = torch.tensor(X, dtype=torch.float32)
    elif feature_type == "Origin":
        X = length_align(X, seq_len)
        return X, y
    else:
        raise ValueError(f"Feature type {feature_type} is not matched.")
    
    if num_tab == 1:
        y = torch.tensor(y, dtype=torch.int64)
    else:
        y = torch.tensor(y, dtype=torch.float32)

    return X, y

def load_iter(X, y, batch_size, is_train=True, num_workers=8, weight_sample=False):
    """
    Load data into an iterator for batch processing.

    Parameters:
    X (Tensor): Feature tensor.
    y (Tensor): Label tensor.
    batch_size (int): Number of samples per batch.
    is_train (bool): Whether the iterator is for training data.
    num_workers (int): Number of workers for data loading.
    weight_sample (bool): Whether to use weighted sampling.

    Returns:
    DataLoader: Data loader for batch processing.
    """
    if weight_sample:
        class_sample_count = np.unique(y.numpy(), return_counts=True)[1]
        weight = 1.0 / class_sample_count
        samples_weight = weight[y.numpy()]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            samples_weight, len(samples_weight)
        )
        if isinstance(X, LazyFeatureArray):
            dataset = LazyTensorDataset(X, y)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    if isinstance(X, LazyFeatureArray):
        dataset = LazyTensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def extract_temporal_feature(X, feat_length=1000):
    # Support both eager numpy arrays and lazy loaders returned by load_data().
    # The lazy path exposes get_feature(index), so we materialize one sample at
    # a time instead of forcing the whole dataset into memory.
    if hasattr(X, "get_feature"):
        abs_X = None
    else:
        abs_X = np.absolute(X)
    new_X = []

    for idx in tqdm(range(X.shape[0])):
        temporal_array = np.zeros((2,feat_length))
        row = X.get_feature(idx) if hasattr(X, "get_feature") else X[idx]
        abs_row = np.absolute(row)
        loading_time = abs_row.max()
        interval = 1.0 * loading_time / feat_length

        for packet in row:
            if packet == 0:
                break
            elif packet > 0:
                order = int(packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[0][order] += 1
            else:
                order = int(-packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[1][order] += 1
        new_X.append(temporal_array)
    new_X = np.array(new_X)
    return new_X

def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths

def agg_interval(packets):
    features = []
    features.append([np.sum(packets>0), np.sum(packets<0)])

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts>0), np.sum(bursts<0)])

    pos_bursts = bursts[bursts>0]
    neg_bursts = np.abs(bursts[bursts<0])
    vals = []
    if len(pos_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(neg_bursts))
    features.append(vals)

    return np.array(features, dtype=np.float32)

def agg_interval2(packets):
    features = []
    features.append(np.sum(packets>0))
    features.append(np.sum(packets<0))

    pos_packets = packets[packets>0]
    neg_packets = np.abs(packets[packets<0])
    features.append(np.sum(np.diff(pos_packets)))
    features.append(np.sum(np.diff(neg_packets)))

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append(np.sum(bursts>0))
    features.append(np.sum(bursts<0))

    pos_bursts = bursts[bursts>0]
    neg_bursts = np.abs(bursts[bursts<0])
    if len(pos_bursts) == 0:
        features.append(0)
    else:
        features.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        features.append(0)
    else:
        features.append(np.mean(neg_bursts))

    return np.array(features, dtype=np.float32)

def process_MTAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((8, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[..., interval_idx] = agg_interval2(cur_packets)
        st_pos = ed_pos
    
    return index, TAF

def extract_MTAF(sequences, num_workers=30):
    """
    Extract the TAF from sequences.

    Parameters:
    sequences (ndarray): Input sequences.

    Returns:
    ndarray: Extracted TAF.
    """
    interval = 20
    max_len = 8000
    sequences *= 1000
    num_sequences = sequences.shape[0]
    TAF = np.zeros((num_sequences, 8, max_len))

    with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
        futures = [executor.submit(process_MTAF, index, sequences[index], interval, max_len) for index in range(num_sequences)]
        with tqdm(total=num_sequences) as pbar:
            for future in as_completed(futures):
                index, result = future.result()
                TAF[index] = result
                pbar.update(1)

    return TAF

def process_TAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((3, 2, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[:, :, interval_idx] = agg_interval(cur_packets)
        st_pos = ed_pos
    
    return index, TAF

def extract_TAF(sequences, num_workers=30):
    """
    Extract the TAF from sequences.

    Parameters:
    sequences (ndarray): Input sequences.

    Returns:
    ndarray: Extracted TAF.
    """
    interval = 40
    max_len = 2000
    sequences *= 1000
    num_sequences = sequences.shape[0]
    TAF = np.zeros((num_sequences, 3, 2, max_len))

    with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
        futures = [executor.submit(process_TAF, index, sequences[index], interval, max_len) for index in range(num_sequences)]
        with tqdm(total=num_sequences) as pbar:
            for future in as_completed(futures):
                index, result = future.result()
                TAF[index] = result
                pbar.update(1)

    return TAF

def process_TAM(index, sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    for pack in sequence:
        if pack == 0:
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    return index, feature

def extract_TAM(sequences, num_workers=30):
    """
    Extract the Traffic Analysis Matrix (TAM) from sequences.

    Parameters:
    sequences (ndarray): Input sequences.

    Returns:
    ndarray: Extracted TAM features.
    """
    maximum_load_time = 80  # Maximum load time for packets
    max_matrix_len = 1800  # Maximum length of the matrix
    num_sequences = sequences.shape[0]
    TAM = np.zeros((num_sequences, 2, max_matrix_len))

    with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
        futures = [executor.submit(process_TAM, index, sequences[index], maximum_load_time, max_matrix_len) for index in range(num_sequences)]
        with tqdm(total=num_sequences) as pbar:
            for future in as_completed(futures):
                index, result = future.result()
                TAM[index] = result
                pbar.update(1)

    return TAM
