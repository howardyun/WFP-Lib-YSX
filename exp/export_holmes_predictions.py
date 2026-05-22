import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from WFlib import models
from WFlib.tools import data_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Export Holmes predictions for Adaptive-Tamaraw")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="Holmes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_file", type=str, default="test")
    parser.add_argument("--feature", type=str, default="TAF")
    parser.add_argument("--seq_len", type=int, default=2000)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--load_name", type=str, default="max_f1")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="Open-world")
    parser.add_argument("--open_threshold", type=float, default=1e-2)
    parser.add_argument("--num_tabs", type=int, default=1)
    return parser.parse_args()


def load_embeddings(model, loader, device):
    embs = []
    ys = []
    model.eval()
    with torch.no_grad():
        for cur_X, cur_y in loader:
            cur_X = cur_X.to(device)
            out = model(cur_X).cpu().numpy()
            embs.append(out)
            ys.append(cur_y.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)


def main():
    args = parse_args()
    fix_seed = 2024
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
    device = torch.device(args.device)

    in_path = os.path.join("./datasets", args.dataset)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    out_dir = args.out_dir or os.path.join(ckp_path, "holmes_predictions")
    os.makedirs(out_dir, exist_ok=True)

    test_file = os.path.join(in_path, f"{args.test_file}.npz")
    test_X, test_y = data_processor.load_data(test_file, args.feature, args.seq_len)
    test_iter = data_processor.load_iter(test_X, test_y, 256, False, 10)

    model = eval(f"models.{args.model}")(len(np.unique(test_y)))
    model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu"))
    model.to(device)

    spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
    assert os.path.exists(spatial_dist_file), f"{spatial_dist_file} does not exist"
    spatial_data = np.load(spatial_dist_file)
    webs_centroid = spatial_data["centroid"]
    webs_radius = spatial_data["radius"]

    embs, y_true = load_embeddings(model, test_iter, device)
    all_sims = 1 - cosine_similarity(embs, webs_centroid)
    all_sims -= webs_radius
    outs = np.argmin(all_sims, axis=1)
    if args.scenario == "Open-world":
        outs_d = np.min(all_sims, axis=1)
        outs[outs_d > args.open_threshold] = len(np.unique(test_y)) - 1

    np.save(os.path.join(out_dir, f"taf_{args.test_file}_Holmes_predictions.npy"), outs.astype(np.int64))
    np.save(os.path.join(out_dir, "test_indices.npy"), np.arange(len(outs), dtype=np.int64))
    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
