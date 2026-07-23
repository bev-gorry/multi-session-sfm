import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

matrix = np.load("/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/EIFFEL/effall-full/D.npy")
df = pd.read_csv("/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/exp_eff_full_ours/EIFFEL/effall-full/rgb_exp.csv")

# Root folder containing the sequence folders (adjust to your layout)
DATA_ROOT = Path("/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/EIFFEL/effall-full/rgb_0")

def img_path(seq_name, rel_path):
    fname = Path(rel_path).name
    hits = list(DATA_ROOT.rglob(fname))
    if not hits:
        raise FileNotFoundError(f"{fname} not found under {DATA_ROOT}")
    return hits[0]

seq_names = df["sequence_name"].unique().tolist()
seq_sizes = [(df["sequence_name"] == s).sum() for s in seq_names]
offsets = dict(zip(seq_names, np.concatenate(([0], np.cumsum(seq_sizes)[:-1]))))
sizes = dict(zip(seq_names, seq_sizes))

def best_match(matrix, df, seq_a, seq_b):
    a0, b0 = offsets[seq_a], offsets[seq_b]
    block = matrix[a0:a0 + sizes[seq_a], b0:b0 + sizes[seq_b]]
    i, j = np.unravel_index(np.argmin(block), block.shape)
    img_a = df.iloc[a0 + i]["path_rgb_0"]
    img_b = df.iloc[b0 + j]["path_rgb_0"]
    return i, j, img_a, img_b, block[i, j]

def show_pair(seq_a, seq_b):
    i, j, rel_a, rel_b, dist = best_match(matrix, df, seq_a, seq_b)
    path_a = img_path(seq_a, rel_a)
    path_b = img_path(seq_b, rel_b)
    print(f"{seq_a}[{i}] {path_a}")
    print(f"{seq_b}[{j}] {path_b}")
    print(f"distance = {dist}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mpimg.imread(path_a))
    axes[0].set_title(f"{seq_a}[{i}]\n{rel_a}")
    axes[1].imshow(mpimg.imread(path_b))
    axes[1].set_title(f"{seq_b}[{j}]\n{rel_b}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Best VPR match — distance = {dist:.4f}")
    plt.tight_layout()
    plt.show()

show_pair("eff16", "eff20")