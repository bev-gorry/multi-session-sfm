### Reads a distance matrix and a csv file of image paths (the same csv file used to compute D.npy)
### and selects a few image pairs from each sequence pair that are far apart in the matrix (to avoid 
### redundancy) and below a distance threshold. Writes out evaluation_points_*.csv files for each 
### sequence pair, which can be used to select points for reprojection evaluation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

matrix = np.load("/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/EIFFEL/effall-full/D.npy")
df = pd.read_csv("/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/exp_eff_full_ours/EIFFEL/effall-full/rgb_exp.csv")

DIST_THRESHOLD = 0.45
N_PER_PAIR = 4
OUT_DIR = Path("/home/beverley/Repos/multi-session-sfm/reprojection/EIFFEL/effall-full")  # where the evaluation_points_*.csv files go

yellow_text = "\033[93m"
reset_text = "\033[0m"
print(f"{yellow_text}DISTANCE MATRIX PATH: {matrix}{reset_text}")
print(f"{yellow_text}RGB.CSV PATH: {df}{reset_text}")
print(f"{yellow_text}OUTPUT DIRECTORY: {OUT_DIR}{reset_text}")
confirm = input(f"Please confirm that the above hard-coded paths are correct (Y/n): ").strip().lower()
if confirm not in ["", "y"]:
    print("Exiting. Please edit the paths in the script and re-run.")
    exit(0)

# map sequence name -> year, e.g. 'eff16' -> '2016'
def seq_to_year(seq_name):
    return "20" + "".join(filter(str.isdigit, seq_name))

seq_names = df["sequence_name"].unique().tolist()
seq_sizes = [(df["sequence_name"] == s).sum() for s in seq_names]
offsets = dict(zip(seq_names, np.concatenate(([0], np.cumsum(seq_sizes)[:-1]))))
sizes = dict(zip(seq_names, seq_sizes))


def select_spread_matches(matrix, seq_a, seq_b, n=4, threshold=0.45):
    a0, b0 = offsets[seq_a], offsets[seq_b]
    block = matrix[a0:a0 + sizes[seq_a], b0:b0 + sizes[seq_b]]

    cand_i, cand_j = np.where(block < threshold)
    if len(cand_i) == 0:
        print(f"⚠️  No matches under {threshold} for {seq_a}-{seq_b}")
        return []
    cand = np.stack([cand_i, cand_j], axis=1).astype(np.float64)
    dists = block[cand_i, cand_j]

    scale = np.array([sizes[seq_a], sizes[seq_b]], dtype=np.float64)
    cand_n = cand / scale

    selected = [int(np.argmin(dists))]
    while len(selected) < min(n, len(cand)):
        sel_pts = cand_n[selected]
        d_to_sel = np.linalg.norm(cand_n[:, None] - sel_pts[None], axis=-1).min(axis=1)
        d_to_sel[selected] = -1
        selected.append(int(np.argmax(d_to_sel)))

    out = []
    for s in selected:
        i, j = int(cand[s, 0]), int(cand[s, 1])
        out.append({
            "seq_a": seq_a, "seq_b": seq_b,
            "i": i, "j": j, "gi": a0 + i, "gj": b0 + j,
            "img0": Path(df.iloc[a0 + i]["path_rgb_0"]).name,   # bare filename
            "img1": Path(df.iloc[b0 + j]["path_rgb_0"]).name,
            "dist": float(block[i, j]),
        })
    return out


pairs = [(seq_names[0], seq_names[1]),
         (seq_names[0], seq_names[2]),
         (seq_names[1], seq_names[2])]

all_matches = []
for sa, sb in pairs:
    matches = select_spread_matches(matrix, sa, sb, n=N_PER_PAIR, threshold=DIST_THRESHOLD)
    all_matches += matches

    # write evaluation_points_yearA-yearB.csv
    year_a, year_b = seq_to_year(sa), seq_to_year(sb)
    out_csv = OUT_DIR / f"evaluation_points_{year_a}-{year_b}.csv"
    eval_df = pd.DataFrame({
        "img0": [m["img0"] for m in matches],
        "img1": [m["img1"] for m in matches],
        "uv_clicked": [""] * len(matches),
        "uv_groundtruth": [""] * len(matches),
    })
    eval_df.to_csv(out_csv, index=False)
    print(f"[💾] {out_csv}  ({len(matches)} pairs)")

for m in all_matches:
    print(f"{m['seq_a']}[{m['i']}] <-> {m['seq_b']}[{m['j']}]  d={m['dist']:.3f}   {m['img0']} <-> {m['img1']}")

# ---- plot matrix with selections ----
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(matrix, cmap="viridis")
plt.colorbar(im, ax=ax, fraction=0.046, label="VPR distance")

bounds = np.cumsum(seq_sizes)
for b in bounds[:-1]:
    ax.axhline(b, color="white", lw=0.8, alpha=0.7)
    ax.axvline(b, color="white", lw=0.8, alpha=0.7)
centers = np.concatenate(([0], bounds[:-1])) + np.array(seq_sizes) / 2
ax.set_xticks(centers); ax.set_xticklabels(seq_names)
ax.set_yticks(centers); ax.set_yticklabels(seq_names)

colors = {pairs[0]: "red", pairs[1]: "orange", pairs[2]: "magenta"}
for m in all_matches:
    c = colors[(m["seq_a"], m["seq_b"])]
    ax.plot(m["gj"], m["gi"], "o", ms=10, mfc="none", mec=c, mew=2)
    ax.plot(m["gi"], m["gj"], "o", ms=10, mfc="none", mec=c, mew=2)

handles = [plt.Line2D([], [], marker="o", mfc="none", mec=c, ls="",
                      label=f"{seq_to_year(sa)}-{seq_to_year(sb)}")
           for (sa, sb), c in colors.items()]
ax.legend(handles=handles, loc="upper right")
ax.set_title(f"VPR matrix — selected matches (d < {DIST_THRESHOLD})")
plt.tight_layout()
plt.show()