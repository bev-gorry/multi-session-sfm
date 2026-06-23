import os
import sys
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from utilities import parse_yaml, plot_rpe_hist
from constants import plotting_parameters, EVAL_POINTS_DIR#, OVERLEAF_PATH

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{amsmath}",
})

sns.set_context(
    "paper",
    font_scale=4,
    rc={
        "lines.linewidth": plotting_parameters["linewidth"],
        "lines.markersize": plotting_parameters["markersize"],
    },
)


BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
PURPLE = [173, 113, 181]
GREY = [127, 127, 127]
GREEN = [113, 191, 110]
RED = [233, 72, 73]
ORANGE = [255, 153, 51]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
purple = [c / 255.0 for c in PURPLE]
grey   = [c / 255.0 for c in GREY]
green  = [c / 255.0 for c in GREEN]
red    = [c / 255.0 for c in RED]
orange = [c / 255.0 for c in ORANGE]


def load_reprojection_errors(path: Path):
    path = Path(path)

    if path.suffix == ".npy":
        errors = np.load(path)
    elif path.suffix in [".csv", ".txt"]:
        errors = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return np.asarray(errors).ravel()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    
    args = parser.parse_args()
    
    exp_name, dataset, subset, log_dir, _ = parse_yaml(args.exp_yaml)

    input_path = Path(f"{EVAL_POINTS_DIR}/{dataset}/{subset}/results/")

    error_candidates = [
        (input_path / "icp" / "reprojection_errors.npy", "COLMAP + ICP", yellow),
        (input_path / "buffer" / "reprojection_errors.npy", "COLMAP + BUFFER-X", blue),
        (input_path / "novpr" / "reprojection_errors.npy", "Ours (Without VPR)", green),
        (input_path / "ours" / "reprojection_errors.npy", "Ours", pink),
    ]

    histogram_path = input_path / f"{dataset.lower()}_{subset}_reprojection_error.png"

    # Load only existing error files
    labels = []
    colors = []
    rpe_list = []
    for path, label, color in error_candidates:
        if path.exists():
            try:
                rpe = load_reprojection_errors(path)
            except Exception:
                continue
            if rpe.size == 0:
                continue
            labels.append(label)
            colors.append(color)
            rpe_list.append(rpe)

    if len(rpe_list) == 0:
        print(f"[⚠️] No reprojection error files found in: {input_path}")
        sys.exit(0)
    
    fig, ax = plt.subplots(figsize=(9, 5))

    max_err = max(rpe.max() for rpe in rpe_list)
    bins = np.linspace(0, max_err, 50)

    for rpe, label, color in zip(rpe_list, labels, colors):
        ax.hist(
            rpe,
            bins=bins,
            alpha=0.8,
            color=color,
            label=label,
        )

    for rpe, color in zip(rpe_list, colors):
        ax.axvline(np.median(rpe), color=color, linestyle="--", linewidth=1)

    ax.set_xlim(0.0, max_err * 1.1)
    ax.set_xlabel("Reprojection Error (pixels)")
    ax.set_ylabel("\\# Keypoint Pairs")
    ax.grid(True, linewidth=0.25, alpha=0.8)

    axins = inset_axes(
        ax,
        width="60%",
        height="55%",
        loc="upper right",
        borderpad=1.0,
    )

    for rpe, color in zip(rpe_list, colors):
        axins.hist(rpe, bins=bins, alpha=0.8, color=color)

    for rpe, color in zip(rpe_list, colors):
        axins.axvline(np.median(rpe), color=color, linestyle="--", linewidth=1)

    # Zoom limits (tune if needed)
    zoom_max = np.percentile(np.concatenate(rpe_list), 85)
    axins.set_xlim(0.0, zoom_max)
    axins.set_ylim(0, None)

    axins.grid(True, linewidth=0.25, alpha=0.5)
    axins.tick_params(labelsize=14)

    # Connector lines
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)

    plt.savefig(histogram_path, bbox_inches="tight", dpi=300)
    # plt.savefig(overleaf_path, bbox_inches="tight", dpi=300)

    print(f"[💾] Histogram saved to: {histogram_path}.")
    
    plt.show()
    plt.close()