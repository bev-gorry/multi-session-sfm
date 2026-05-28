import os
import sys
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from utilities import parse_yaml, plot_rpe_hist
from constants import plotting_parameters, EVAL_POINTS_DIR

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",
    })

sns.set_context("paper", font_scale=plotting_parameters['font_scale'], rc={"lines.linewidth": plotting_parameters['linewidth'], "lines.markersize": plotting_parameters['markersize']})


BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
grey   = [c / 255.0 for c in GREY]

if __name__ == "__main__":
    
    METHOD = "OURS"   # 'ours' or 'colmap' or 'icp' or 'buffer' or 'vpr' or 'seq'
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    
    args = parser.parse_args()
    
    exp_name, dataset, subset, log_dir, dist_threshold = parse_yaml(args.exp_yaml)
    
    all_errors = []
    
    csv_files = [
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2017.csv",
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2018.csv",
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2017-2018.csv",
    ]
    
    methods = ["ours"]#, "vpr", "ours", "colmap", "icp", "buffer"]
    
    for method in methods:
            
        input_path = Path(f"{EVAL_POINTS_DIR}/{dataset}/{subset}/results/{method}")
        error_files = [
            input_path / "reprojection_errors_evaluation_points_2016-2017.npy",
            input_path / "reprojection_errors_evaluation_points_2016-2018.npy",
            input_path / "reprojection_errors_evaluation_points_2017-2018.npy",
        ]

        histogram_path = input_path / "reprojection_error.png"


        def load_reprojection_errors(path):
            path = Path(path)

            if path.suffix == ".npy":
                errors = np.load(path)

            elif path.suffix in [".csv", ".txt"]:
                errors = np.loadtxt(path, delimiter=",")

            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

            errors = np.asarray(errors).ravel()
            return errors


        reprojection_errors = []

        for f in error_files:
            errs = load_reprojection_errors(f)
            reprojection_errors.append(errs)

        reprojection_errors = np.concatenate(reprojection_errors)
        np.save(input_path / "reprojection_errors.npy", reprojection_errors)

        plt.figure(figsize=(8, 5))
        rpe_plot = plot_rpe_hist(
            reprojection_errors,
            color=blue,
        )
        rpe_plot.savefig(histogram_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        rpe_mean = np.mean(reprojection_errors)
        rpe_median = np.median(reprojection_errors)
        print(f"Mean RPE: {rpe_mean:.2f} pixels, Median RPE: {rpe_median:.2f} pixels")
        
        # write / append reprojection errors with header
        rpe_output_path = input_path / f"reprojection_errors.txt"

        with open(rpe_output_path, "a") as f:
            f.write("# ====================== CONCATENATED REPROJECTION ERRORS ======================\n")
            f.write(f"# Num points: {len(reprojection_errors)}\n")
            f.write(f"# Mean RPE (px): {rpe_mean:.6f}\n")
            f.write(f"# Median RPE (px): {rpe_median:.6f}\n")
            f.write("\n")

        print(f"[💾] Reprojection errors appended to: {rpe_output_path}")
