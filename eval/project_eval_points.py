### Reads img0, img1, pid, uv_clicked, uv_groundtruth from CSV. For each image pair, projects the 3D points from uv_clicked
### onto img1 and computes reprojection errors. Visualizes the results and plots RPE histogram. Reprojection errors are accumulated
### into one big list of all points from all image pairs.


import os
import sys
import cv2
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from math import sqrt
from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from baselines.VSLAM_LAB.path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION
from baselines.VSLAM_LAB.Baselines.colmap.scripts.python.read_write_model import read_model
from utilities import parse_yaml, get_colmap_image_by_name, project_colmap_point, plot_kpts_on_image_pair, plot_rpe_hist
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

def is_empty(val):
    if pd.isna(val):
        return True
    if isinstance(val, str):
        val = val.strip()
        return val == "" or val == "[]"
    return False

def find_pid(uv_clicked, sift_kpts, img):
    best_pid = -1
    for (i, sift_kpt) in enumerate(sift_kpts):
        u_sift, v_sift = sift_kpt
        dist = sqrt((u_sift - uv_clicked[0]) ** 2 + (v_sift - uv_clicked[1]) ** 2)
        if dist == 0:
            best_pid = img.point3D_ids[i]
    return best_pid


if __name__ == "__main__":
    
    METHOD = "OURS"   # 'ours' or 'colmap' or 'icp' or 'buffer' or 'vpr' or 'seq'
    
    parser = argparse.ArgumentParser(description="Count pairs below threshold in a distance matrix.")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    
    args = parser.parse_args()
    
    exp_name, dataset, subset, log_dir, dist_threshold = parse_yaml(args.exp_yaml)
    
    all_errors = []
    
    csv_files = [
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2017.csv",
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2018.csv",
        f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2017-2018.csv",
    ]

    if METHOD in ["OURS", "COLMAP", "VPR", "SEQ"]:
        model0 = Path(f'{VSLAMLAB_EVALUATION}/{exp_name}/{dataset}/{subset}/colmap_00000/0')
        model1 = model0
        
        rgb_path0 = Path(f'{VSLAMLAB_BENCHMARK}/{dataset}/{subset}/rgb_0')
        rgb_path1 = rgb_path0

    if METHOD in ["BUFFER", "ICP"]:
        if METHOD == "BUFFER":
            folder = "BUFFER"
        else:
            folder = "COLMAP"
        
        print(f"MODEL: {METHOD}, two models required. Evaluation not implemented yet. Exiting.")
        exit(1)
        
        # model0 = Path(f'/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/sangohenka/SESOKO/{folder}/ssk16-{subset}/colmap_00000/0_transformed')
        # rgb_path0 = Path(f'/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/SESOKO/ssk16-{subset}/rgb_0')
        
        # model1 = Path(f'/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/sangohenka/SESOKO/{folder}/ssk17-{subset}/colmap_00000/0_transformed')
        # rgb_path1 = Path(f'/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/SESOKO/ssk17-{subset}/rgb_0')
    
    # only use this method is evaluating a joint reconstruction (not ICP)
    for csv in csv_files:
        csv_file = Path(csv)
    
        if not csv_file.exists():
            print(f"⚠️  CSV file not found: {csv_file}")
            exit(0)
        df = pd.read_csv(csv_file)
        if "uv_clicked" not in df.columns or "uv_groundtruth" not in df.columns:
            print("⚠️  CSV file must contain 'uv_clicked' and 'uv_groundtruth' columns.")
            exit(0)
        
        evaluation_dir = csv_file.parent / "results" / METHOD.lower()
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # load images and colmap model
        cameras0, images0, points3D0 = read_model(model0, ext=".bin")
        cameras1, images1, points3D1 = read_model(model1, ext=".bin")

        reprojection_errors = []
        for idx, row in df.iterrows():
            
            if any(is_empty(row.get(col)) for col in ["img0", "img1", "uv_clicked", "uv_groundtruth"]):
                print(f"⚠️  Skipping row {idx}: empty required cell(s) detected.")
                continue

            img0_name = row["img0"]
            img1_name = row["img1"]
            uv_clicked = np.array(json.loads(row["uv_clicked"]), dtype=np.float32)
            uv_gt = np.array(json.loads(row["uv_groundtruth"]), dtype=np.float32)
                    
            path_0 = Path(rgb_path0 / img0_name)
            path_1 = Path(rgb_path1 / img1_name)

            img0 = get_colmap_image_by_name(images0, img0_name)
            img1 = get_colmap_image_by_name(images1, img1_name)
            
            if img0 is None or img1 is None:
                print(f"⚠️  Image not found in COLMAP model: {img0_name} or {img1_name}")
                continue
            
            camera1 = cameras1[img1.camera_id]
            h1, w1 = camera1.height, camera1.width

            print(f"Loaded {len(uv_clicked)} point pairs: {img0_name}  →  {img1_name}")
            
            sift_kpts = img0.xys 
            # print(sift_kpts)
            
            uv_projected = []
            for click, groundtruth in zip(uv_clicked, uv_gt):
                u_clicked, v_clicked = click
                u_gt, v_gt = groundtruth
                
                # find the pid
                pid = find_pid(click, sift_kpts, img0)
                if pid == -1:
                    print(f"⚠️  No 3D point associated with clicked keypoint at {click} in image {img0_name}, skipping.")
                    continue
                
                xyz = points3D0[pid].xyz
                u_proj, v_proj = project_colmap_point(xyz, img1, camera1)
                
                # exclude out-of-bounds points (comment this out if you want to include them)
                # if u_proj < 0 or u_proj >= w1 or v_proj < 0 or v_proj >= h1:
                #     print(f"⚠️  Transformed point ({u_proj:.2f}, {v_proj:.2f}) is out of bounds for image size ({w1}, {h1}). Skipping.")
                #     continue
                
                uv_projected.append([u_proj, v_proj])
                reprojection_error = sqrt((u_proj - u_gt) ** 2 + (v_proj - v_gt) ** 2)
                reprojection_errors.append(reprojection_error)
            
            # visualize results
            image0 = cv2.cvtColor(cv2.imread(str(path_0)), cv2.COLOR_BGR2RGB)
            image1 = cv2.cvtColor(cv2.imread(str(path_1)), cv2.COLOR_BGR2RGB)

            fig, axs = plot_kpts_on_image_pair(image0, image1, uv_clicked, uv_gt, uv_projected)
            fig.savefig(evaluation_dir / f"reprojection_{idx:03d}_{img0_name[:-4]}_to_{img1_name[:-4]}.png", bbox_inches="tight", dpi=300)
            # fig.savefig(evaluation_dir / f"reprojection_{idx:03d}_{img0_name[:-4]}_to_{img1_name[:-4]}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
        reprojection_errors = np.array(reprojection_errors)
        np.save(evaluation_dir / f"reprojection_errors_{csv_file.name[:-4]}.npy", reprojection_errors)
        plt.figure(figsize=(8, 5))
        rpe_plot = plot_rpe_hist(
            reprojection_errors,
            color=blue,
        )
        rpe_plot.savefig(evaluation_dir / f"reprojection_error_histogram_{csv_file.name[:-4]}.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        rpe_mean = np.mean(reprojection_errors)
        rpe_median = np.median(reprojection_errors)
        print(f"Mean RPE: {rpe_mean:.2f} pixels, Median RPE: {rpe_median:.2f} pixels")
        
        # write / append reprojection errors with header
        rpe_output_path = evaluation_dir / f"reprojection_errors.txt"

        with open(rpe_output_path, "a") as f:
            f.write("# ==================================================\n")
            f.write(f"# Model 0: {model0}\n")
            f.write(f"# Model 1: {model1}\n")
            f.write(f"# Image Pair CSV: {csv_file}\n")
            f.write(f"# Num points: {len(reprojection_errors)}\n")
            f.write(f"# Mean RPE (px): {rpe_mean:.6f}\n")
            f.write(f"# Median RPE (px): {rpe_median:.6f}\n")
            # f.write("# Reprojection errors (pixels):\n")

            # for e in reprojection_errors:
            #     f.write(f"{e:.6f}\n")

            f.write("\n")  # blank line between blocks

        print(f"[💾] Reprojection errors appended to: {rpe_output_path}")
        
        # plt.show()
        plt.close()
        
        