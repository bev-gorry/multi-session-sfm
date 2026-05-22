### Take colmap model, iterate over all 3D points. Get images that see the point and which sequence they are in. Set rgb
### of point and write model 0_colored.


import os
import sys
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from utilities.parse_yaml import parse_yaml

from baselines.VSLAM_LAB.path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION

from baselines.VSLAM_LAB.Baselines.colmap.scripts.python.read_write_model import read_model, write_model


BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

def tint_rgb(original_rgb, tint_rgb, alpha=0.35):
    """
    original_rgb: np.ndarray shape (3,), uint8
    tint_rgb: list or np.ndarray shape (3,)
    alpha: float in [0,1]
    """
    return np.clip(
        (1 - alpha) * original_rgb.astype(np.float32)
        + alpha * np.array(tint_rgb, dtype=np.float32),
        0, 255
    ).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count pairs below threshold in a distance matrix.")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    
    args = parser.parse_args()
    
    exp_name, dataset, subset, log_dir, dist_threshold = parse_yaml(args.exp_yaml)
    
    alpha_tint = 0.4
    
    # subsets = ["s01", "s02", "s03", "s04"]
    # subsets = ["s30"]
    # for subset in subsets:
        
    model_path = Path(f'{VSLAMLAB_EVALUATION}/{exp_name}/{dataset}/{subset}/colmap_00000/0')
    rgb_path = Path(f'{VSLAMLAB_BENCHMARK}/{dataset}/{subset}/rgb_0')
    
    rgb_csv = rgb_path.parent / 'rgb.csv'
    rgb_df = pd.read_csv(rgb_csv)
    
    cameras, images, points3D = read_model(model_path, ext='.bin')
    
    numpoints_ssk16 = 0
    numpoints_ssk17 = 0
    numpoints_ssk18 = 0
    numpoints_ssk16_ssk17 = 0
    numpoints_ssk17_ssk18 = 0
    numpoints_ssk16_ssk18 = 0
    numpoints_ssk16_ssk17_ssk18 = 0
    for pid, point in tqdm(points3D.items(), desc="Coloring point cloud by sequence"):
        
        # if pid > 100:
        #     break

        sequences_that_see_this_point = set()
        
        xyz = point.xyz
        img_ids = points3D[pid].image_ids
        points2d_ids = points3D[pid].point2D_idxs
        orig_rgb = point.rgb.copy()
        
        for img_id, point2D_id in zip(img_ids, points2d_ids):
                image = images[img_id]
                image_name = Path(image.name).name
                camera = cameras[image.camera_id]
                sequence_name = rgb_df.loc[rgb_df["path_rgb_0"] == f"rgb_0/{image_name}", "sequence_name"].iloc[0]
                
                sequences_that_see_this_point.add(sequence_name)
                
        if "ssk18" in sequences_that_see_this_point and "ssk17" in sequences_that_see_this_point and "ssk16" in sequences_that_see_this_point:
            # point.rgb[:] = np.array(GREY, dtype=np.uint8)
            point.rgb[:] = tint_rgb(orig_rgb, GREY, alpha=alpha_tint)
            numpoints_ssk16_ssk17_ssk18 += 1
        if "ssk16" in sequences_that_see_this_point and "ssk17" in sequences_that_see_this_point and "ssk18" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(GREY, dtype=np.uint8)
            point.rgb[:] = tint_rgb(orig_rgb, GREY, alpha=alpha_tint)
            numpoints_ssk16_ssk17 += 1
        if "ssk17" in sequences_that_see_this_point and "ssk18" in sequences_that_see_this_point and "ssk16" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(GREY, dtype=np.uint8)
            point.rgb[:] = tint_rgb(orig_rgb, GREY, alpha=alpha_tint)
            numpoints_ssk17_ssk18 += 1
        if "ssk16" in sequences_that_see_this_point and "ssk18" in sequences_that_see_this_point and "ssk17" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(GREY, dtype=np.uint8)
            point.rgb[:] = tint_rgb(orig_rgb, GREY, alpha=alpha_tint)
            numpoints_ssk16_ssk18 += 1
        if "ssk16" in sequences_that_see_this_point and "ssk17" not in sequences_that_see_this_point and "ssk18" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(BLUE, dtype=np.uint8)    # seen in 2016
            point.rgb[:] = tint_rgb(orig_rgb, BLUE, alpha=alpha_tint)
            numpoints_ssk16 += 1
        if "ssk17" in sequences_that_see_this_point and "ssk16" not in sequences_that_see_this_point and "ssk18" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(YELLOW, dtype=np.uint8)  # seen in 2017
            point.rgb[:] = tint_rgb(orig_rgb, YELLOW, alpha=alpha_tint)
            numpoints_ssk17 += 1
        if "ssk18" in sequences_that_see_this_point and "ssk16" not in sequences_that_see_this_point and "ssk17" not in sequences_that_see_this_point:
            # point.rgb[:] = np.array(PINK, dtype=np.uint8)   # seen in 2018
            point.rgb[:] = tint_rgb(orig_rgb, PINK, alpha=alpha_tint)
            numpoints_ssk18 += 1


    model_root = Path(model_path).parent
    colored_output_path = model_root / "0_colored"
    colored_output_path.mkdir(exist_ok=True)

    print(f"\n[💾] Writing sequence-colored COLMAP model to: {colored_output_path}")
    write_model(cameras, images, points3D, str(colored_output_path), ext=".bin")