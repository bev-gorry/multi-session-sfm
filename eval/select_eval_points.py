### Take image pair and combined model. For each clicked 2D point in image0, transform it to image 1 using homography.
### Viewing the points on image1 click on a corresponding keypoint and click a new location to move / correct it.
### Write img0 (string), img1 (string), uv_clicked (list of pairs), uv_groundtruth (list of pairs) to csv.


import os
import sys
import cv2
import json
import torch
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from baselines.VSLAM_LAB.path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION
from baselines.VSLAM_LAB.Baselines.colmap.scripts.python.read_write_model import read_model
from baselines.VSLAM_LAB.Baselines.LightGlue.lightglue import LightGlue, SuperPoint
from utilities import parse_yaml, get_colmap_image_by_name, unrotate_kps_W, extract_keypoints, feature_matching, show_image_with_clickable_points, plot_kpts_on_image_pair
from constants import EVAL_POINTS_DIR

BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
grey   = [c / 255.0 for c in GREY]

class PointEditor:
    def __init__(self, ax, points, select_radius=12):
        self.ax = ax
        self.points = points
        self.active_idx = None
        self.select_radius = select_radius

        self.scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c='green',
            s=120,
            marker='o',
            edgecolors='white'
        )

        fig = ax.figure
        fig.canvas.mpl_connect("button_press_event", self.on_click)

    def find_nearest_point(self, x, y):
        dist = np.linalg.norm(self.points - np.array([x, y]), axis=1)
        idx = np.argmin(dist)
        if dist[idx] <= self.select_radius:
            return idx
        return None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        # If no active point, select nearest
        if self.active_idx is None:
            idx = self.find_nearest_point(x, y)
            if idx is not None:
                self.active_idx = idx
                print(f"[SELECT] Point {idx} selected")
            return

        # Otherwise, move the active point
        print(f"[MOVE]   Point {self.active_idx} → ({x:.1f}, {y:.1f})")
        self.points[self.active_idx] = [x, y]
        self.scatter.set_offsets(self.points)
        self.ax.figure.canvas.draw_idle()

        # reset
        self.active_idx = None

def compute_homography(path_0, path_1):
    feats_dict0, feats_rot0, h0, w0 = extract_keypoints(path_0, features="superpoint")
    feats_dict1, feats_rot1, h1, w1 = extract_keypoints(path_1, features="superpoint")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1, flash=True).eval().to(device)
    
    matches_tensor = feature_matching(feats_rot0, feats_rot1, matcher=matcher, exhaustive=True) 
    print(len(matches_tensor))

    pts0 = feats_dict0['keypoints'].squeeze(0).cpu().numpy().astype(np.float32)
    pts1 = feats_dict1['keypoints'].squeeze(0).cpu().numpy().astype(np.float32)

    rot0 = feats_dict0['rotations'].squeeze(0).cpu().numpy().astype(np.float32)
    rot1 = feats_dict1['rotations'].squeeze(0).cpu().numpy().astype(np.float32)

    pts0 = unrotate_kps_W(pts0, rot0, h0, w0)
    pts1 = unrotate_kps_W(pts1, rot1, h1, w1)

    pts0 = pts0[matches_tensor[:,0]]
    pts1 = pts1[matches_tensor[:,1]]

    H, inlier_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        
    return H, inlier_mask, pts0, pts1

def plot_warped_image(H, inliers, pts0, pts1, path_to_image0, path_to_image1, fig, ax, uv_projected=None, uv_groundtruth=None):

    if H is None:
        raise RuntimeError("cv2.findHomography failed; not enough good matches or points are degenerate.")

    mask = inliers.ravel().astype(bool)
    pts0 = pts0[mask]
    pts1 = pts1[mask]
    print(len(pts0))
    print(len(pts1))
    # --- Load images via OpenCV for warping (H,W,C in BGR) ---
    cv_img0 = cv2.imread(path_to_image0, cv2.IMREAD_COLOR)
    cv_img1 = cv2.imread(path_to_image1, cv2.IMREAD_COLOR)
    h, w = cv_img1.shape[:2] 

    # --- Warp and display ---
    warped = cv2.warpPerspective(cv_img0, H, (w, h))
    projected_kpts = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2), H).reshape(-1, 2)

    alpha = 0.25
    blue_bgr = (BLUE[2], BLUE[1], BLUE[0] )# (180, 119, 31)
    yellow_bgr = (YELLOW[2], YELLOW[1], YELLOW[0])   
    
    tint_filter = np.full_like(cv_img1, blue_bgr, dtype=np.uint8)
    tinted_img = cv2.addWeighted(cv_img1, 1 - alpha, tint_filter, alpha, 0)

    # create a mask for the warped query image and the database image
    fg_mask = (warped != 0).any(axis=-1)
    fg_mask_uint8 = fg_mask.astype(np.uint8) * 255
    fg_contours, _ = cv2.findContours(fg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_bg = (warped == 0).all(axis=-1)
    mask_uint8 = mask_bg.astype(np.uint8) * 255
    bg_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # remove the overlap from the background mask to keep coloring clear
    composite = warped.copy()
    composite[mask_bg] = tinted_img[mask_bg]

    cv2.drawContours(composite, bg_contours, -1, blue_bgr, thickness=4)   
    cv2.drawContours(composite, fg_contours, -1, yellow_bgr, thickness=3) 
                
    # plot actual keypoints from the database image and projected keypoints from the query image
    #plt.figure(figsize=(14, 14))
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    ax.scatter(pts1[:, 0], pts1[:, 1], c='blue', marker='o',edgecolors='white', s=30, label='Actual Keypoints')
    ax.scatter(projected_kpts[:, 0], projected_kpts[:, 1], c=[yellow], s=30, marker='o', edgecolors='white', label='Warped Keypoints')
    if uv_projected:
        ax.scatter(uv_projected[:, 0], uv_projected[:, 1], c='red', s=150, marker='X', linewidths=1, edgecolors='white', label="Projected Point")
    ax.scatter(uv_groundtruth[:, 0], uv_groundtruth[:, 1], c='green', s=150, marker='X', linewidths=1, edgecolors='white', label='Corresponding Keypoint')
    ax.axis('off')
    
    return fig, ax
    #plt.show()

def is_populated(val):
    if pd.isna(val):
        return False
    if isinstance(val, str):
        val = val.strip()
        if val == "" or val == "[]":
            return False
        return True
    return True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    
    args = parser.parse_args()
    
    exp_name, dataset, subset, log_dir, dist_threshold = parse_yaml(args.exp_yaml)
    
    model0 = Path(f'{VSLAMLAB_EVALUATION}/{exp_name}/{dataset}/{subset}/colmap_00000/0')
    model1 = model0
    
    rgb_path0 = Path(f'{VSLAMLAB_BENCHMARK}/{dataset}/{subset}/rgb_0')
    rgb_path1 = rgb_path0
    
    csv_file = Path(f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2017.csv")

    # read csv file to get image pairs
    if not csv_file.exists():
        print(f"⚠️  CSV file not found: {csv_file}")
        exit(0)
    df = pd.read_csv(csv_file)
    
    if "uv_clicked" not in df.columns:
        df["uv_clicked"] = None
    if "uv_groundtruth" not in df.columns:
        df["uv_groundtruth"] = None

    else:
        df["uv_clicked"] = df["uv_clicked"].astype(object)
        df["uv_groundtruth"] = df["uv_groundtruth"].astype(object)

    # load images and colmap model
    cameras0, images0, points3D0 = read_model(model0, ext=".bin")
    cameras1, images1, points3D1 = read_model(model1, ext=".bin")
    
    for idx, row in df.iterrows():
        if is_populated(row.get("uv_clicked")):
            print(f"[SKIP] Row {idx} already populated.")
            continue
        img0_name = row['img0']
        img1_name = row['img1']
        
        path_0 = Path(rgb_path0 / img0_name)
        path_1 = Path(rgb_path1 / img1_name)

        img0 = get_colmap_image_by_name(images0, img0_name)
        img1 = get_colmap_image_by_name(images1, img1_name)
        camera1 = cameras1[img1.camera_id]
        h1, w1 = camera1.height, camera1.width
    
        H, inlier_mask, pts0_H, pts1_H = compute_homography(path_0, path_1)
        
        # show images to help with kpt selection
        image0 = cv2.cvtColor(cv2.imread(str(path_0)), cv2.COLOR_BGR2RGB)
        image1 = cv2.cvtColor(cv2.imread(str(path_1)), cv2.COLOR_BGR2RGB)
        plot_kpts_on_image_pair(image0, image1, None, None, None)

        # get keypoints from the first image andlookup its corresponding 3D point
        results = show_image_with_clickable_points(path_0, img0)
        
        uv_clicked = []
        uv_groundtruth = []
        for result in results:
            u_clicked, v_clicked, pid = result['u'], result['v'], result['pid']
            if pid == -1:
                print(f"⚠️  No 3D point found for clicked point ({u_clicked:.2f}, {v_clicked:.2f}). Skipping.")
                continue

            # transform clicked point using homography
            pt = np.array([[[u_clicked, v_clicked]]], dtype=np.float32)
            u_gt, v_gt = cv2.perspectiveTransform(pt, H)[0, 0]
            
            # ignore points that transform out of image bounds because we cannot correct them
            if u_gt < 0 or u_gt >= w1 or v_gt < 0 or v_gt >= h1:
                print(f"⚠️  Transformed point ({u_gt:.2f}, {v_gt:.2f}) is out of bounds for image size ({w1}, {h1}). Skipping.")
                continue
            
            uv_clicked.append([u_clicked, v_clicked])
            uv_groundtruth.append([float(u_gt), float(v_gt)])

        if not uv_clicked:
            print(f"[SKIP] No valid points clicked for row {idx}. Continuing.")
            continue

        # fig, axs = plot_warped_image(H, inlier_mask, pts0_H, pts1_H, str(path_0), str(path_1), fig=None, ax=plt.gca(), uv_projected=None, uv_groundtruth=np.array(uv_groundtruth))
    
        uv_gt_np = np.array(uv_groundtruth, dtype=np.float32)
    
        # manually correct groundtruth
        _, axs = plot_kpts_on_image_pair(image0, image1, uv_clicked, None, None)
        editor = PointEditor(axs[1], uv_gt_np)
        plt.show()
    
        uv_groundtruth = uv_gt_np.tolist()
        
        df.loc[idx, "uv_clicked"] = json.dumps(uv_clicked)
        df.loc[idx, "uv_groundtruth"] = json.dumps(uv_groundtruth)
        
        print(f"[✅] Updated row {idx}: {img0_name} -> {img1_name}")
    
    df.to_csv(csv_file, index=False)
    print(f"\n[💾] Updated CSV saved: {csv_file}")