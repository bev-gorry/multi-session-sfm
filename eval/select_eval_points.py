### Take image pair and combined model. For each clicked 2D point in image0, transform it to image 1 using the fundamental matrix.
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
from utilities import parse_yaml, get_colmap_image_by_name, unrotate_kps_W, extract_keypoints, feature_matching, show_image_with_clickable_points, plot_kpts_on_image_pair, get_pair_colors
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
    def __init__(self, ax, points, colors, select_radius=12):

        self.ax = ax
        self.points = points
        self.colors = colors
        self.active_idx = None
        self.select_radius = select_radius

        self.scatter = ax.scatter(
            points[:,0],
            points[:,1],
            c=colors,
            s=150,
            edgecolors="white",
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

def epipolar_line_in_window(F, u0, v0, x_min, x_max, y_min, y_max):
    """Epipolar line of (u0,v0) clipped to a viewing window. Returns two endpoints or None."""
    line = cv2.computeCorrespondEpilines(
        np.array([[[u0, v0]]], dtype=np.float32), 1, F)[0, 0]
    a, b, c = line
    pts = []
    if abs(b) > 1e-9:  # y = -(a x + c)/b at window x-edges
        for x in (x_min, x_max):
            y = -(a * x + c) / b
            if y_min <= y <= y_max:
                pts.append((x, y))
    if abs(a) > 1e-9:  # x = -(b y + c)/a at window y-edges
        for y in (y_min, y_max):
            x = -(b * y + c) / a
            if x_min <= x <= x_max:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    # take the two most distant intersection points (handles corner duplicates)
    pts = np.array(pts)
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i], pts[j]


class ZoomedPointRefiner:
    """
    Interactive refinement of one correspondence.
    Left: zoomed image0 around the clicked point (reference).
    Right: zoomed image1 around the H/F estimate, with epipolar line.
      - left click: set corrected point
      - scroll: zoom in/out (both panels)
      - enter: accept current point
      - escape: skip this point
    """
    def __init__(self, image0, image1, u0, v0, H, F, zoom=200):
        self.u0, self.v0 = u0, v0
        self.F = F
        self.zoom = zoom
        self.h1, self.w1 = image1.shape[:2]

        # H estimate + F refinement as starting point
        self.u_h, self.v_h = transfer_point_with_homography(H, u0, v0)
        if F is not None:
            self.u_gt, self.v_gt = transfer_H_refined_by_F(H, F, u0, v0)
        else:
            self.u_gt, self.v_gt = self.u_h, self.v_h
        self.accepted = None

        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))
        self.axs[0].imshow(image0)
        self.axs[1].imshow(image1)
        self.axs[0].plot(u0, v0, 'o', ms=10, mfc='none', mec='green', mew=2)
        self.axs[0].plot(u0, v0, '+', ms=14, color='green')
        self.axs[0].set_title("image0 — clicked point")

        # H estimate (cyan) and current/refined point (red)
        self.axs[1].plot(self.u_h, self.v_h, 'x', ms=10, color='blue',
                         label='H estimate')
        (self.marker,) = self.axs[1].plot(self.u_gt, self.v_gt, 'o', ms=10,
                                          mfc='none', mec='red', mew=2,
                                          label='current')
        (self.cross,) = self.axs[1].plot(self.u_gt, self.v_gt, '+', ms=14,
                                         color='red')
        self.epi_artist = None
        self.axs[1].legend(loc='upper right')
        self.axs[1].set_title("image1 — click to correct | scroll zoom | enter accept | esc skip")

        self._set_views()
        self._draw_epiline()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def _set_views(self):
        z = self.zoom
        self.axs[0].set_xlim(self.u0 - z, self.u0 + z)
        self.axs[0].set_ylim(self.v0 + z, self.v0 - z)   # inverted y for images
        self.axs[1].set_xlim(self.u_gt - z, self.u_gt + z)
        self.axs[1].set_ylim(self.v_gt + z, self.v_gt - z)

    def _draw_epiline(self):
        if self.F is None:
            return
        if self.epi_artist is not None:
            self.epi_artist.remove()
            self.epi_artist = None
        x_min, x_max = sorted(self.axs[1].get_xlim())
        y_min, y_max = sorted(self.axs[1].get_ylim())
        seg = epipolar_line_in_window(self.F, self.u0, self.v0,
                                      x_min, x_max, y_min, y_max)
        if seg is not None:
            (x1, y1), (x2, y2) = seg
            (self.epi_artist,) = self.axs[1].plot(
                [x1, x2], [y1, y2], '--', color=yellow, lw=1.5, alpha=0.8)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.axs[1] or event.button != 1:
            return
        self.u_gt, self.v_gt = event.xdata, event.ydata
        self.marker.set_data([self.u_gt], [self.v_gt])
        self.cross.set_data([self.u_gt], [self.v_gt])
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        self.zoom *= 0.8 if event.button == 'up' else 1.25
        self.zoom = float(np.clip(self.zoom, 30, max(self.h1, self.w1)))
        self._set_views()
        self._draw_epiline()

    def on_key(self, event):
        if event.key == 'enter':
            self.accepted = (float(self.u_gt), float(self.v_gt))
            plt.close(self.fig)
        elif event.key == 'escape':
            self.accepted = None
            plt.close(self.fig)

def compute_homography_and_fundamental_matrix(path_0, path_1):
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

    
    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 5.0)

    return H, F

def transfer_H_refined_by_F(H, F, u, v):
    u_h, v_h = transfer_point_with_homography(H, u, v)
    line = cv2.computeCorrespondEpilines(
        np.array([[[u, v]]], dtype=np.float32), 1, F)[0, 0]
    a, b, c = line
    t = (a * u_h + b * v_h + c) / (a * a + b * b)
    return u_h - a * t, v_h - b * t

def transfer_point_with_homography(H, u_clicked, v_clicked):
    pt0 = np.array([[[u_clicked, v_clicked]]], dtype=np.float32)
    pt1 = cv2.perspectiveTransform(pt0, H)[0, 0]
    u_gt, v_gt = pt1[0], pt1[1]
    return u_gt, v_gt

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
    
    ### IMPORTANT: change csv file name here to populate other years (2016-2017, 2018-2019, 2020-2021)
    csv_file = Path(f"{EVAL_POINTS_DIR}/{dataset}/{subset}/evaluation_points_2016-2018.csv")
    
    yellow_text = "\033[93m"
    reset_text = "\033[0m"
    print(f"{yellow_text}CSV FILE: {csv_file}{reset_text}")

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
        
        try: 
            img0_name = row['img0']
            img1_name = row['img1']
            
            path_0 = Path(rgb_path0 / img0_name)
            path_1 = Path(rgb_path1 / img1_name)

            img0 = get_colmap_image_by_name(images0, img0_name)
            img1 = get_colmap_image_by_name(images1, img1_name)
            
            if img0 is None:
                print(f"[ERROR] Could not find '{img0_name}' in COLMAP model. Skipping.")
                continue

            if img1 is None:
                print(f"[ERROR] Could not find '{img1_name}' in COLMAP model. Skipping.")
                continue

            camera1 = cameras1[img1.camera_id]
            h1, w1 = camera1.height, camera1.width
        
            H, F = compute_homography_and_fundamental_matrix(path_0, path_1)
            
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
                
                refiner = ZoomedPointRefiner(image0, image1, u_clicked, v_clicked, H, F, zoom=200)
                if refiner.accepted is None:
                    print("⚠️  Point skipped by user.")
                    continue
                u_gt, v_gt = refiner.accepted

                if not (0 <= u_gt < w1 and 0 <= v_gt < h1):
                    print(f"⚠️  Corrected point out of bounds. Skipping.")
                    continue
                
                uv_clicked.append([u_clicked, v_clicked])
                uv_groundtruth.append([float(u_gt), float(v_gt)])

            if not uv_clicked:
                print(f"[SKIP] No valid points clicked for row {idx}. Continuing.")
                continue

            # fig, axs = plot_warped_image(H, inlier_mask, pts0_H, pts1_H, str(path_0), str(path_1), fig=None, ax=plt.gca(), uv_projected=None, uv_groundtruth=np.array(uv_groundtruth))
        
            uv_gt_np = np.array(uv_groundtruth, dtype=np.float32)
        
            # manually correct groundtruth
            colors = get_pair_colors(len(uv_clicked))

            _, axs = plot_kpts_on_image_pair(
                image0,
                image1,
                uv_clicked,
                None,
                None
            )
            editor = PointEditor(axs[1], uv_gt_np, colors)
            
            plt.show()
        
            uv_groundtruth = uv_gt_np.tolist()
            
            df.loc[idx, "uv_clicked"] = json.dumps(uv_clicked)
            df.loc[idx, "uv_groundtruth"] = json.dumps(uv_groundtruth)
            
            print(f"[✅] Updated row {idx}: {img0_name} -> {img1_name}")
            df.to_csv(csv_file, index=False)
            print(f"[💾] Saved progress to {csv_file}")
        
        except Exception as e:
            print(f"Failed on row {idx}: {e}")
            df.to_csv(csv_file, index=False)
            print(f"[💾] Saved progress to {csv_file}")
            continue
        
    df.to_csv(csv_file, index=False)
    print(f"\n[💾] Updated CSV saved: {csv_file}")