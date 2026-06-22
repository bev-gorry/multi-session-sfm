import os
import sys
import cv2
import torch
import numpy as np
# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path
# from math import sqrt, radians
# from transforms3d.euler import euler2quat

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from baselines.VSLAM_LAB.Baselines.LightGlue.lightglue import LightGlue, SuperPoint, SIFT
from baselines.VSLAM_LAB.Baselines.LightGlue.lightglue.utils import load_image, rbd
from experiment import load_exp_yaml

BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]
RED = [214, 39, 40]
GREEN = [113, 191, 110]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
grey   = [c / 255.0 for c in GREY]
red    = [c / 255.0 for c in RED]
green  = [c / 255.0 for c in GREEN]

def parse_yaml(yaml_file):
    args = load_exp_yaml(yaml_file)
    
    exp_name = args['exp_name']
    
    dataset = args['dataset']
    subset = args['subset']
    
    log_dir = args['log_dir']
    
    dist_threshold = args['dist_threshold']
    
    return exp_name, dataset, subset, log_dir, dist_threshold

def get_colmap_image_by_name(images_dict, name):
    for img in images_dict.values():
        if os.path.basename(img.name) == name:
            return img
    return None

def show_image_with_clickable_points(img_path, img_colmap):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xys = img_colmap.xys  # shape (N, 2)
    point_ids = img_colmap.point3D_ids  # shape (N,)
    
    # exclude keypoints with no 3D point
    valid_mask = point_ids != -1
    xys = xys[valid_mask]
    point_ids = point_ids[valid_mask]
    
    results = []

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.scatter(xys[:, 0], xys[:, 1], s=10, c='cyan')

    def onclick(event):
        if event.inaxes != ax:
            return

        click_point = np.array([event.xdata, event.ydata])

        # compute nearest 2D keypoint
        dists = np.linalg.norm(xys - click_point, axis=1)
        idx = np.argmin(dists)

        u, v = xys[idx]
        pid = point_ids[idx]
        
        result = {"u": None, "v": None, "pid": None}
        
        result["u"] = float(u)
        result["v"] = float(v)
        result["pid"] = int(pid)
        
        results.append(result)
        
        ax.scatter([u], [v], c='red', s=150, marker='X', linewidths=1, edgecolors='white')
        fig.canvas.draw()
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return results


def project_colmap_point(xyz_world, image, camera):
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)

    xyz_cam = R @ xyz_world.reshape(3, 1) + t
    X, Y, Z = xyz_cam.flatten()

    if Z <= 0:
        return None

    params = camera.params

    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]

    elif camera.model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]

        # normalized coordinates
        x = X / Z
        y = Y / Z

        r2 = x*x + y*y
        r4 = r2 * r2

        # radial distortion
        x_dist = x * (1 + k1*r2)
        y_dist = y * (1 + k1*r2)

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        return u, v

    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params

        # normalized coordinates
        x = X / Z
        y = Y / Z

        r2 = x*x + y*y
        r4 = r2 * r2

        # radial + tangential distortion
        x_dist = x * (1 + k1*r2 + k2*r4) + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * (1 + k1*r2 + k2*r4) + p1*(r2 + 2*y*y) + 2*p2*x*y

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        return u, v

    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported")

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v

def plot_kpts_on_image_pair(image0, image1, uv_clicked, uv_gt=None, uv_proj=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].imshow(image0)
    axs[0].axis('off')
    if uv_clicked is not None:
        axs[0].scatter(
            [u for u, v in uv_clicked],
            [v for u, v in uv_clicked],
            color=yellow, s=150, edgecolors='white'
        )
    axs[1].imshow(image1)
    axs[1].axis('off')
    
    # if uv_gt is not None and uv_proj is not None:
    #     if len(uv_gt) != len(uv_proj):
    #         return fig, axs


    #     for (u_gt, v_gt), (u_pr, v_pr) in zip(uv_gt, uv_proj):
    #         axs[1].plot(
    #         [u_gt, u_pr],
    #         [v_gt, v_pr],
    #         color=red,
    #         linewidth=3,
    #         alpha=1.0,
    #         zorder=1,
    #         )
    #         # axs[1].arrow(
    #         # u_gt, v_gt,
    #         # u_pr - u_gt, v_pr - v_gt,
    #         # color=red,
    #         # width=0.5,
    #         # head_width=6,
    #         # alpha=0.8,
    #         # length_includes_head=True
    #         # )
    
    if uv_gt is not None:
        axs[1].scatter(
            [u for u, v in uv_gt],
            [v for u, v in uv_gt],
            zorder=4,
            color=yellow, s=150, edgecolors='white'
        )
    if uv_proj is not None:
        axs[1].scatter(
            [u for u, v in uv_proj],
            [v for u, v in uv_proj],
            zorder=4,
            color=blue, s=150, marker='X', linewidths=1, edgecolors='white',
        )
        

    # plt.show()
    return fig, axs

def plot_rpe_hist(errors, color, label=''):
    mean = errors.mean()
    median = np.median(errors)

    plt.hist(errors, bins=50, alpha=0.7, color=color, label=label)
    plt.axvline(mean, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean:.2f}")
    plt.axvline(median, color="green", linestyle="--", linewidth=1, label=f"Median: {median:.2f}")
    plt.xlabel("Reprojection Error (pixels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    if label:
        plt.legend()
    return plt

# ---------- ---------- utilities for feature matching ---------- ---------- #


def unrotate_kps_W(kps_rot, k, H, W):
    # Ensure inputs are Numpy
    if hasattr(kps_rot, 'cpu'): kps_rot = kps_rot.cpu().numpy()
    if hasattr(k, 'cpu'): k = k.cpu().numpy()
    
    # Squeeze if necessary
    if k.ndim > 1: k = k.squeeze()
    if kps_rot.ndim > 2: kps_rot = kps_rot.squeeze()

    x_r = kps_rot[:, 0]
    y_r = kps_rot[:, 1]
    
    x = np.zeros_like(x_r)
    y = np.zeros_like(y_r)
    
    mask0 = (k == 0)
    x[mask0], y[mask0] = x_r[mask0], y_r[mask0]
    
    mask1 = (k == 1)
    x[mask1], y[mask1] = (W - 1) - y_r[mask1], x_r[mask1]
    
    mask2 = (k == 2)
    x[mask2], y[mask2] = (W - 1) - x_r[mask2], (H - 1) - y_r[mask2]
    
    mask3 = (k == 3)
    x[mask3], y[mask3] = y_r[mask3], (H - 1) - x_r[mask3]
    
    return np.stack([x, y], axis=-1)

def extract_keypoints(path_to_image0, features='superpoint', rotations = [0,1,2,3]):
    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    timg = load_image(path_to_image0).to(device)
    _, h, w = timg.shape

    if features == 'sift':
        extractor = SIFT(max_num_keypoints=2048).eval().to(device)
        feats = extractor.extract(timg)
        return feats , h, w
    
    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # --- Extract local features ---
    feats = {}
    for k in (rotations):
        timg_rotated = torch.rot90(timg, k, dims=(1, 2))
        feats[k] = extractor.extract(timg_rotated)
        #print(f"Extracted {feats[k]['keypoints'].shape[1]} keypoints for rotation {k}")

    # --- Merge features back to original coordinate system ---
    all_keypoints = []
    all_scores = []
    all_descriptors = []
    all_rotations = []
    for k, feat in feats.items():
        kpts = feat['keypoints']  # Shape (1, N, 2)
        num_kpts = kpts.shape[1]
        
        rot_indices = torch.full((1, num_kpts), k, dtype=torch.long, device=device)
        all_keypoints.append(feat['keypoints'])
        all_scores.append(feat['keypoint_scores'])
        all_descriptors.append(feat['descriptors'])
        all_rotations.append(rot_indices)

    # Concatenate all features along the keypoint dimension (dim=1)
    feats_merged = {
        'keypoints': torch.cat(all_keypoints, dim=1),
        'keypoint_scores': torch.cat(all_scores, dim=1),
        'descriptors': torch.cat(all_descriptors, dim=1),
        'rotations': torch.cat(all_rotations, dim=1)
    }
    
    num_kpts = feats_merged['keypoints'].shape[1]

    return feats_merged , feats, h, w

def lightglue_matching(feats0, feats1, matcher = None):
    if matcher is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matcher = LightGlue(features='superpoint').eval().to(device)
    
    out_k = matcher({'image0': feats0, 'image1': feats1})
    _, _, out_k = [rbd(x) for x in [feats0, feats1, out_k]]   # remove batch dim
    return out_k['matches']

def feature_matching(feats0, feats1, matcher = None, exhaustive = True):
    best_rot = 0
    best_num_matches = 0
    matches_tensor = None
 
    # Find the best rotation alignment
    for rot in [0,1,2,3]:
        matches_tensor_rot = lightglue_matching(feats0[0], feats1[rot], matcher = matcher)
        if (len(matches_tensor_rot) > best_num_matches):
            best_num_matches = len(matches_tensor_rot)
            best_rot = rot
            matches_tensor = matches_tensor_rot

    if matches_tensor is not None and len(matches_tensor) > 0:
        matches_np = matches_tensor.cpu().numpy().astype(np.uint32)
    else:
        return None

    # Adjust matches to account for rotations
    for k in range(best_rot):
        matches_np[:,1] += feats1[k]['keypoints'].shape[1]
    all_matches = [matches_np]  

    if not exhaustive:
        return matches_np
    
    # Find the other rotation combinations
    rots = []
    for rot in [1, 2, 3]:
        rot_i = best_rot + rot
        if rot_i >=4:
            rot_i = rot_i -4
        rots.append(rot_i)

    # Compute matches for the other rotation combinations
    for rot_i in [1,2,3]:
        rot_j = rots[rot_i-1]

        matches_tensor_rot = lightglue_matching(feats0[rot_i], feats1[rot_j], matcher = matcher)
        matches_np_i = matches_tensor_rot.cpu().numpy().astype(np.uint32)
        if rot_i > 0:
            for k in range(rot_i):
                matches_np_i[:,0] += feats0[k]['keypoints'].shape[1]
        if rot_j > 0:
            for k in range(rot_j):
                matches_np_i[:,1] += feats1[k]['keypoints'].shape[1]

        all_matches.append(matches_np_i)
        # print(f"Rotation {rot_i} vs {rot_j}: {len(matches_tensor_rot)} matches")

    # Stack all matches together
    matches_stacked = (
        np.vstack(all_matches) if len(all_matches) and all_matches[0].size else
        np.empty((0, 2), dtype=np.uint32)
    )
    
    # if best_rot > 0:
    #     for k in range(best_rot):
    #         print(f"Adjusting for rotation {k}")
    #         matches_np[:,1] += feats1[k]['keypoints'].shape[1]

    # return matches_np
    return matches_stacked
