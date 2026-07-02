import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from utilities import parse_yaml

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")
    parser.add_argument("--max_rgb", type=int, default=3000, help="Maximum number of RGB images to consider.")
    # parser.add_argument("--threshold", type=float, help="Threshold between 0 and 1")
    
    args = parser.parse_args()
    
    _, _, _, log_dir, dist_thresh = parse_yaml(args.exp_yaml)
    matrix_path = f"{log_dir}/D_cleaned.npy"
    
    max_rgb = args.max_rgb

    # dist_thresh = args.threshold
    
    if not Path(matrix_path).exists():
        print(f"ERROR: Matrix file {matrix_path} does not exist.")
        return
    
    # Load matrix
    D = np.load(matrix_path)
    D_thresholded = D.copy()
    
    # Set values above threshold to infinity
    D_thresholded[D_thresholded > dist_thresh] = np.inf
    
    # Set remaining finite values to 1 (indicating pairs below threshold)
    D_binary = D_thresholded.copy()
    D_binary[np.isfinite(D_binary)] = 1
    
    # Count pairs below threshold
    count = int(np.sum(D_thresholded <= dist_thresh))
    if max_rgb is not None and count > max_rgb:
        print(f"WARNING: Found {count} pairs below threshold but max_rgb is set to {max_rgb}. Capping count to {max_rgb} due to subsampling.")
        count = int(max_rgb)
    
    # Estimate the time taken to do feature matching for all pairs below threshold
    # Assuming 0.1 seconds per pair (this is just an example, actual time may vary based on hardware and implementation)
    time_per_pair = 0.05
    total_time_seconds = (count/2) * time_per_pair
    total_time_minutes = total_time_seconds / 3600

    print(f"Matrix shape: {D.shape}")
    print(f"Threshold: {dist_thresh}")
    print(f"Pairs with distance <= threshold (accounting for symmetry): {count/2}")
    print(f"Estimated time for feature matching: {total_time_minutes:.2f} hours")
    
    # Save binary matrix
    # np.save(f"{Path(matrix_path).parent}/D_binary_{dist_thresh}.npy", D_binary)
    np.save(f"{Path(matrix_path).parent}/D_binary.npy", D_binary)

    # Visualize matrix
    fig_main, ax_main = plt.subplots(figsize=(8, 6))
    im = ax_main.imshow(D_binary, cmap="gray", aspect="auto")
    # plt.colorbar(im, ax=ax_main, label="Distance")
    ax_main.set_xlabel("Query")
    ax_main.set_ylabel("Database")
    ax_main.set_title("Binary Distance Matrix")
    ax_main.xaxis.set_ticks_position('top')
    ax_main.xaxis.set_label_position('top')
    
    plt.show()

if __name__ == "__main__":
    main()