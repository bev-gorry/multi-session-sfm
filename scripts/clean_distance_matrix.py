import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from utilities import parse_yaml

def main():
    parser = argparse.ArgumentParser(description="Clean distance matrix by setting self-distances and same-sequence distances to infinity.")
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml", help="Path to experiment YAML file.")

    args = parser.parse_args()
    
    _, _, _, log_dir, _ = parse_yaml(args.exp_yaml)
    matrix_path = f"{log_dir}/D.npy"
    rgb_csv = f"{log_dir}/rgb.csv"
    
    # Remove other distance matrices
    matrix_dir = Path(matrix_path).parent
    for file in matrix_dir.glob("D_*.npy"):
            file.unlink()
    
    # Load matrix
    D = np.load(matrix_path)
    
    max_distance = np.max(D)
    min_distance = np.min(D)

    # Normalize to [0, 1]
    D = (D - min_distance) / (max_distance - min_distance)

    D_cleaned = D.copy()
    
    # Set diagonal to infinity to ignore self-distances
    D_cleaned[np.eye(D_cleaned.shape[0], dtype=bool)] = np.inf
    
    # Set values from the same sequence to infinity (assuming sequences are contiguous blocks in the matrix)
    rgb_df = pd.read_csv(rgb_csv)
    for i in tqdm(range(D_cleaned.shape[0]), desc="Filtering distance matrix"):
        query_seq = rgb_df.iloc[i]['sequence_name']
        for j in range(D_cleaned.shape[1]):
            db_seq = rgb_df.iloc[j]['sequence_name']
            if query_seq == db_seq:
                D_cleaned[i, j] = np.inf
    
    np.save(f"{Path(matrix_path).parent}/D_cleaned.npy", D_cleaned)
    
    # Visualize matrix
    fig_main, ax_main = plt.subplots(figsize=(8, 6))
    im = ax_main.imshow(D_cleaned, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax_main, label="Distance")
    ax_main.set_xlabel("Query")
    ax_main.set_ylabel("Database")
    ax_main.set_title("Filtered Distance Matrix")
    ax_main.xaxis.set_ticks_position('top')
    ax_main.xaxis.set_label_position('top')
    
    plt.show()
    
    

if __name__ == "__main__":
    main()