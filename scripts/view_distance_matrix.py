import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="View a distance matrix.")
    parser.add_argument("matrix_path", type=str, help="Path to D.npy distance matrix")
    
    args = parser.parse_args()
    
    matrix_path = args.matrix_path

    D = np.load(matrix_path)

    fig_main, ax_main = plt.subplots(figsize=(8, 6))
    im = ax_main.imshow(D, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax_main, label="Distance")
    ax_main.set_xlabel("Query")
    ax_main.set_ylabel("Database")
    ax_main.set_title("Distance Matrix")
    ax_main.xaxis.set_ticks_position('top')
    ax_main.xaxis.set_label_position('top')
    
    plt.show()

if __name__ == "__main__":
    main()