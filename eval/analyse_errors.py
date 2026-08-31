import numpy as np
import argparse
from pathlib import Path


def load_errors(file_list):
    """Load and concatenate reprojection errors from multiple npy files."""
    all_errors = []
    per_sequence = {}

    for f in file_list:
        errors = np.load(f)
        errors = errors[np.isfinite(errors)]  # remove NaNs/Infs if any
        per_sequence[f.stem] = errors
        all_errors.append(errors)

    pooled = np.concatenate(all_errors)
    return pooled, per_sequence


def compute_statistics(errors, threshold):
    median = np.median(errors)
    pct_above = 100.0 * np.sum(errors > threshold) / len(errors)
    return median, pct_above


def summarize_method(name, file_list, threshold):
    pooled, per_seq = load_errors(file_list)

    # median, pct_above = compute_statistics(pooled, threshold)

    # print(f"\n--- {name} ---")
    # print(f"Pooled median reprojection error: {median:.2f} px")
    # print(f"% correspondences above {threshold}px: {pct_above:.2f}%")

    # print("\nPer-sequence medians:")
    # for seq, errs in per_seq.items():
    #     print(f"  {seq}: {np.median(errs):.2f} px")

    # return median, pct_above
    median, pct_above = compute_statistics(pooled, threshold)

    seq_medians = {seq: np.median(errs) for seq, errs in per_seq.items()}

    print("\nPer-sequence medians:")
    for seq, med in seq_medians.items():
        print(f"  {seq}: {med:.2f} px")

    average_median = np.mean(list(seq_medians.values()))
    print(f"\nAverage of sequence medians: {average_median:.2f} px")

    return median, pct_above, average_median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Pixel threshold for error statistics")

    args = parser.parse_args()
    
    icp_dir = Path("/home/beverley/Repos/multi-session-sfm/reprojection/COMBINED/icp")
    buffer_dir = Path("/home/beverley/Repos/multi-session-sfm/reprojection/COMBINED/buffer")
    ours_dir = Path("/home/beverley/Repos/multi-session-sfm/reprojection/COMBINED/ours")

    icp_files = sorted(Path(icp_dir).glob("*.npy"))
    buffer_files = sorted(Path(buffer_dir).glob("*.npy"))
    ours_files = sorted(Path(ours_dir).glob("*.npy"))

    icp_median, icp_pct, icp_avg = summarize_method(
        "Post-hoc ICP", icp_files, args.threshold
    )

    buffer_median, buffer_pct, buffer_avg = summarize_method(
        "Buffer Reconstruction", buffer_files, args.threshold
    )

    ours_median, ours_pct, ours_avg = summarize_method(
        "Ours Reconstruction", ours_files, args.threshold
    )

    icp_improvement = 100.0 * (icp_avg - ours_avg) / icp_avg
    buffer_improvement = 100.0 * (buffer_avg - ours_avg) / buffer_avg

    print("\n=== Average of Per-Sequence Medians ===")
    print(f"ICP:      {icp_avg:.2f} px")
    print(f"BUFFER-X: {buffer_avg:.2f} px")
    print(f"Ours:     {ours_avg:.2f} px")

    print("\n=== Relative Improvement ===")
    print(f"vs ICP:      {icp_improvement:.2f}%")
    print(f"vs BUFFER-X: {buffer_improvement:.2f}%")

    print("\n=== Numbers for your paper ===")
    print(f"YY.Y  -> threshold: {args.threshold}")
    print(f"ZZ    -> ICP % above threshold: {icp_pct:.2f}%")
    print(f"DD.D  -> Buffer % above threshold: {buffer_pct:.2f}%")
    print(f"AA.A  -> Ours median: {ours_median:.2f} px")


if __name__ == "__main__":
    main()