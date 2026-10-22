"""
Re-click existing annotations for the inter-annotator consistency study. The query points (uv_clicked) of the original
annotation stay fixed; for each one, mark the same physical point in the other image. Each point opens a zoomed
side-by-side view (left: the fixed query point; right: the starting guess, homography transfer refined onto the
epipolar line (yellow dashed), exactly as in the original annotation):

  left click   move the point          scroll   zoom both views
  enter        accept                  escape   cannot identify this point (recorded as NaN)
  q / close    save and quit (run the same command again to continue where you stopped)

Needs only numpy, pandas, matplotlib and OpenCV (pixi env `annotate`, also on macOS) and a kit written by
eval/precompute_click_init.py. Writes <kit>/annotator_<ID>/evaluation_points_<years>.csv with the original columns
(img0, img1, uv_clicked, uv_groundtruth), saved after every point; never touches the original annotation files.

usage: python eval/reclick_eval_points.py --kit <kit folder> --annotator <ID>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.click_gui import ZoomedPointRefiner  # noqa: E402

NOT_IDENTIFIED = [float("nan"), float("nan")]


def load_progress(path: Path) -> dict[tuple[str, str], list]:
    """(img0, img1) -> database points clicked so far, from an earlier session's output csv."""
    if not path.is_file():
        return {}
    return {(r.img0, r.img1): json.loads(r.uv_groundtruth) for r in pd.read_csv(path).itertuples()}


def save(path: Path, rows: list[dict], progress: dict) -> None:
    out = [{"img0": r["img0"], "img1": r["img1"], "uv_clicked": json.dumps(r["uv_clicked"]),
            "uv_groundtruth": json.dumps(progress.get((r["img0"], r["img1"]), []))} for r in rows]
    pd.DataFrame(out).to_csv(path, index=False)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", type=Path, required=True, help="kit folder (init.json, images/)")
    parser.add_argument("--annotator", type=str, required=True, help="annotator ID, e.g. B")
    parser.add_argument("--zoom", type=float, default=200, help="initial half-width of the zoomed view (px)")
    args = parser.parse_args()

    kit = json.loads((args.kit / "init.json").read_text())
    out_dir = args.kit / f"annotator_{args.annotator}"
    out_dir.mkdir(exist_ok=True)
    total = sum(len(r["uv_clicked"]) for rows in kit["csvs"].values() for r in rows)
    done = 0

    for csv_name, rows in kit["csvs"].items():
        out_csv = out_dir / csv_name
        progress = load_progress(out_csv)
        for pair_index, row in enumerate(rows, 1):
            key = (row["img0"], row["img1"])
            clicked = progress.setdefault(key, [])
            done += len(clicked)
            if len(clicked) == len(row["uv_clicked"]):
                continue
            image0, image1 = (read_rgb(args.kit / "images" / row[k]) for k in ("img0", "img1"))
            h1, w1 = image1.shape[:2]
            H = np.array(row["H"], dtype=np.float64)
            F = None if row["F"] is None else np.array(row["F"], dtype=np.float64)
            for point_index in range(len(clicked), len(row["uv_clicked"])):
                u0, v0 = row["uv_clicked"][point_index]
                title = (f"annotator {args.annotator} | {csv_name} | pair {pair_index}/{len(rows)} | "
                         f"point {point_index + 1}/{len(row['uv_clicked'])} | {done}/{total} done")
                refiner = ZoomedPointRefiner(image0, image1, u0, v0, H, F, zoom=args.zoom, title=title)
                if refiner.quit:
                    save(out_csv, rows, progress)
                    print(f"saved {done}/{total} points to {out_dir}; run the same command to continue")
                    return
                point = refiner.accepted
                inside = point is not None and 0 <= point[0] < w1 and 0 <= point[1] < h1
                clicked.append([float(point[0]), float(point[1])] if inside else NOT_IDENTIFIED)
                done += 1
                save(out_csv, rows, progress)
        save(out_csv, rows, progress)
    skipped = sum(np.isnan(p[0]) for prog in (load_progress(out_dir / c) for c in kit["csvs"]) for pts in prog.values()
                  for p in pts)
    print(f"all {total} points done ({skipped} marked as not identifiable); results in {out_dir}")


if __name__ == "__main__":
    main()
