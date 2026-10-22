"""
Build an annotation kit for re-clicking existing annotations (inter-annotator consistency study): another annotator
keeps the original query clicks (uv_clicked) and marks only the corresponding point in the other image, starting from
the same initial guess as the original annotation (homography transfer refined onto the epipolar line,
eval/select_eval_points.py). The kit needs no torch, LightGlue or COLMAP model, so it can be clicked on any machine:

  <out>/images/<name>      the images of the annotated pairs (copied from --image_dirs, the resolution they were
                           annotated at)
  <out>/init.json          per csv and row: img0, img1, the query clicks, H and F (SuperPoint at 4 rotations + LightGlue,
                           RANSAC 5 px, as compute_homography_and_fundamental_matrix in eval/select_eval_points.py)

Clicking: eval/reclick_eval_points.py --kit <out> --annotator <ID>.

usage (lightglue env, repo root): python eval/precompute_click_init.py --csv_dir reprojection/SESOKO/sskall-s05
    --image_dirs <dir> [<dir> ...] --out <kit folder>
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lightglue import LightGlue  # noqa: E402
from utilities import extract_keypoints, feature_matching, unrotate_kps_W  # noqa: E402


def homography_and_fundamental(path0: Path, path1: Path, matcher) -> tuple[np.ndarray, np.ndarray | None]:
    """H and F of an image pair from SuperPoint (4 rotations) + LightGlue matches, as the original annotation GUI."""
    feats0, rot_feats0, h0, w0 = extract_keypoints(path0, features="superpoint")
    feats1, rot_feats1, h1, w1 = extract_keypoints(path1, features="superpoint")
    matches = feature_matching(rot_feats0, rot_feats1, matcher=matcher, exhaustive=True)
    pts0 = unrotate_kps_W(feats0["keypoints"].squeeze(0).cpu().numpy().astype(np.float32),
                          feats0["rotations"].squeeze(0).cpu().numpy(), h0, w0)[matches[:, 0]]
    pts1 = unrotate_kps_W(feats1["keypoints"].squeeze(0).cpu().numpy().astype(np.float32),
                          feats1["rotations"].squeeze(0).cpu().numpy(), h1, w1)[matches[:, 1]]
    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 5.0)
    print(f"    {path0.name} -> {path1.name}: {len(matches)} matches")
    return H, F


def find_image(name: str, image_dirs: list[Path]) -> Path:
    hits = [d / name for d in image_dirs if (d / name).is_file()]
    if len(hits) != 1:
        raise SystemExit(f"{name}: found {len(hits)} times in {image_dirs}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_dir", type=Path, required=True, help="folder with the evaluation_points_*.csv to re-click")
    parser.add_argument("--image_dirs", type=Path, nargs="+", required=True,
                        help="folders holding the images at the resolution they were annotated at")
    parser.add_argument("--out", type=Path, required=True, help="kit folder to write")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1, flash=True).eval().to(device)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    kit = {"source_csv_dir": str(args.csv_dir), "csvs": {}}
    for csv in sorted(args.csv_dir.glob("evaluation_points_*.csv")):
        rows = []
        for _, row in pd.read_csv(csv).iterrows():
            uv_clicked = json.loads(row["uv_clicked"]) if isinstance(row["uv_clicked"], str) else []
            if not uv_clicked:
                continue
            paths = [find_image(row[k], args.image_dirs) for k in ("img0", "img1")]
            for p in paths:
                shutil.copy2(p, args.out / "images" / p.name)
            with torch.no_grad():
                H, F = homography_and_fundamental(*paths, matcher)
            rows.append({"img0": row["img0"], "img1": row["img1"], "uv_clicked": uv_clicked,
                         "H": H.tolist(), "F": None if F is None else F[:3].tolist()})
        kit["csvs"][csv.name] = rows
        print(f"{csv.name}: {len(rows)} pairs, {sum(len(r['uv_clicked']) for r in rows)} points")
    (args.out / "init.json").write_text(json.dumps(kit, indent=1))
    print(f"kit written to {args.out}")


if __name__ == "__main__":
    main()
