"""Reprojection-error evaluation for the port tests.

Same maths as eval/project_eval_points.py: each annotated click in img0 -> its keypoint -> that keypoint's 3D point
-> projected into img1 (OPENCV camera model) -> pixel distance to the annotated ground truth in img1.
Reads the COLMAP text model (cameras.txt, images.txt, points3D.txt), so it works for COLMAP 3.x and 4.x outputs.

How a click is turned into a 3D point (--match):
  exact_last   the original rule until 2026-07-23: the LAST keypoint exactly at the click; skipped if it has no 3D point
  exact_first  the rule from 2026-07-23: the first keypoint exactly at the click that has a 3D point
  nearest      the nearest keypoint that has a 3D point, if within --radius px (default 1.0; needed when the SIFT version
               or the images change)
  neighbourhood  feature-agnostic: the clicks sit on SIFT/SuperPoint keypoints of the annotation model, so a model with
               other features cannot score them by keypoint. Instead the click is back-projected to the median depth of
               every 3D point observed near it (at least 3), whatever its feature. "Near" = within RADIUS_FRACTION of the
               query image width (~2.5%: exactly 15 px for Sesoko's 608 px images), or --radius px. On s05 the ranking of the
               feature/matcher runs is the same at 10, 15 and 20 px, and 15 px scores nearly every click of the
               learned-feature models; check the same range on any new dataset before fixing the radius

Annotations are in the pixel coordinates of the images they were clicked on (ANNOTATION_WIDTHS). When a model's
images are wider or narrower (Sesoko was regenerated at 608 px instead of 607 px, from the same raw frames), x is
rescaled to the model's image width first; checked on every s05 click, the residual is 0.08 px.

Prints n (points scored out of all clicks), median and mean error, and why the other clicks were not scored.

Example:
  python evaluate_rpe.py --model <exp>/colmap_00000 \
      --csv reprojection/SESOKO/sskall-s05/evaluation_points_2016-2017.csv ...
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

# Width (px) of the images each dataset's annotations were clicked on, keyed by the dataset folder in the csv path
ANNOTATION_WIDTHS = {"SESOKO": 607}
# --match neighbourhood: 3D points needed within the radius for a robust median depth
MIN_NEIGHBOURS = 3
RADIUS_FRACTION = 15 / 608  # neighbourhood radius as a fraction of the query image width: 15 px for Sesoko (~2.5%)
NEAREST_RADIUS = 1.0     # px
NOT_ANNOTATED = "not_annotated"  # NaN ground truth: the annotator marked the point as not identifiable


def read_cameras(path):
    cameras = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cameras[int(t[0])] = (t[1], int(t[2]), np.array([float(x) for x in t[4:]]))
    return cameras


def to_image_width(uv, annotation_width, width):
    """Annotated (u, v) -> the same points in an image `width` px wide; the frames are the same raw images resized,
    so only x stretches (pixel-centre convention). No change when the widths match or the dataset is not listed."""
    if annotation_width is None or annotation_width == width:
        return uv
    uv = uv.copy()
    uv[:, 0] = (uv[:, 0] + 0.5) * width / annotation_width - 0.5
    return uv


def read_images(path):
    """image file name -> (R, t, camera_id, keypoints (N, 2), point3D ids (N,))"""
    lines = [line for line in open(path) if not line.startswith("#")]
    images = {}
    for i in range(0, len(lines) - 1, 2):
        t = lines[i].split()
        if not t:
            continue
        qw, qx, qy, qz = map(float, t[1:5])
        R = np.array([[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
                      [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
                      [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy]])
        p = lines[i + 1].split()
        xys = np.array([[float(p[k]), float(p[k + 1])] for k in range(0, len(p), 3)]).reshape(-1, 2)
        pids = np.array([int(p[k + 2]) for k in range(0, len(p), 3)], dtype=np.int64)
        images[os.path.basename(t[9])] = (R, np.array(list(map(float, t[5:8]))), int(t[8]), xys, pids)
    return images


def read_points3D(path):
    points = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        points[int(t[0])] = np.array(list(map(float, t[1:4])))
    return points


def project(xyz, R, t, camera):
    """utilities.project_colmap_point for the OPENCV model; None if the point is behind the camera."""
    model, _, params = camera
    if model != "OPENCV":
        raise NotImplementedError(f"camera model {model}")
    X, Y, Z = R @ xyz + t
    if Z <= 0:
        return None
    fx, fy, cx, cy, k1, k2, p1, p2 = params
    x, y = X / Z, Y / Z
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def undistort(u, v, camera, iterations=20):
    """Pixel -> normalised image coordinates (x, y) for the OPENCV model: the inverse of project()'s distortion by
    fixed-point iteration, as in COLMAP."""
    _, _, params = camera
    fx, fy, cx, cy, k1, k2, p1, p2 = params
    xd, yd = (u - cx) / fx, (v - cy) / fy
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial
        y = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
    return x, y


def neighbours(click, xys, pids, radius):
    """Mask of the observations with a 3D point within `radius` px of the click."""
    return (np.hypot(xys[:, 0] - click[0], xys[:, 1] - click[1]) <= radius) & (pids != -1)


def click_to_xyz(click, image, camera, points, radius):
    """Feature-agnostic lookup: the click's 3D position from the model's points observed within `radius` px of it in
    this image, whatever feature they come from. Depth = median camera-frame depth of those points; the click is
    back-projected to that depth. Returns (world xyz or None, reason it was not found)."""
    R, t, _, xys, pids = image
    near = neighbours(click, xys, pids, radius)
    if near.sum() < MIN_NEIGHBOURS:
        return None, f"fewer_than_{MIN_NEIGHBOURS}_points_within_radius"
    depth = np.median([(R @ points[pid] + t)[2] for pid in np.unique(pids[near])])
    if depth <= 0:
        return None, "behind_camera"
    x, y = undistort(click[0], click[1], camera)
    return R.T @ (depth * np.array([x, y, 1.0]) - t), ""


def click_to_point3D(click, xys, pids, match, radius):
    """Returns (point3D id or None, reason it was not found)."""
    dist = np.hypot(xys[:, 0] - click[0], xys[:, 1] - click[1])
    if match == "nearest":
        dist = np.where(pids != -1, dist, np.inf)
        k = int(np.argmin(dist))
        return (pids[k], "") if dist[k] <= radius else (None, "no_keypoint_within_radius")
    exact = np.where(dist == 0)[0]
    if len(exact) == 0:
        return None, "no_exact_keypoint"
    if match == "exact_last":
        return (pids[exact[-1]], "") if pids[exact[-1]] != -1 else (None, "keypoint_without_3D_point")
    with_point = [k for k in exact if pids[k] != -1]
    return (pids[with_point[0]], "") if with_point else (None, "keypoint_without_3D_point")


def read_model(model):
    """(cameras, images, points) of a COLMAP text model folder; pass it to score_clicks to score it several ways."""
    return (read_cameras(os.path.join(model, "cameras.txt")), read_images(os.path.join(model, "images.txt")),
            read_points3D(os.path.join(model, "points3D.txt")))


def radius_px(match, radius, width):
    """The radius in px: `radius` if given, else NEAREST_RADIUS or RADIUS_FRACTION of the query image width."""
    if radius is not None:
        return radius
    return NEAREST_RADIUS if match == "nearest" else RADIUS_FRACTION * width


def project_clicks(model, csvs, match="neighbourhood", radius=None):
    """Every annotated click of `csvs` projected with `model` (a COLMAP text model folder, or read_model's result).
    Returns ([one dict per click: img0, img1, click and truth (in the model's pixels), radius, uv (projection or None),
    error (px or None), reason (why not scored, "" if scored)], {(annotation width, model width) rescaled})."""
    cameras, images, points = read_model(model) if isinstance(model, (str, os.PathLike)) else model
    records, rescaled = [], set()
    for csv in csvs:
        annotation_width = next((w for name, w in ANNOTATION_WIDTHS.items() if name in os.path.abspath(csv).split(os.sep)), None)
        for _, row in pd.read_csv(csv).iterrows():
            uv_clicked = np.array(json.loads(row["uv_clicked"]), dtype=np.float32).reshape(-1, 2)
            uv_truth = np.array(json.loads(row["uv_groundtruth"]), dtype=np.float32).reshape(-1, 2)
            pair = {"img0": row["img0"], "img1": row["img1"]}
            if row["img0"] not in images or row["img1"] not in images:
                records += [{**pair, "click": c, "truth": g, "radius": None, "uv": None, "error": None,
                             "reason": "image_not_registered" if np.all(np.isfinite(g)) else NOT_ANNOTATED}
                            for c, g in zip(uv_clicked, uv_truth)]
                continue
            image0 = images[row["img0"]]
            _, _, camera0, xys, pids = image0
            R1, t1, camera1, _, _ = images[row["img1"]]
            width0, width1 = cameras[camera0][1], cameras[camera1][1]
            uv_clicked = to_image_width(uv_clicked, annotation_width, width0)
            uv_truth = to_image_width(uv_truth, annotation_width, width1)
            rescaled |= {(annotation_width, w) for w in (width0, width1) if annotation_width not in (None, w)}
            r = radius_px(match, radius, width0)
            for click, truth in zip(uv_clicked, uv_truth):
                if not np.all(np.isfinite(truth)):  # re-click study: the annotator could not identify this point
                    records.append({**pair, "click": click, "truth": truth, "radius": r, "uv": None, "error": None,
                                    "reason": NOT_ANNOTATED})
                    continue
                if match == "neighbourhood":
                    xyz, reason = click_to_xyz(click, image0, cameras[camera0], points, r)
                else:
                    pid, reason = click_to_point3D(click, xys, pids, match, r)
                    xyz = None if pid is None else points[pid]
                uv = None if xyz is None else project(xyz, R1, t1, cameras[camera1])
                if xyz is not None and uv is None:
                    reason = "behind_camera"
                error = None if uv is None else float(np.hypot(uv[0] - truth[0], uv[1] - truth[1]))
                records.append({**pair, "click": click, "truth": truth, "radius": r, "uv": uv, "error": error,
                                "reason": reason})
    return records, rescaled


def score_clicks(model, csvs, match="neighbourhood", radius=None):
    """Reprojection error of every annotated click (see project_clicks).
    Returns (errors (n,), {reason: clicks not scored}, total clicks, {(annotation width, model width) rescaled}).
    Points an annotator could not identify (NaN ground truth) are not part of the total."""
    records, rescaled = project_clicks(model, csvs, match, radius)
    records = [r for r in records if r["reason"] != NOT_ANNOTATED]
    errors = np.array([r["error"] for r in records if r["error"] is not None])
    skipped = {}
    for r in records:
        if r["error"] is None:
            skipped[r["reason"]] = skipped.get(r["reason"], 0) + 1
    return errors, skipped, len(records), rescaled


def main():
    parser = argparse.ArgumentParser(description="Reprojection error of annotated cross-session clicks")
    parser.add_argument("--model", required=True, help="folder with cameras.txt, images.txt, points3D.txt")
    parser.add_argument("--csv", required=True, nargs="+", help="evaluation_points_<year pair>.csv files")
    parser.add_argument("--match", default="neighbourhood",
                        choices=["exact_last", "exact_first", "nearest", "neighbourhood"])
    # nearest, 1.0 px: on re-extracted SIFT keypoints, radii up to 1 px add clicks without raising the median (s05:
    # 0.5 px 90/145 2.48 px, 1.0 px 105/145 2.44 px, 1.5 px 116/145 2.61 px), so the extra matches are the same points
    parser.add_argument("--radius", type=float, default=None,
                        help=f"px, for --match nearest (default {NEAREST_RADIUS}) / neighbourhood (default "
                             f"{RADIUS_FRACTION * 100:.1f}%% of the query image width)")  # %% for argparse
    parser.add_argument("--save", default=None, help="optional .npy path for the per-point errors")
    args = parser.parse_args()

    loaded = read_model(args.model)
    errors, skipped, clicks, rescaled = score_clicks(loaded, args.csv, args.match, args.radius)
    _, images, points = loaded
    print(f"model: {args.model}")
    print(f"  registered images: {len(images)}, 3D points: {len(points)}")
    radius = (f"{args.radius} px" if args.radius is not None else
              f"{NEAREST_RADIUS} px" if args.match == "nearest" else f"{RADIUS_FRACTION:.1%} of the image width")
    print(f"  match rule: {args.match}" + (f" (radius {radius})" if args.match in ("nearest", "neighbourhood") else ""))
    print(f"  annotation x rescaled: {', '.join(f'{a} -> {w} px' for a, w in sorted(rescaled)) or 'no (widths match)'}")
    print(f"  scored: {errors.size} / {clicks} clicks   not scored: {skipped or 'none'}")
    if errors.size:
        print(f"  median: {np.median(errors):.2f} px   mean: {errors.mean():.2f} px")
    if args.save:
        np.save(args.save, errors)


if __name__ == "__main__":
    main()
