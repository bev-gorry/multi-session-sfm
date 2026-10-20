### Colour a COLMAP text model by the sessions that observe each 3D point (Fig. 7), with the colours and tint of
### vis/color_pointcloud_sesoko.py: seen by one session only -> that session's colour (1st BLUE, 2nd YELLOW, 3rd PINK,
### sessions sorted by name, e.g. ssk16/ssk17/ssk18); seen by two or more sessions -> GREY. Works on any text model
### (joint runs, and the post-hoc merged models posthoc/sim3, sim3_ba, icp, buffer), no experiment yaml needed.
###
### Writes <out>/points.ply and, without --session, <out>/model/ (the model with recoloured points3D.txt, for the COLMAP
### GUI). --session keeps only the points that session observes (Fig. 7 per-year columns); PLY only, because dropping
### points would leave the model's 2D observations pointing at missing 3D points.
###
### usage: python vis/color_pointcloud_sessions.py --model <model folder> --rgb_csv <benchmark>/<DATASET>/<seq>/rgb.csv
###            --out <folder> [--session ssk16]

import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

# Same colours as vis/color_pointcloud_sesoko.py and utilities.py
BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]
SESSION_COLOURS = [BLUE, YELLOW, PINK]
ALPHA_TINT = 0.4


def tint_rgb(original_rgb, tint, alpha=ALPHA_TINT):
    return np.clip((1 - alpha) * original_rgb + alpha * np.array(tint, dtype=np.float32), 0, 255).astype(np.uint8)


def image_sessions(model: Path, rgb_csv: Path) -> dict[int, str]:
    """COLMAP image id -> session name (rgb.csv's sequence_name of the image file)."""
    session_of = {Path(p).name: s for p, s in pd.read_csv(rgb_csv)[["path_rgb_0", "sequence_name"]].values}
    lines = [line for line in open(model / "images.txt") if not line.startswith("#")]
    sessions = {}
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if t:
            sessions[int(t[0])] = session_of[Path(t[9]).name]
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Colour a COLMAP text model by observing session")
    parser.add_argument("--model", type=Path, required=True, help="folder with cameras.txt, images.txt, points3D.txt")
    parser.add_argument("--rgb_csv", type=Path, required=True, help="the sequence's rgb.csv (image -> sequence_name)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", default=None, help="keep only points this session observes, e.g. ssk16")
    args = parser.parse_args()

    sessions = image_sessions(args.model, args.rgb_csv)
    colour_of = dict(zip(sorted(set(sessions.values())), SESSION_COLOURS))

    out_lines, xyz, rgb, counts = [], [], [], {}
    for line in open(args.model / "points3D.txt"):
        if line.startswith("#") or not line.strip():
            out_lines.append(line)
            continue
        t = line.split()
        seen = frozenset(sessions[int(i)] for i in t[8::2])
        if args.session and args.session not in seen:
            continue
        tint = colour_of[next(iter(seen))] if len(seen) == 1 else GREY
        colour = tint_rgb(np.array(t[4:7], dtype=np.float32), tint)
        t[4:7] = map(str, colour)
        out_lines.append(" ".join(t) + "\n")
        xyz.append([float(v) for v in t[1:4]])
        rgb.append(colour)
        key = " + ".join(sorted(seen))
        counts[key] = counts.get(key, 0) + 1

    args.out.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    cloud.colors = o3d.utility.Vector3dVector(np.array(rgb, dtype=np.float64) / 255)
    o3d.io.write_point_cloud(str(args.out / "points.ply"), cloud)
    if not args.session:
        model_out = args.out / "model"
        model_out.mkdir(exist_ok=True)
        for f in args.model.glob("*.txt"):
            if f.name != "points3D.txt":
                shutil.copy2(f, model_out / f.name)
        (model_out / "points3D.txt").write_text("".join(out_lines))

    print(f"{args.model} -> {args.out}  ({len(xyz)} points)")
    for key, n in sorted(counts.items(), key=lambda kv: (kv[0].count("+"), kv[0])):
        print(f"  {key:28s} {n:8d}")


if __name__ == "__main__":
    main()
