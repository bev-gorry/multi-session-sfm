import argparse
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment import load_exp_yaml, resolve_repo_path

try:
    from baselines.VSLAM_LAB.path_constants import (
        VSLAMLAB_BENCHMARK,
        VSLAMLAB_EVALUATION,
    )
    from baselines.VSLAM_LAB.Baselines.colmap.scripts.python.read_write_model import (
        read_model,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import VSLAM-LAB COLMAP helpers. Run the VSLAM-LAB setup first."
    ) from exc


SH_C0 = 0.28209479177387814
PLY_PROPERTIES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def _safe_file_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "session"


def _as_path(path: str | None) -> Path | None:
    return resolve_repo_path(path) if path else None


def _parse_exp_yaml(yaml_file: str) -> tuple[str, str, str]:
    args = load_exp_yaml(yaml_file)
    return args["exp_name"], args["dataset"], args["subset"]


def _resolve_model_path(args, exp_name: str, dataset: str, subset: str) -> Path:
    if args.model_path:
        return _as_path(args.model_path)
    return Path(VSLAMLAB_EVALUATION) / exp_name / dataset / subset / "colmap_00000" / "0"


def _resolve_benchmark_path(args, dataset: str, subset: str) -> Path:
    if args.benchmark_path:
        return _as_path(args.benchmark_path)
    return Path(VSLAMLAB_BENCHMARK) / dataset / subset


def _resolve_output_dir(args, exp_name: str, dataset: str, subset: str) -> Path:
    if args.output_dir:
        return _as_path(args.output_dir)
    return Path("outputs") / "gaussian_splats" / exp_name / dataset / subset


def _image_lookup(rgb_df: pd.DataFrame) -> dict[str, str]:
    lookup = {}
    for _, row in rgb_df.iterrows():
        image_path = Path(row["path_rgb_0"])
        sequence_name = str(row["sequence_name"])
        lookup[image_path.name] = sequence_name
        lookup[image_path.as_posix()] = sequence_name
    return lookup


def _sequence_for_image(image_name: str, lookup: dict[str, str]) -> str | None:
    if image_name in lookup:
        return lookup[image_name]
    return lookup.get(Path(image_name).name)


def _collect_session_point_ids(points3d, images, rgb_lookup: dict[str, str]):
    session_point_ids: dict[str, set[int]] = defaultdict(set)
    skipped = 0

    for point_id, point in tqdm(points3d.items(), desc="Indexing point sessions"):
        for image_id in point.image_ids:
            image = images.get(int(image_id))
            if image is None:
                skipped += 1
                continue
            sequence_name = _sequence_for_image(image.name, rgb_lookup)
            if sequence_name is None:
                skipped += 1
                continue
            session_point_ids[sequence_name].add(int(point_id))

    return session_point_ids, skipped


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
    return (rgb01 - 0.5) / SH_C0


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _ply_header(vertex_count: int) -> bytes:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {name}" for name in PLY_PROPERTIES)
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def _write_gaussian_ply(
    output_path: Path,
    points3d,
    point_ids: list[int],
    scale: float,
    opacity: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_scale = math.log(scale)
    logit_opacity = _logit(opacity)
    row_struct = struct.Struct("<17f")

    with open(output_path, "wb") as f:
        f.write(_ply_header(len(point_ids)))
        for point_id in tqdm(point_ids, desc=f"Writing {output_path.name}"):
            point = points3d[point_id]
            f_dc = _rgb_to_sh_dc(point.rgb)
            f.write(
                row_struct.pack(
                    float(point.xyz[0]),
                    float(point.xyz[1]),
                    float(point.xyz[2]),
                    0.0,
                    0.0,
                    0.0,
                    float(f_dc[0]),
                    float(f_dc[1]),
                    float(f_dc[2]),
                    logit_opacity,
                    log_scale,
                    log_scale,
                    log_scale,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export one COLMAP-initialized Gaussian Splat PLY per session. "
            "The splats use COLMAP point positions/colors and are grouped by rgb.csv sequence_name."
        )
    )
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=0.015)
    parser.add_argument("--opacity", type=float, default=0.8)
    parser.add_argument(
        "--max-points-per-session",
        type=int,
        default=None,
        help="Optional deterministic cap for quick previews.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    exp_name, dataset, subset = _parse_exp_yaml(args.exp_yaml)
    model_path = _resolve_model_path(args, exp_name, dataset, subset)
    benchmark_path = _resolve_benchmark_path(args, dataset, subset)
    output_dir = _resolve_output_dir(args, exp_name, dataset, subset)
    rgb_csv = benchmark_path / "rgb.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"COLMAP model path does not exist: {model_path}")
    if not rgb_csv.exists():
        raise FileNotFoundError(f"rgb.csv does not exist: {rgb_csv}")
    if args.scale <= 0:
        raise ValueError("--scale must be greater than zero")

    rgb_df = pd.read_csv(rgb_csv)
    sequences = args.sessions or list(dict.fromkeys(rgb_df["sequence_name"].astype(str)))
    cameras, images, points3d = read_model(str(model_path), ext="")
    del cameras

    session_point_ids, skipped = _collect_session_point_ids(
        points3d, images, _image_lookup(rgb_df)
    )

    manifest_rows = []
    for sequence_name in sequences:
        point_ids = sorted(session_point_ids.get(sequence_name, set()))
        if args.max_points_per_session is not None:
            point_ids = point_ids[: args.max_points_per_session]
        if not point_ids:
            print(f"Warning: no points observed by session {sequence_name}")
            continue

        output_path = output_dir / f"{_safe_file_stem(sequence_name)}.ply"
        _write_gaussian_ply(output_path, points3d, point_ids, args.scale, args.opacity)
        manifest_rows.append(
            {
                "session": sequence_name,
                "path": output_path.as_posix(),
                "points": len(point_ids),
            }
        )

    manifest = output_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    print(f"Wrote {len(manifest_rows)} session splat assets to {output_dir}")
    print(f"Wrote manifest: {manifest}")
    if skipped:
        print(f"Skipped {skipped} observations without image/session metadata.")


if __name__ == "__main__":
    main()
