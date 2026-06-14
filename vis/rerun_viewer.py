import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
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


BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

SESSION_COLORS = [BLUE, YELLOW, PINK]
SHARED_COLOR = GREY


@dataclass(frozen=True)
class Observation:
    point_id: int
    image_id: int
    image_name: str
    sequence_name: str
    xy: tuple[float, float]


def _safe_entity_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _as_path(path: str | None) -> Path | None:
    return resolve_repo_path(path) if path else None


def _load_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "The rerun SDK is not installed. Run this through Pixi, e.g. "
            "`pixi run -e lightglue rerun-viewer --exp_yaml=arguments/exp_test.yaml`."
        ) from exc
    return rr


def _parse_exp_yaml(yaml_file: str) -> tuple[str, str, str]:
    args = load_exp_yaml(yaml_file)
    return args["exp_name"], args["dataset"], args["subset"]


def _read_image_rgb(image_path: Path, strict_images: bool) -> np.ndarray | None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to log images. Run through Pixi's lightglue environment."
        ) from exc

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        if strict_images:
            raise RuntimeError(f"OpenCV could not read image: {image_path}")
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _camera_matrix(camera) -> np.ndarray:
    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model in {"RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _tint_rgb(original_rgb: np.ndarray, tint_rgb: list[int], alpha: float) -> np.ndarray:
    return np.clip(
        (1.0 - alpha) * original_rgb.astype(np.float32)
        + alpha * np.array(tint_rgb, dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)


def _image_lookup(rgb_df: pd.DataFrame) -> dict[str, pd.Series]:
    lookup = {}
    for _, row in rgb_df.iterrows():
        path = Path(row["path_rgb_0"])
        lookup[path.name] = row
        lookup[path.as_posix()] = row
    return lookup


def _sequence_for_image(image_name: str, lookup: dict[str, pd.Series]) -> str | None:
    if image_name in lookup:
        return str(lookup[image_name]["sequence_name"])
    basename = Path(image_name).name
    if basename in lookup:
        return str(lookup[basename]["sequence_name"])
    return None


def _resolve_model_path(args, exp_name: str, dataset: str, subset: str) -> Path:
    if args.model_path:
        return _as_path(args.model_path)
    return Path(VSLAMLAB_EVALUATION) / exp_name / dataset / subset / "colmap_00000" / "0"


def _resolve_benchmark_path(args, dataset: str, subset: str) -> Path:
    if args.benchmark_path:
        return _as_path(args.benchmark_path)
    return Path(VSLAMLAB_BENCHMARK) / dataset / subset


def _resolve_splat_dir(args, exp_name: str, dataset: str, subset: str) -> Path:
    if args.splat_dir:
        return _as_path(args.splat_dir)
    return Path("outputs") / "gaussian_splats" / exp_name / dataset / subset


def _collect_observations(points3d, images, rgb_lookup):
    point_observations: dict[int, list[Observation]] = defaultdict(list)
    image_observations: dict[int, list[Observation]] = defaultdict(list)
    point_sequences: dict[int, set[str]] = defaultdict(set)
    skipped = 0

    for point_id, point in tqdm(points3d.items(), desc="Indexing point tracks"):
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image = images.get(int(image_id))
            if image is None or point2d_idx < 0 or point2d_idx >= len(image.xys):
                skipped += 1
                continue
            sequence_name = _sequence_for_image(image.name, rgb_lookup)
            if sequence_name is None:
                skipped += 1
                continue

            xy = tuple(float(v) for v in image.xys[int(point2d_idx)])
            observation = Observation(
                point_id=int(point_id),
                image_id=int(image_id),
                image_name=Path(image.name).name,
                sequence_name=sequence_name,
                xy=xy,
            )
            point_observations[int(point_id)].append(observation)
            image_observations[int(image_id)].append(observation)
            point_sequences[int(point_id)].add(sequence_name)

    return point_observations, image_observations, point_sequences, skipped


def _log_world_header(rr, dataset: str, subset: str):
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/README",
        rr.TextDocument(
            f"# {dataset}/{subset}\n\n"
            "Toggle sessions from the entity tree under `world/sessions`. "
            "Each session contains camera/image entities and the 3D points observed by that session. "
            "Point track documents are under `world/point_tracks/pid_<POINT3D_ID>`.",
            media_type="text/markdown",
        ),
        static=True,
    )


def _log_session_points(
    rr,
    points3d,
    point_sequences: dict[int, set[str]],
    sequences: list[str],
    alpha_tint: float,
    point_radius: float,
):
    session_color = {
        sequence_name: SESSION_COLORS[idx % len(SESSION_COLORS)]
        for idx, sequence_name in enumerate(sequences)
    }

    for sequence_name in sequences:
        positions = []
        colors = []
        labels = []

        for point_id, point in points3d.items():
            observed_sequences = point_sequences.get(int(point_id), set())
            if sequence_name not in observed_sequences:
                continue

            color = (
                session_color[sequence_name]
                if len(observed_sequences) == 1
                else SHARED_COLOR
            )
            positions.append(point.xyz)
            colors.append(_tint_rgb(point.rgb, color, alpha_tint))
            labels.append(f"pid={point_id} sessions={','.join(sorted(observed_sequences))}")

        if positions:
            rr.log(
                f"world/sessions/{sequence_name}/points3D",
                rr.Points3D(
                    np.asarray(positions),
                    colors=np.asarray(colors, dtype=np.uint8),
                    labels=labels,
                    radii=point_radius,
                ),
                static=True,
            )


def _log_images(
    rr,
    cameras,
    images,
    image_observations: dict[int, list[Observation]],
    rgb_lookup: dict[str, pd.Series],
    image_root: Path,
    sequences: list[str],
    max_images_per_session: int | None,
    strict_images: bool,
):
    logged_per_sequence = defaultdict(int)
    missing_images = 0

    for image_id, image in tqdm(images.items(), desc="Logging cameras and images"):
        sequence_name = _sequence_for_image(image.name, rgb_lookup)
        if sequence_name not in sequences:
            continue
        if (
            max_images_per_session is not None
            and logged_per_sequence[sequence_name] >= max_images_per_session
        ):
            continue

        row = rgb_lookup.get(image.name)
        if row is None:
            row = rgb_lookup.get(Path(image.name).name)
        image_rel = Path(row["path_rgb_0"]) if row is not None else Path(image.name)
        image_path = image_root / image_rel
        if not image_path.exists():
            missing_images += 1
            if strict_images:
                raise FileNotFoundError(f"Missing image: {image_path}")
            continue

        camera = cameras[image.camera_id]
        entity = f"world/sessions/{sequence_name}/images/{_safe_entity_name(image.name)}"
        rotation = image.qvec2rotmat()
        translation = image.tvec
        rr.log(
            entity,
            rr.Transform3D(
                mat3x3=rotation,
                translation=translation,
                relation=rr.TransformRelation.ChildFromParent,
            ),
            static=True,
        )
        rr.log(
            entity,
            rr.Pinhole(
                image_from_camera=_camera_matrix(camera),
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
            static=True,
        )

        image_rgb = _read_image_rgb(image_path, strict_images)
        if image_rgb is None:
            missing_images += 1
            continue
        rr.log(f"{entity}/rgb", rr.Image(image_rgb), static=True)

        observations = image_observations.get(int(image_id), [])
        if observations:
            rr.log(
                f"{entity}/rgb/point_pixels",
                rr.Points2D(
                    [obs.xy for obs in observations],
                    labels=[f"pid={obs.point_id}" for obs in observations],
                    colors=[
                        SESSION_COLORS[sequences.index(sequence_name) % len(SESSION_COLORS)]
                    ],
                    radii=3.0,
                ),
                static=True,
            )

        logged_per_sequence[sequence_name] += 1

    return logged_per_sequence, missing_images


def _track_markdown(point_id: int, observations: list[Observation]) -> str:
    grouped = defaultdict(list)
    for obs in observations:
        grouped[obs.sequence_name].append(obs)

    lines = [f"# POINT3D_ID {point_id}", ""]
    for sequence_name in sorted(grouped):
        lines.extend([f"## {sequence_name}", ""])
        for obs in sorted(grouped[sequence_name], key=lambda item: item.image_name):
            lines.append(
                f"- `{obs.image_name}` at pixel ({obs.xy[0]:.1f}, {obs.xy[1]:.1f})"
            )
        lines.append("")
    return "\n".join(lines)


def _log_point_tracks(
    rr,
    points3d,
    point_observations: dict[int, list[Observation]],
    max_track_docs: int | None,
    point_radius: float,
    log_track_images: bool,
    image_root: Path,
    rgb_lookup: dict[str, pd.Series],
    strict_images: bool,
):
    logged = 0
    for point_id, observations in tqdm(
        point_observations.items(), desc="Logging point tracks"
    ):
        if max_track_docs is not None and logged >= max_track_docs:
            break
        point = points3d.get(point_id)
        if point is None:
            continue

        entity = f"world/point_tracks/pid_{point_id}"
        rr.log(
            f"{entity}/point",
            rr.Points3D(
                [point.xyz],
                colors=[point.rgb],
                labels=[f"pid={point_id}"],
                radii=point_radius * 2.0,
            ),
            static=True,
        )
        rr.log(
            f"{entity}/images",
            rr.TextDocument(
                _track_markdown(point_id, observations), media_type="text/markdown"
            ),
            static=True,
        )

        if log_track_images:
            for obs in observations:
                row = rgb_lookup.get(obs.image_name)
                image_rel = Path(row["path_rgb_0"]) if row is not None else Path(obs.image_name)
                image_rgb = _read_image_rgb(image_root / image_rel, strict_images)
                if image_rgb is None:
                    continue

                image_entity = (
                    f"{entity}/{obs.sequence_name}/{_safe_entity_name(obs.image_name)}/rgb"
                )
                rr.log(image_entity, rr.Image(image_rgb), static=True)
                rr.log(
                    f"{image_entity}/projection",
                    rr.Points2D(
                        [obs.xy],
                        labels=[f"pid={point_id}"],
                        colors=[PINK],
                        radii=8.0,
                    ),
                    static=True,
                )

        logged += 1
    return logged


def _default_splat_paths(splat_dir: Path, sequences: list[str]) -> list[Path]:
    if not splat_dir.exists():
        return []

    paths = []
    for sequence_name in sequences:
        session_path = splat_dir / f"{_safe_entity_name(sequence_name)}.ply"
        if session_path.exists():
            paths.append(session_path)

    seen = {path.resolve() for path in paths}
    for path in sorted(splat_dir.glob("*.ply")):
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return paths


def _log_gaussian_splats(rr, splat_paths: list[str | Path]):
    for splat_path in splat_paths:
        path = Path(splat_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gaussian splat asset not found: {path}")
        rr.log(
            f"world/gaussian_splats/{_safe_entity_name(path.name)}",
            rr.Asset3D(path=path),
            static=True,
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="View a multi-session COLMAP reconstruction in Rerun."
    )
    parser.add_argument("--exp_yaml", type=str, default="arguments/exp_test.yaml")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default=None, help="Write an .rrd instead of spawning the viewer."
    )
    parser.add_argument("--application-id", type=str, default="multi_session_sfm")
    parser.add_argument("--sessions", nargs="*", default=None, help="Subset/order of session names to log.")
    parser.add_argument("--max-images-per-session", type=int, default=None)
    parser.add_argument("--max-track-docs", type=int, default=None)
    parser.add_argument(
        "--log-track-images",
        action="store_true",
        help="Also log actual images under each point-track entity. Use with --max-track-docs on large reconstructions.",
    )
    parser.add_argument("--point-radius", type=float, default=0.015)
    parser.add_argument("--alpha-tint", type=float, default=0.4)
    parser.add_argument("--strict-images", action="store_true")
    parser.add_argument(
        "--splat-asset",
        action="append",
        default=[],
        help="Path to a Gaussian splat/3D asset to log under world/gaussian_splats. Can be repeated.",
    )
    parser.add_argument(
        "--splat-dir",
        type=str,
        default=None,
        help="Directory containing per-session .ply splats. Defaults to outputs/gaussian_splats/<exp>/<dataset>/<subset>.",
    )
    parser.add_argument(
        "--no-auto-splats",
        action="store_true",
        help="Disable automatic loading of .ply files from --splat-dir.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    rr = _load_rerun()

    exp_name, dataset, subset = _parse_exp_yaml(args.exp_yaml)
    model_path = _resolve_model_path(args, exp_name, dataset, subset)
    benchmark_path = _resolve_benchmark_path(args, dataset, subset)
    splat_dir = _resolve_splat_dir(args, exp_name, dataset, subset)
    rgb_csv = benchmark_path / "rgb.csv"
    image_root = benchmark_path

    if not model_path.exists():
        raise FileNotFoundError(f"COLMAP model path does not exist: {model_path}")
    if not rgb_csv.exists():
        raise FileNotFoundError(f"rgb.csv does not exist: {rgb_csv}")

    rgb_df = pd.read_csv(rgb_csv)
    sequences = args.sessions or list(dict.fromkeys(rgb_df["sequence_name"].astype(str)))
    if len(sequences) != 3:
        print(f"Warning: expected 3 sessions, logging {len(sequences)}: {sequences}")

    cameras, images, points3d = read_model(str(model_path), ext="")
    rgb_lookup = _image_lookup(rgb_df)
    point_observations, image_observations, point_sequences, skipped = _collect_observations(
        points3d, images, rgb_lookup
    )

    rr.init(args.application_id, spawn=args.output is None)
    if args.output:
        Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.output)

    _log_world_header(rr, dataset, subset)
    _log_session_points(
        rr,
        points3d,
        point_sequences,
        sequences,
        alpha_tint=args.alpha_tint,
        point_radius=args.point_radius,
    )
    logged_images, missing_images = _log_images(
        rr,
        cameras,
        images,
        image_observations,
        rgb_lookup,
        image_root,
        sequences,
        max_images_per_session=args.max_images_per_session,
        strict_images=args.strict_images,
    )
    logged_tracks = _log_point_tracks(
        rr,
        points3d,
        point_observations,
        max_track_docs=args.max_track_docs,
        point_radius=args.point_radius,
        log_track_images=args.log_track_images,
        image_root=image_root,
        rgb_lookup=rgb_lookup,
        strict_images=args.strict_images,
    )
    splat_paths = list(args.splat_asset)
    if not args.no_auto_splats:
        splat_paths.extend(_default_splat_paths(splat_dir, sequences))
    _log_gaussian_splats(rr, splat_paths)

    print(f"Logged sessions: {', '.join(sequences)}")
    if splat_paths:
        print(f"Logged Gaussian splat assets: {len(splat_paths)}")
    print(f"Logged images per session: {dict(logged_images)}")
    print(f"Logged point track documents: {logged_tracks}")
    if skipped:
        print(f"Skipped {skipped} point observations without image/session metadata.")
    if missing_images:
        print(f"Skipped {missing_images} missing image files.")


if __name__ == "__main__":
    main()
