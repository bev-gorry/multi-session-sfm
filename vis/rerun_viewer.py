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
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class Observation:
    point_id: int
    image_id: int
    image_name: str
    sequence_name: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class SessionPointCloud:
    point_ids: np.ndarray
    positions: np.ndarray
    colors: np.ndarray
    labels: list[str]


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
            "`pixi run -e rerun-viewer rerun-viewer --exp_yaml=arguments/exp_test.yaml`."
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
            "OpenCV is required to log images. Run through Pixi's rerun-viewer environment."
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


def _session_color_map(sequences: list[str]) -> dict[str, list[int]]:
    return {
        sequence_name: SESSION_COLORS[idx % len(SESSION_COLORS)]
        for idx, sequence_name in enumerate(sequences)
    }


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


def _build_session_point_clouds(
    points3d,
    point_sequences: dict[int, set[str]],
    sequences: list[str],
    alpha_tint: float,
) -> dict[str, SessionPointCloud]:
    session_color = _session_color_map(sequences)
    clouds = {}

    for sequence_name in sequences:
        point_ids = []
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
            point_ids.append(int(point_id))
            positions.append(point.xyz)
            colors.append(_tint_rgb(point.rgb, color, alpha_tint))
            labels.append(f"pid={point_id} sessions={','.join(sorted(observed_sequences))}")

        clouds[sequence_name] = SessionPointCloud(
            point_ids=np.asarray(point_ids, dtype=np.int64),
            positions=np.asarray(positions, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            labels=labels,
        )

    return clouds


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
    session_point_clouds: dict[str, SessionPointCloud],
    point_radius: float,
):
    for sequence_name, cloud in session_point_clouds.items():
        if len(cloud.positions):
            rr.log(
                f"world/sessions/{sequence_name}/points3D",
                rr.Points3D(
                    cloud.positions,
                    colors=cloud.colors,
                    labels=cloud.labels,
                    radii=point_radius,
                ),
                static=True,
            )


def _project_points_to_image(positions: np.ndarray, image, camera) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

    rotation = image.qvec2rotmat()
    xyz_camera = positions @ rotation.T + image.tvec
    x_camera = xyz_camera[:, 0]
    y_camera = xyz_camera[:, 1]
    z_camera = xyz_camera[:, 2]
    in_front = z_camera > 0.0

    x_normalized = np.zeros_like(x_camera)
    y_normalized = np.zeros_like(y_camera)
    x_normalized[in_front] = x_camera[in_front] / z_camera[in_front]
    y_normalized[in_front] = y_camera[in_front] / z_camera[in_front]

    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        x_distorted = x_normalized
        y_distorted = y_normalized
    elif camera.model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        x_distorted = x_normalized
        y_distorted = y_normalized
    elif camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]
        r2 = x_normalized * x_normalized + y_normalized * y_normalized
        distortion = 1.0 + k1 * r2
        x_distorted = x_normalized * distortion
        y_distorted = y_normalized * distortion
    elif camera.model in {"RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1, k2 = params[3], params[4]
        r2 = x_normalized * x_normalized + y_normalized * y_normalized
        distortion = 1.0 + k1 * r2 + k2 * r2 * r2
        x_distorted = x_normalized * distortion
        y_distorted = y_normalized * distortion
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x_normalized * x_normalized + y_normalized * y_normalized
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_distorted = (
            x_normalized * radial
            + 2.0 * p1 * x_normalized * y_normalized
            + p2 * (r2 + 2.0 * x_normalized * x_normalized)
        )
        y_distorted = (
            y_normalized * radial
            + p1 * (r2 + 2.0 * y_normalized * y_normalized)
            + 2.0 * p2 * x_normalized * y_normalized
        )
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")

    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    visible = (
        in_front
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(camera.width))
        & (v >= 0.0)
        & (v < float(camera.height))
    )
    indices = np.flatnonzero(visible)
    xy = np.column_stack([u[indices], v[indices]]).astype(np.float32)
    return xy, indices


def _log_images(
    rr,
    cameras,
    images,
    image_observations: dict[int, list[Observation]],
    session_point_clouds: dict[str, SessionPointCloud],
    rgb_lookup: dict[str, pd.Series],
    image_root: Path,
    sequences: list[str],
    max_images_per_session: int | None,
    strict_images: bool,
    log_reprojected_points: bool,
    reprojected_point_radius: float,
):
    logged_per_sequence = defaultdict(int)
    missing_images = 0
    reprojected_per_sequence = defaultdict(int)
    session_color = _session_color_map(sequences)

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

        if log_reprojected_points:
            for source_sequence, cloud in session_point_clouds.items():
                xy, indices = _project_points_to_image(cloud.positions, image, camera)
                if len(indices) == 0:
                    continue

                rr.log(
                    f"{entity}/rgb/reprojected_points/{source_sequence}",
                    rr.Points2D(
                        xy,
                        labels=[
                            f"pid={int(cloud.point_ids[idx])} from={source_sequence}"
                            for idx in indices
                        ],
                        colors=[session_color[source_sequence]],
                        radii=reprojected_point_radius,
                    ),
                    static=True,
                )
                reprojected_per_sequence[source_sequence] += int(len(indices))

        logged_per_sequence[sequence_name] += 1

    return logged_per_sequence, missing_images, reprojected_per_sequence


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


def _read_gaussian_splat_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header did not terminate: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError(f"Only binary_little_endian PLY splats are supported: {path}")

        vertex_count = None
        properties = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[:1] == ["element"]:
                in_vertex = False
                continue
            if in_vertex and parts[:2] == ["property", "float"]:
                properties.append(parts[2])

        if vertex_count is None:
            raise ValueError(f"PLY is missing a vertex element: {path}")

        dtype = np.dtype([(name, "<f4") for name in properties])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"}
    missing = sorted(required.difference(vertices.dtype.names or ()))
    if missing:
        raise ValueError(f"PLY is missing required Gaussian splat fields {missing}: {path}")

    positions = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    dc_rgb = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]])
    rgb = np.clip((SH_C0 * dc_rgb + 0.5) * 255.0, 0, 255).astype(np.uint8)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"]))
    alpha = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)
    colors = np.column_stack([rgb, alpha])

    scale_fields = [
        name
        for name in ("scale_0", "scale_1", "scale_2")
        if name in vertices.dtype.names
    ]
    if scale_fields:
        log_scales = np.column_stack([vertices[name] for name in scale_fields])
        radii = np.exp(log_scales).mean(axis=1).astype(np.float32)
    else:
        radii = np.full(vertex_count, 0.015, dtype=np.float32)

    return positions, colors, radii


def _log_gaussian_splats(rr, splat_paths: list[str | Path], splat_radius_scale: float):
    for splat_path in splat_paths:
        path = Path(splat_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gaussian splat asset not found: {path}")
        positions, colors, radii = _read_gaussian_splat_ply(path)
        radii = radii * splat_radius_scale
        entity = f"world/gaussian_splats/{_safe_entity_name(path.name)}"
        rr.log(
            entity,
            rr.Points3D(positions, colors=colors, radii=radii),
            static=True,
        )
        rr.log(
            f"{entity}/source",
            rr.TextDocument(
                f"`{path}`\n\nLogged {len(positions):,} Gaussian splat centers from PLY.",
                media_type="text/markdown",
            ),
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
    parser.add_argument("--reprojected-point-radius", type=float, default=2.0)
    parser.add_argument("--alpha-tint", type=float, default=0.4)
    parser.add_argument("--strict-images", action="store_true")
    parser.add_argument(
        "--no-reprojected-points",
        action="store_true",
        help=(
            "Do not log per-image overlays of all in-frame 3D points from each session. "
            "This is useful when exporting a smaller .rrd."
        ),
    )
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
    parser.add_argument(
        "--splat-radius-scale",
        type=float,
        default=1.0,
        help="Multiply decoded Gaussian splat radii for easier inspection in Rerun.",
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
    session_point_clouds = _build_session_point_clouds(
        points3d, point_sequences, sequences, alpha_tint=args.alpha_tint
    )

    rr.init(args.application_id, spawn=args.output is None)
    if args.output:
        Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.output)

    _log_world_header(rr, dataset, subset)
    _log_session_points(
        rr,
        session_point_clouds,
        point_radius=args.point_radius,
    )
    logged_images, missing_images, reprojected_points = _log_images(
        rr,
        cameras,
        images,
        image_observations,
        session_point_clouds,
        rgb_lookup,
        image_root,
        sequences,
        max_images_per_session=args.max_images_per_session,
        strict_images=args.strict_images,
        log_reprojected_points=not args.no_reprojected_points,
        reprojected_point_radius=args.reprojected_point_radius,
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
    _log_gaussian_splats(rr, splat_paths, splat_radius_scale=args.splat_radius_scale)

    print(f"Logged sessions: {', '.join(sequences)}")
    if splat_paths:
        print(f"Logged Gaussian splat assets: {len(splat_paths)}")
    print(f"Logged images per session: {dict(logged_images)}")
    if reprojected_points:
        print(f"Logged reprojected 2D points by source session: {dict(reprojected_points)}")
    print(f"Logged point track documents: {logged_tracks}")
    if skipped:
        print(f"Skipped {skipped} point observations without image/session metadata.")
    if missing_images:
        print(f"Skipped {missing_images} missing image files.")


if __name__ == "__main__":
    main()
