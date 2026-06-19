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
    class_ids: np.ndarray
    keypoint_ids: np.ndarray
    positions: np.ndarray
    colors: np.ndarray
    labels: list[str]
    track_summaries: list[str]


@dataclass(frozen=True)
class ImageRecord:
    frame: int
    session_frame: int
    timestamp_ns: int
    image_id: int
    image_name: str
    sequence_name: str
    image_path: Path


@dataclass(frozen=True)
class PixelLookupResult:
    point_id: int
    source_session: str
    candidate_type: str
    projected_xy: tuple[float, float]
    distance_px: float


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


def _point_identity_components(point_ids: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
    point_ids = np.asarray(point_ids, dtype=np.uint64)
    class_ids = (point_ids // 65536).astype(np.uint16)
    keypoint_ids = (point_ids % 65536).astype(np.uint16)
    return class_ids, keypoint_ids


def _image_lookup(rgb_df: pd.DataFrame) -> dict[str, pd.Series]:
    lookup = {}
    for _, row in rgb_df.iterrows():
        path = Path(row["path_rgb_0"])
        lookup[path.name] = row
        lookup[path.as_posix()] = row
    return lookup


def _timestamp_column(rgb_df: pd.DataFrame) -> str:
    candidates = [col for col in rgb_df.columns if col.startswith("ts_rgb_0")]
    if not candidates:
        raise KeyError("rgb.csv must contain a timestamp column starting with 'ts_rgb_0'")
    return candidates[0]


def _sequence_for_image(image_name: str, lookup: dict[str, pd.Series]) -> str | None:
    if image_name in lookup:
        return str(lookup[image_name]["sequence_name"])
    basename = Path(image_name).name
    if basename in lookup:
        return str(lookup[basename]["sequence_name"])
    return None


def _prepare_image_records(
    rgb_df: pd.DataFrame,
    images,
    rgb_lookup: dict[str, pd.Series],
    image_root: Path,
    sequences: list[str],
    max_images_per_session: int | None,
    strict_images: bool,
) -> tuple[list[ImageRecord], int]:
    timestamp_col = _timestamp_column(rgb_df)
    image_by_name = {}
    for image_id, image in images.items():
        image_by_name[image.name] = (int(image_id), image)
        image_by_name[Path(image.name).name] = (int(image_id), image)

    rows = rgb_df.copy()
    rows["sequence_name"] = rows["sequence_name"].astype(str)
    rows = rows[rows["sequence_name"].isin(sequences)]
    rows = rows.sort_values(["sequence_name", timestamp_col, "path_rgb_0"])

    session_frame_by_path = {}
    for sequence_name, group in rows.groupby("sequence_name", sort=False):
        for session_frame, (_, row) in enumerate(group.iterrows()):
            session_frame_by_path[Path(row["path_rgb_0"]).name] = session_frame

    candidate_rows = rows.sort_values([timestamp_col, "sequence_name", "path_rgb_0"])
    kept_per_sequence = defaultdict(int)
    records = []
    missing = 0

    for _, row in candidate_rows.iterrows():
        sequence_name = str(row["sequence_name"])
        if (
            max_images_per_session is not None
            and kept_per_sequence[sequence_name] >= max_images_per_session
        ):
            continue

        image_rel = Path(row["path_rgb_0"])
        lookup_key = image_rel.name
        image_item = image_by_name.get(lookup_key) or image_by_name.get(image_rel.as_posix())
        if image_item is None:
            missing += 1
            continue

        image_path = image_root / image_rel
        if not image_path.exists():
            missing += 1
            if strict_images:
                raise FileNotFoundError(f"Missing image: {image_path}")
            continue

        image_id, image = image_item
        row_lookup = rgb_lookup.get(image.name)
        if row_lookup is None:
            row_lookup = rgb_lookup.get(Path(image.name).name)
        if row_lookup is None:
            missing += 1
            continue

        records.append(
            ImageRecord(
                frame=len(records),
                session_frame=session_frame_by_path[lookup_key],
                timestamp_ns=int(row[timestamp_col]),
                image_id=image_id,
                image_name=Path(image.name).name,
                sequence_name=sequence_name,
                image_path=image_path,
            )
        )
        kept_per_sequence[sequence_name] += 1

    return records, missing


def _prepare_image_catalog(
    rgb_df: pd.DataFrame,
    images,
    image_root: Path,
    sequences: list[str],
    strict_images: bool,
) -> tuple[dict[int, ImageRecord], int]:
    records, missing = _prepare_image_records(
        rgb_df,
        images,
        _image_lookup(rgb_df),
        image_root,
        sequences,
        max_images_per_session=None,
        strict_images=strict_images,
    )
    return {record.image_id: record for record in records}, missing


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


def _collect_observations(points3d, images, rgb_lookup, point_ids: list[int] | None = None):
    point_observations: dict[int, list[Observation]] = defaultdict(list)
    image_observations: dict[int, list[Observation]] = defaultdict(list)
    point_sequences: dict[int, set[str]] = defaultdict(set)
    skipped = 0

    if point_ids is None:
        point_items = points3d.items()
        description = "Indexing point tracks"
    else:
        point_items = [
            (int(point_id), points3d[int(point_id)])
            for point_id in point_ids
            if int(point_id) in points3d
        ]
        description = "Indexing inspected point tracks"

    for point_id, point in tqdm(point_items, desc=description):
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


def _track_summary(observations: list[Observation], limit: int) -> str:
    if not observations:
        return "no registered image observations"

    sorted_observations = sorted(
        observations, key=lambda item: (item.sequence_name, item.image_name, item.xy)
    )
    pieces = [
        f"{obs.sequence_name}:{obs.image_name}@({obs.xy[0]:.1f},{obs.xy[1]:.1f})"
        for obs in sorted_observations[:limit]
    ]
    remaining = len(sorted_observations) - len(pieces)
    if remaining > 0:
        pieces.append(f"+{remaining} more")
    return "; ".join(pieces)


def _build_session_point_clouds(
    points3d,
    point_observations: dict[int, list[Observation]],
    point_sequences: dict[int, set[str]],
    sequences: list[str],
    alpha_tint: float,
    track_summary_limit: int,
) -> dict[str, SessionPointCloud]:
    session_color = _session_color_map(sequences)
    clouds = {}

    for sequence_name in sequences:
        point_ids = []
        positions = []
        colors = []
        labels = []
        track_summaries = []

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
            track_summaries.append(
                _track_summary(point_observations.get(int(point_id), []), track_summary_limit)
            )

        point_ids_array = np.asarray(point_ids, dtype=np.int64)
        class_ids, keypoint_ids = _point_identity_components(point_ids_array)
        clouds[sequence_name] = SessionPointCloud(
            point_ids=point_ids_array,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            positions=np.asarray(positions, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            labels=labels,
            track_summaries=track_summaries,
        )

    return clouds


def _log_world_header(rr, dataset: str, subset: str):
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/README",
        rr.TextDocument(
            f"# {dataset}/{subset}\n\n"
            "This is a precomputed-asset viewer: no reconstruction, retrieval, matching, "
            "or VPR distance-matrix step is run here.\n\n"
            "Scrub the `frame` timeline to step through sorted images. The `world/current_camera/image` "
            "view shows the active image plus selectable 2D observations and cross-session "
            "3D-point reprojections. Toggle point clouds, reprojection layers, rays, and "
            "Gaussian splat centers from the entity tree.",
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
                    show_labels=False,
                    class_ids=cloud.class_ids,
                    keypoint_ids=cloud.keypoint_ids,
                    radii=point_radius,
                ),
                static=True,
            )
            rr.log(
                f"world/sessions/{sequence_name}/points3D",
                rr.AnyValues(
                    point_id=cloud.point_ids.tolist(),
                    session=[sequence_name] * len(cloud.point_ids),
                    observations=cloud.track_summaries,
                ),
                static=True,
            )


def _project_points_to_image(
    positions: np.ndarray, image, camera
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(positions) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

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
    return xy, indices, z_camera[indices].astype(np.float32)


def _camera_center(image) -> np.ndarray:
    rotation = image.qvec2rotmat()
    return (-rotation.T @ image.tvec).astype(np.float32)


def _log_static_cameras(rr, cameras, images, records: list[ImageRecord]):
    for record in tqdm(records, desc="Logging static camera frustums"):
        image = images[record.image_id]
        camera = cameras[image.camera_id]
        entity = f"world/sessions/{record.sequence_name}/cameras/{_safe_entity_name(record.image_name)}"
        rr.log(
            entity,
            rr.Transform3D(
                mat3x3=image.qvec2rotmat(),
                translation=image.tvec,
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
        rr.log(
            entity,
            rr.AnyValues(
                image_id=record.image_id,
                image_name=record.image_name,
                session=record.sequence_name,
                frame=record.frame,
                session_frame=record.session_frame,
                timestamp_ns=record.timestamp_ns,
            ),
            static=True,
        )


def _log_2d_observations(
    rr,
    entity: str,
    observations: list[Observation],
    color: list[int],
    target_record: ImageRecord,
    radius: float,
    static: bool = False,
):
    if not observations:
        return

    observation_point_ids = [obs.point_id for obs in observations]
    observation_class_ids, observation_keypoint_ids = _point_identity_components(
        observation_point_ids
    )
    rr.log(
        entity,
        rr.Points2D(
            [obs.xy for obs in observations],
            labels=[f"pid={obs.point_id}" for obs in observations],
            show_labels=False,
            class_ids=observation_class_ids,
            keypoint_ids=observation_keypoint_ids,
            colors=[color],
            radii=radius,
        ),
        static=static,
    )
    rr.log(
        entity,
        rr.AnyValues(
            point_id=observation_point_ids,
            source_session=[obs.sequence_name for obs in observations],
            target_session=[target_record.sequence_name] * len(observations),
            target_image=[target_record.image_name] * len(observations),
            pixel_u=[float(obs.xy[0]) for obs in observations],
            pixel_v=[float(obs.xy[1]) for obs in observations],
        ),
        static=static,
    )


def _log_static_camera_images(
    rr,
    cameras,
    images,
    records: list[ImageRecord],
    image_observations: dict[int, list[Observation]],
    session_point_clouds: dict[str, SessionPointCloud],
    sequences: list[str],
    log_reprojected_points: bool,
    reprojected_point_radius: float,
    max_reprojected_points_per_session: int | None,
):
    session_color = _session_color_map(sequences)
    for record in tqdm(records, desc="Logging static camera images"):
        image = images[record.image_id]
        camera = cameras[image.camera_id]
        camera_entity = (
            f"world/sessions/{record.sequence_name}/cameras/"
            f"{_safe_entity_name(record.image_name)}"
        )
        image_entity = f"{camera_entity}/image"
        rr.log(image_entity, rr.EncodedImage(path=record.image_path), static=True)
        rr.log(
            image_entity,
            rr.AnyValues(
                image_id=record.image_id,
                image_name=record.image_name,
                session=record.sequence_name,
                frame=record.frame,
                session_frame=record.session_frame,
                timestamp_ns=record.timestamp_ns,
            ),
            static=True,
        )
        _log_2d_observations(
            rr,
            f"{image_entity}/observed_points",
            image_observations.get(int(record.image_id), []),
            session_color[record.sequence_name],
            record,
            radius=3.0,
            static=True,
        )

        if log_reprojected_points:
            for source_sequence, cloud in session_point_clouds.items():
                xy, indices, depths = _project_points_to_image(
                    cloud.positions, image, camera
                )
                if len(indices) == 0:
                    continue
                if (
                    max_reprojected_points_per_session is not None
                    and len(indices) > max_reprojected_points_per_session
                ):
                    chosen = np.linspace(
                        0, len(indices) - 1, max_reprojected_points_per_session
                    ).astype(np.int64)
                    xy = xy[chosen]
                    indices = indices[chosen]
                    depths = depths[chosen]

                reprojection_entity = (
                    f"{image_entity}/reprojected_points/{source_sequence}"
                )
                rr.log(
                    reprojection_entity,
                    rr.Points2D(
                        xy,
                        labels=[
                            f"pid={int(cloud.point_ids[idx])} from={source_sequence}"
                            for idx in indices
                        ],
                        show_labels=False,
                        class_ids=cloud.class_ids[indices],
                        keypoint_ids=cloud.keypoint_ids[indices],
                        colors=[session_color[source_sequence]],
                        radii=reprojected_point_radius,
                    ),
                    static=True,
                )
                rr.log(
                    reprojection_entity,
                    rr.AnyValues(
                        point_id=[int(cloud.point_ids[idx]) for idx in indices],
                        source_session=[source_sequence] * len(indices),
                        target_session=[record.sequence_name] * len(indices),
                        target_image=[record.image_name] * len(indices),
                        pixel_u=xy[:, 0].astype(float).tolist(),
                        pixel_v=xy[:, 1].astype(float).tolist(),
                        depth=depths.astype(float).tolist(),
                        world_x=cloud.positions[indices, 0].astype(float).tolist(),
                        world_y=cloud.positions[indices, 1].astype(float).tolist(),
                        world_z=cloud.positions[indices, 2].astype(float).tolist(),
                    ),
                    static=True,
                )


def _find_support_images(
    record: ImageRecord,
    image_observations: dict[int, list[Observation]],
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
    sequences: list[str],
    support_images_per_session: int,
    include_active_session: bool,
) -> dict[str, list[tuple[ImageRecord, int]]]:
    support = {sequence_name: [] for sequence_name in sequences}
    if support_images_per_session <= 0:
        return support

    current_point_ids = {
        obs.point_id for obs in image_observations.get(int(record.image_id), [])
    }
    if not current_point_ids:
        return support

    counts_by_image: dict[int, int] = defaultdict(int)
    for point_id in current_point_ids:
        for obs in point_observations.get(point_id, []):
            if obs.image_id == record.image_id:
                continue
            candidate = image_catalog.get(int(obs.image_id))
            if candidate is None:
                continue
            if (
                not include_active_session
                and candidate.sequence_name == record.sequence_name
            ):
                continue
            counts_by_image[int(obs.image_id)] += 1

    grouped: dict[str, list[tuple[ImageRecord, int]]] = defaultdict(list)
    for image_id, shared_count in counts_by_image.items():
        candidate = image_catalog.get(image_id)
        if candidate is not None:
            grouped[candidate.sequence_name].append((candidate, shared_count))

    for sequence_name in sequences:
        ranked = sorted(
            grouped.get(sequence_name, []),
            key=lambda item: (-item[1], abs(item[0].timestamp_ns - record.timestamp_ns), item[0].image_name),
        )
        support[sequence_name] = ranked[:support_images_per_session]

    return support


def _log_support_images(
    rr,
    cameras,
    images,
    record: ImageRecord,
    image_observations: dict[int, list[Observation]],
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
    sequences: list[str],
    support_images_per_session: int,
    include_active_session: bool,
):
    if support_images_per_session <= 0:
        return

    session_color = _session_color_map(sequences)
    rr.log("world/support_images", rr.Clear(recursive=True))
    support_images = _find_support_images(
        record,
        image_observations,
        point_observations,
        image_catalog,
        sequences,
        support_images_per_session,
        include_active_session,
    )

    for sequence_name, ranked_records in support_images.items():
        for slot_idx, (support_record, shared_count) in enumerate(ranked_records):
            image = images[support_record.image_id]
            camera = cameras[image.camera_id]
            entity = f"world/support_images/{sequence_name}/slot_{slot_idx}"
            image_entity = f"{entity}/image"

            rr.log(
                entity,
                rr.Transform3D(
                    mat3x3=image.qvec2rotmat(),
                    translation=image.tvec,
                    relation=rr.TransformRelation.ChildFromParent,
                ),
            )
            rr.log(
                entity,
                rr.Pinhole(
                    image_from_camera=_camera_matrix(camera),
                    resolution=[camera.width, camera.height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
            rr.log(image_entity, rr.EncodedImage(path=support_record.image_path))
            rr.log(
                f"{image_entity}/status",
                rr.TextDocument(
                    "# Support image\n\n"
                    f"- shared active-image points: `{shared_count}`\n"
                    f"- session: `{support_record.sequence_name}`\n"
                    f"- session_frame: `{support_record.session_frame}`\n"
                    f"- image: `{support_record.image_name}`\n"
                    f"- image_id: `{support_record.image_id}`",
                    media_type="text/markdown",
                ),
            )
            _log_2d_observations(
                rr,
                f"{image_entity}/observed_points",
                image_observations.get(int(support_record.image_id), []),
                session_color[sequence_name],
                support_record,
                radius=4.0,
            )


def _log_current_image_timeline(
    rr,
    cameras,
    images,
    records: list[ImageRecord],
    image_observations: dict[int, list[Observation]],
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
    session_point_clouds: dict[str, SessionPointCloud],
    sequences: list[str],
    log_reprojected_points: bool,
    reprojected_point_radius: float,
    show_rays: bool,
    max_rays_per_session: int,
    max_reprojected_points_per_session: int | None,
    support_images_per_session: int,
    include_active_support_session: bool,
):
    logged_per_sequence = defaultdict(int)
    reprojected_per_sequence = defaultdict(int)
    session_color = _session_color_map(sequences)

    for record in tqdm(records, desc="Logging timeline images"):
        image = images[record.image_id]
        camera = cameras[image.camera_id]
        rr.set_time("frame", sequence=record.frame)
        rr.set_time("capture_time", timestamp=record.timestamp_ns * 1e-9)

        current_camera = "world/current_camera"
        rr.log(
            current_camera,
            rr.Transform3D(
                mat3x3=image.qvec2rotmat(),
                translation=image.tvec,
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )
        rr.log(
            current_camera,
            rr.Pinhole(
                image_from_camera=_camera_matrix(camera),
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )

        image_entity = "world/current_camera/image"
        rr.log(image_entity, rr.EncodedImage(path=record.image_path))
        rr.log(f"{image_entity}/observed_points", rr.Clear(recursive=True))
        rr.log(f"{image_entity}/reprojected_points", rr.Clear(recursive=True))
        rr.log("world/current_rays", rr.Clear(recursive=True))
        rr.log(
            "current/status",
            rr.TextDocument(
                "# Current image\n\n"
                f"- frame: `{record.frame}`\n"
                f"- session: `{record.sequence_name}`\n"
                f"- session_frame: `{record.session_frame}`\n"
                f"- image: `{record.image_name}`\n"
                f"- image_id: `{record.image_id}`\n"
                f"- timestamp_ns: `{record.timestamp_ns}`",
                media_type="text/markdown",
            ),
        )
        rr.log(
            image_entity,
            rr.AnyValues(
                image_id=record.image_id,
                image_name=record.image_name,
                session=record.sequence_name,
                frame=record.frame,
                session_frame=record.session_frame,
                timestamp_ns=record.timestamp_ns,
            ),
        )

        observations = image_observations.get(int(record.image_id), [])
        _log_2d_observations(
            rr,
            f"{image_entity}/observed_points",
            observations,
            session_color[record.sequence_name],
            record,
            radius=3.0,
        )
        _log_support_images(
            rr,
            cameras,
            images,
            record,
            image_observations,
            point_observations,
            image_catalog,
            sequences,
            support_images_per_session,
            include_active_support_session,
        )

        if log_reprojected_points:
            for source_sequence, cloud in session_point_clouds.items():
                xy, indices, depths = _project_points_to_image(cloud.positions, image, camera)
                if len(indices) == 0:
                    continue
                if (
                    max_reprojected_points_per_session is not None
                    and len(indices) > max_reprojected_points_per_session
                ):
                    chosen = np.linspace(
                        0, len(indices) - 1, max_reprojected_points_per_session
                    ).astype(np.int64)
                    xy = xy[chosen]
                    indices = indices[chosen]
                    depths = depths[chosen]

                entity = f"{image_entity}/reprojected_points/{source_sequence}"
                rr.log(
                    entity,
                    rr.Points2D(
                        xy,
                        labels=[
                            f"pid={int(cloud.point_ids[idx])} from={source_sequence}"
                            for idx in indices
                        ],
                        show_labels=False,
                        class_ids=cloud.class_ids[indices],
                        keypoint_ids=cloud.keypoint_ids[indices],
                        colors=[session_color[source_sequence]],
                        radii=reprojected_point_radius,
                    ),
                )
                rr.log(
                    entity,
                    rr.AnyValues(
                        point_id=[int(cloud.point_ids[idx]) for idx in indices],
                        source_session=[source_sequence] * len(indices),
                        target_session=[record.sequence_name] * len(indices),
                        target_image=[record.image_name] * len(indices),
                        pixel_u=xy[:, 0].astype(float).tolist(),
                        pixel_v=xy[:, 1].astype(float).tolist(),
                        depth=depths.astype(float).tolist(),
                        world_x=cloud.positions[indices, 0].astype(float).tolist(),
                        world_y=cloud.positions[indices, 1].astype(float).tolist(),
                        world_z=cloud.positions[indices, 2].astype(float).tolist(),
                    ),
                )
                reprojected_per_sequence[source_sequence] += int(len(indices))

                if show_rays and max_rays_per_session > 0:
                    ray_indices = indices[:max_rays_per_session]
                    camera_center = _camera_center(image)
                    strips = [
                        [camera_center, cloud.positions[idx].astype(np.float32)]
                        for idx in ray_indices
                    ]
                    rr.log(
                        f"world/current_rays/{source_sequence}",
                        rr.LineStrips3D(
                            strips,
                            colors=[session_color[source_sequence]],
                            radii=0.005,
                            labels=[
                                f"pid={int(cloud.point_ids[idx])} ray from {record.image_name}"
                                for idx in ray_indices
                            ],
                            show_labels=False,
                        ),
                    )

        logged_per_sequence[record.sequence_name] += 1

    return logged_per_sequence, reprojected_per_sequence


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
        class_ids, keypoint_ids = _point_identity_components([point_id])
        rr.log(
            f"{entity}/point",
            rr.Points3D(
                [point.xyz],
                colors=[point.rgb],
                labels=[f"pid={point_id}"],
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
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
                        class_ids=class_ids,
                        keypoint_ids=keypoint_ids,
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


def _log_gaussian_splats(
    rr,
    splat_paths: list[str | Path],
    splat_radius_scale: float,
    splat_min_radius: float,
    splat_max_radius: float | None,
    splat_opacity_scale: float,
    max_splats_per_session: int | None,
):
    for splat_path in splat_paths:
        path = Path(splat_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gaussian splat asset not found: {path}")
        positions, colors, radii = _read_gaussian_splat_ply(path)
        total_splats = len(positions)
        if max_splats_per_session is not None and total_splats > max_splats_per_session:
            indices = np.linspace(0, total_splats - 1, max_splats_per_session).astype(
                np.int64
            )
            positions = positions[indices]
            colors = colors[indices]
            radii = radii[indices]
        radii = radii * splat_radius_scale
        if splat_min_radius > 0.0:
            radii = np.maximum(radii, splat_min_radius)
        if splat_max_radius is not None:
            radii = np.minimum(radii, splat_max_radius)
        if splat_opacity_scale != 1.0:
            colors = colors.copy()
            colors[:, 3] = np.clip(
                colors[:, 3].astype(np.float32) * splat_opacity_scale, 0, 255
            ).astype(np.uint8)
        entity = f"world/gaussian_splats/{_safe_entity_name(path.stem)}"
        rr.log(
            entity,
            rr.Points3D(positions, colors=colors, radii=radii),
            static=True,
        )
        rr.log(
            entity,
            rr.AnyValues(
                splat_asset=str(path),
                logged_splats=len(positions),
                total_splats=total_splats,
                point_style_visualization=True,
                true_gaussian_rasterizer=False,
                splat_radius_scale=splat_radius_scale,
                splat_min_radius=splat_min_radius,
                splat_max_radius=-1.0 if splat_max_radius is None else splat_max_radius,
                splat_opacity_scale=splat_opacity_scale,
            ),
            static=True,
        )
        rr.log(
            f"{entity}/source",
            rr.TextDocument(
                f"`{path}`\n\n"
                f"Logged {len(positions):,} of {total_splats:,} Gaussian splat centers "
                "from PLY as a radius/opacity point proxy. Rerun 0.33 does not expose "
                "a native anisotropic 3D Gaussian rasterizer, so this is improved "
                "inspection geometry rather than continuous splat rendering.",
                media_type="text/markdown",
            ),
            static=True,
        )


def _find_image_record(
    image_catalog: dict[int, ImageRecord],
    image_name: str,
) -> ImageRecord:
    requested = Path(image_name).name
    exact = [
        record
        for record in image_catalog.values()
        if record.image_name == image_name or Path(record.image_name).name == requested
    ]
    if not exact:
        raise ValueError(f"Could not find image '{image_name}' in the COLMAP/rgb.csv catalog.")
    if len(exact) > 1:
        sessions = ", ".join(sorted(record.sequence_name for record in exact))
        raise ValueError(
            f"Image name '{image_name}' is ambiguous across sessions: {sessions}. "
            "Pass the catalog image name exactly as shown in Rerun."
        )
    return exact[0]


def _nearest_projected_candidate(
    cloud: SessionPointCloud,
    image,
    camera,
    click_xy: np.ndarray,
    source_session: str,
) -> PixelLookupResult | None:
    projected_xy, indices, _depths = _project_points_to_image(
        cloud.positions, image, camera
    )
    if len(indices) == 0:
        return None

    distances = np.linalg.norm(projected_xy - click_xy[None, :], axis=1)
    nearest_idx = int(np.argmin(distances))
    cloud_idx = int(indices[nearest_idx])
    xy = projected_xy[nearest_idx]
    return PixelLookupResult(
        point_id=int(cloud.point_ids[cloud_idx]),
        source_session=source_session,
        candidate_type="reprojected",
        projected_xy=(float(xy[0]), float(xy[1])),
        distance_px=float(distances[nearest_idx]),
    )


def _lookup_point_from_pixel(
    cameras,
    images,
    record: ImageRecord,
    click_xy: tuple[float, float],
    image_observations: dict[int, list[Observation]],
    session_point_clouds: dict[str, SessionPointCloud],
    source_session: str | None,
    max_distance_px: float,
) -> tuple[PixelLookupResult, list[PixelLookupResult]]:
    image = images[record.image_id]
    camera = cameras[image.camera_id]
    click = np.asarray(click_xy, dtype=np.float32)
    candidates = []

    if source_session is None:
        observations = image_observations.get(int(record.image_id), [])
        if observations:
            observation_xy = np.asarray([obs.xy for obs in observations], dtype=np.float32)
            distances = np.linalg.norm(observation_xy - click[None, :], axis=1)
            nearest_idx = int(np.argmin(distances))
            obs = observations[nearest_idx]
            candidates.append(
                PixelLookupResult(
                    point_id=int(obs.point_id),
                    source_session=record.sequence_name,
                    candidate_type="observed",
                    projected_xy=(float(obs.xy[0]), float(obs.xy[1])),
                    distance_px=float(distances[nearest_idx]),
                )
            )
        source_sessions = list(session_point_clouds)
    else:
        if source_session not in session_point_clouds:
            raise ValueError(
                f"Unknown --inspect-source-session '{source_session}'. "
                f"Choose from: {', '.join(session_point_clouds)}"
            )
        source_sessions = [source_session]

    for candidate_session in source_sessions:
        candidate = _nearest_projected_candidate(
            session_point_clouds[candidate_session],
            image,
            camera,
            click,
            candidate_session,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.distance_px,
            candidate.candidate_type != "observed",
            candidate.source_session,
            candidate.point_id,
        ),
    )
    if not candidates:
        raise ValueError(
            f"No observed or reprojected points were available for {record.image_name}."
        )

    selected = candidates[0]
    if selected.distance_px > max_distance_px:
        raise ValueError(
            f"Nearest point is {selected.distance_px:.1f}px from the requested pixel, "
            f"outside --inspect-pixel-max-distance={max_distance_px:.1f}px."
        )
    return selected, candidates


def _print_point_correspondences(
    result: PixelLookupResult,
    target_record: ImageRecord,
    click_xy: tuple[float, float],
    candidates: list[PixelLookupResult],
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
):
    print("")
    print("Pixel correspondence lookup")
    print(
        f"  target: {target_record.sequence_name}/{target_record.image_name} "
        f"pixel=({click_xy[0]:.1f}, {click_xy[1]:.1f})"
    )
    print(
        f"  selected: point_id={result.point_id} source={result.source_session} "
        f"type={result.candidate_type} projected=({result.projected_xy[0]:.1f}, "
        f"{result.projected_xy[1]:.1f}) distance={result.distance_px:.2f}px"
    )
    print("  nearest candidate per available layer:")
    for candidate in candidates:
        print(
            f"    {candidate.source_session:>8} {candidate.candidate_type:<11} "
            f"pid={candidate.point_id:<10} distance={candidate.distance_px:6.2f}px "
            f"xy=({candidate.projected_xy[0]:.1f}, {candidate.projected_xy[1]:.1f})"
        )

    observations = sorted(
        point_observations.get(result.point_id, []),
        key=lambda obs: (
            obs.sequence_name,
            image_catalog[int(obs.image_id)].timestamp_ns
            if int(obs.image_id) in image_catalog
            else 0,
            obs.image_name,
        ),
    )
    print("  COLMAP observation images:")
    if not observations:
        print("    none in the selected image catalog")
    for obs in observations:
        print(
            f"    {obs.sequence_name}/{obs.image_name} "
            f"pixel=({obs.xy[0]:.1f}, {obs.xy[1]:.1f})"
        )
    print("")


def _print_point_id_correspondences(
    point_ids: list[int],
    points3d,
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
):
    for point_id in point_ids:
        point = points3d.get(int(point_id))
        if point is None:
            print(f"POINT3D_ID {point_id}: not found")
            continue

        observations = sorted(
            point_observations.get(int(point_id), []),
            key=lambda obs: (
                obs.sequence_name,
                image_catalog[int(obs.image_id)].timestamp_ns
                if int(obs.image_id) in image_catalog
                else 0,
                obs.image_name,
            ),
        )
        print("")
        print(f"POINT3D_ID {int(point_id)}")
        print(
            f"  xyz=({float(point.xyz[0]):.4f}, {float(point.xyz[1]):.4f}, "
            f"{float(point.xyz[2]):.4f})"
        )
        print(f"  COLMAP observation images: {len(observations)}")
        if not observations:
            print("    none in the selected image catalog")
        for obs in observations:
            print(
                f"    {obs.sequence_name}/{obs.image_name} "
                f"pixel=({obs.xy[0]:.1f}, {obs.xy[1]:.1f})"
            )
        print("")


def _log_inspected_point_tracks(
    rr,
    cameras,
    images,
    points3d,
    point_observations: dict[int, list[Observation]],
    image_catalog: dict[int, ImageRecord],
    sequences: list[str],
    point_ids: list[int],
    max_observations_per_point: int | None,
    image_view_limit: int,
    ray_radius: float,
) -> list[str]:
    if not point_ids:
        return []

    session_color = _session_color_map(sequences)
    image_view_origins = []

    for point_id in point_ids:
        point = points3d.get(int(point_id))
        if point is None:
            print(f"Warning: --inspect-point-id={point_id} is not in the COLMAP model.")
            continue

        observations = [
            obs
            for obs in point_observations.get(int(point_id), [])
            if int(obs.image_id) in image_catalog
        ]
        observations = sorted(
            observations,
            key=lambda obs: (
                obs.sequence_name,
                image_catalog[int(obs.image_id)].timestamp_ns,
                obs.image_name,
            ),
        )
        total_observations = len(observations)
        if (
            max_observations_per_point is not None
            and total_observations > max_observations_per_point
        ):
            observations = observations[:max_observations_per_point]

        class_ids, keypoint_ids = _point_identity_components([int(point_id)])
        point_xyz = np.asarray(point.xyz, dtype=np.float32)
        base_entity = f"world/inspected_points/pid_{int(point_id)}"
        rr.log(
            base_entity,
            rr.Points3D(
                [point_xyz],
                colors=[[255, 255, 255, 255]],
                radii=[max(ray_radius * 3.0, 0.08)],
                labels=[f"POINT3D_ID {int(point_id)}"],
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
            ),
            static=True,
        )
        rr.log(
            base_entity,
            rr.AnyValues(
                point_id=int(point_id),
                total_observations=total_observations,
                logged_observations=len(observations),
                world_x=float(point_xyz[0]),
                world_y=float(point_xyz[1]),
                world_z=float(point_xyz[2]),
            ),
            static=True,
        )

        rays = []
        ray_colors = []
        status_lines = [
            f"# POINT3D_ID {int(point_id)}",
            "",
            f"- logged observations: `{len(observations)}` of `{total_observations}`",
            f"- xyz: `({point_xyz[0]:.4f}, {point_xyz[1]:.4f}, {point_xyz[2]:.4f})`",
            "",
        ]

        for obs_idx, obs in enumerate(observations):
            record = image_catalog[int(obs.image_id)]
            image = images[record.image_id]
            camera = cameras[image.camera_id]
            observation_entity = (
                f"{base_entity}/observations/"
                f"{obs_idx:03d}_{record.sequence_name}_{_safe_entity_name(record.image_name)}"
            )
            image_entity = f"{observation_entity}/image"
            color = session_color[record.sequence_name]

            rr.log(
                observation_entity,
                rr.Transform3D(
                    mat3x3=image.qvec2rotmat(),
                    translation=image.tvec,
                    relation=rr.TransformRelation.ChildFromParent,
                ),
                static=True,
            )
            rr.log(
                observation_entity,
                rr.Pinhole(
                    image_from_camera=_camera_matrix(camera),
                    resolution=[camera.width, camera.height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                static=True,
            )
            rr.log(image_entity, rr.EncodedImage(path=record.image_path), static=True)
            rr.log(
                f"{image_entity}/point",
                rr.Points2D(
                    [obs.xy],
                    labels=[f"pid={int(point_id)}"],
                    show_labels=True,
                    class_ids=class_ids,
                    keypoint_ids=keypoint_ids,
                    colors=[color],
                    radii=10.0,
                ),
                static=True,
            )
            rr.log(
                f"{image_entity}/point",
                rr.AnyValues(
                    point_id=int(point_id),
                    source_session=obs.sequence_name,
                    target_session=record.sequence_name,
                    target_image=record.image_name,
                    image_id=record.image_id,
                    session_frame=record.session_frame,
                    pixel_u=float(obs.xy[0]),
                    pixel_v=float(obs.xy[1]),
                    world_x=float(point_xyz[0]),
                    world_y=float(point_xyz[1]),
                    world_z=float(point_xyz[2]),
                ),
                static=True,
            )
            rr.log(
                f"{image_entity}/status",
                rr.TextDocument(
                    "# Point observation\n\n"
                    f"- point_id: `{int(point_id)}`\n"
                    f"- session: `{record.sequence_name}`\n"
                    f"- session_frame: `{record.session_frame}`\n"
                    f"- image: `{record.image_name}`\n"
                    f"- pixel: `({obs.xy[0]:.1f}, {obs.xy[1]:.1f})`",
                    media_type="text/markdown",
                ),
                static=True,
            )

            rays.append([_camera_center(image), point_xyz])
            ray_colors.append(color)
            status_lines.append(
                f"- `{record.sequence_name}` `{record.image_name}` "
                f"pixel `({obs.xy[0]:.1f}, {obs.xy[1]:.1f})`"
            )
            if len(image_view_origins) < image_view_limit:
                image_view_origins.append(image_entity)

        if rays:
            rr.log(
                f"{base_entity}/rays",
                rr.LineStrips3D(
                    rays,
                    colors=ray_colors,
                    radii=ray_radius,
                    labels=[
                        f"pid={int(point_id)} obs={idx}" for idx in range(len(rays))
                    ],
                    show_labels=False,
                ),
                static=True,
            )
        rr.log(
            f"{base_entity}/summary",
            rr.TextDocument("\n".join(status_lines), media_type="text/markdown"),
            static=True,
        )

    return image_view_origins


def _send_blueprint(
    rr,
    sequences: list[str],
    support_images_per_session: int,
    inspected_image_origins: list[str] | None = None,
    inspected_point_ids: list[int] | None = None,
    focused_inspection: bool = False,
):
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return

    right_views = []
    row_shares = []

    if not focused_inspection:
        right_views.append(
            rrb.Spatial2DView(
                origin="world/current_camera/image",
                name="Current image",
            )
        )
        row_shares.append(4)

    if not focused_inspection and support_images_per_session > 0:
        support_views = []
        for sequence_name in sequences:
            for slot_idx in range(support_images_per_session):
                support_views.append(
                    rrb.Spatial2DView(
                        origin=f"world/support_images/{sequence_name}/slot_{slot_idx}/image",
                        name=f"{sequence_name} support {slot_idx + 1}",
                    )
                )
        right_views.append(
            rrb.Horizontal(
                *support_views,
                name="Cross-session support images",
            )
        )
        row_shares.append(2)

    if inspected_image_origins:
        inspected_views = [
            rrb.Spatial2DView(
                origin=origin,
                name=origin.split("/")[-2],
            )
            for origin in inspected_image_origins
        ]
        right_views.append(
            rrb.Horizontal(
                *inspected_views,
                name="Inspected point observations",
            )
        )
        row_shares.append(2)

    summary_origin = "current/status"
    summary_name = "Image status"
    if focused_inspection and inspected_point_ids:
        summary_origin = f"world/inspected_points/pid_{int(inspected_point_ids[0])}/summary"
        summary_name = "Point observations"
    elif focused_inspection:
        summary_origin = "world/README"
        summary_name = "Inspection status"

    right_views.append(
        rrb.TextDocumentView(
            origin=summary_origin,
            name=summary_name,
        )
    )
    row_shares.append(1)

    layout = rrb.Horizontal(
        rrb.Spatial3DView(
            origin="world",
            name="3D reconstruction",
            line_grid=False,
        ),
        rrb.Vertical(
            *right_views,
            row_shares=row_shares,
        ),
        column_shares=[3, 2],
    )
    panels = [rrb.SelectionPanel(expanded=True)]
    if not focused_inspection:
        panels.append(rrb.TimePanel(expanded=True, timeline="frame"))

    rr.send_blueprint(
        rrb.Blueprint(
            layout,
            *panels,
            collapse_panels=False,
        ),
        make_active=True,
        make_default=True,
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
    parser.add_argument(
        "--max-images-per-session",
        type=int,
        default=25,
        help="Maximum timeline images per session. Use --all-images for the complete sequence.",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Log every sorted image on the timeline.",
    )
    parser.add_argument(
        "--log-camera-images",
        action="store_true",
        help=(
            "Also log each selected image and its 2D COLMAP observations under its "
            "static camera frustum. Use with --all-images when every frustum should "
            "open to an image and participate in point/image correspondence picking."
        ),
    )
    parser.add_argument(
        "--log-camera-reprojections",
        action="store_true",
        help=(
            "When --log-camera-images is enabled, also log cross-session reprojection "
            "overlays under each static camera image. This can be very large with "
            "--all-images unless --max-reprojected-points-per-session is set."
        ),
    )
    parser.add_argument(
        "--max-track-docs",
        type=int,
        default=0,
        help="Optional number of full point-track document entities to log. Default keeps the viewer light.",
    )
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
        "--max-reprojected-points-per-session",
        type=int,
        default=None,
        help="Optional deterministic cap for each source-session reprojection overlay.",
    )
    parser.add_argument(
        "--support-images-per-session",
        type=int,
        default=0,
        help=(
            "Show this many cross-session support images per session for the active image, "
            "ranked by shared COLMAP 3D point tracks. Default is 0 to keep the viewer light."
        ),
    )
    parser.add_argument(
        "--support-include-active-session",
        action="store_true",
        help="Also show same-session support images in the support-image row.",
    )
    parser.add_argument(
        "--show-rays",
        action="store_true",
        help="Draw capped camera-to-point rays for the current image.",
    )
    parser.add_argument(
        "--max-rays-per-session",
        type=int,
        default=50,
        help="Maximum rays to draw per source session when --show-rays is enabled.",
    )
    parser.add_argument(
        "--point-metadata-observation-limit",
        type=int,
        default=12,
        help="Number of image/pixel observations to include in each selectable 3D point summary.",
    )
    parser.add_argument(
        "--inspect-point-id",
        type=int,
        action="append",
        default=[],
        help=(
            "Focus the recording on this COLMAP POINT3D_ID. Can be repeated. "
            "Logs the point, every logged observation image for its track, and "
            "persistent camera-to-point rays under world/inspected_points."
        ),
    )
    parser.add_argument(
        "--inspect-image",
        type=str,
        default=None,
        help=(
            "Resolve a point directly from an image name. Must be used with "
            "--inspect-pixel. The resolved point is added to --inspect-point-id."
        ),
    )
    parser.add_argument(
        "--inspect-pixel",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        default=None,
        help="Pixel coordinate to look up in --inspect-image.",
    )
    parser.add_argument(
        "--inspect-source-session",
        type=str,
        default=None,
        help=(
            "Restrict --inspect-pixel lookup to reprojected points from this source "
            "session. Recommended when clicking a colored cross-session overlay."
        ),
    )
    parser.add_argument(
        "--inspect-pixel-max-distance",
        type=float,
        default=15.0,
        help="Maximum pixel distance allowed for automatic image/pixel point lookup.",
    )
    parser.add_argument(
        "--inspect-lookup-only",
        action="store_true",
        help=(
            "Print correspondences for --inspect-point-id or image/pixel lookup and "
            "exit without logging or opening Rerun."
        ),
    )
    parser.add_argument(
        "--inspect-with-context",
        action="store_true",
        help=(
            "Keep the full point clouds, camera timeline, and reprojection overlays "
            "when inspecting points. By default point inspection uses a lightweight "
            "focused recording."
        ),
    )
    parser.add_argument(
        "--max-inspect-observations-per-point",
        type=int,
        default=None,
        help="Optional cap on logged observation images for each --inspect-point-id.",
    )
    parser.add_argument(
        "--inspect-point-image-views",
        type=int,
        default=4,
        help="Maximum inspected-point observation images to place directly in the blueprint.",
    )
    parser.add_argument(
        "--inspect-ray-radius",
        type=float,
        default=0.03,
        help="World-space radius for persistent inspected-point rays.",
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
        help="Disable automatic loading of .ply files from --splat-dir when --show-splats is used.",
    )
    parser.add_argument(
        "--show-splats",
        action="store_true",
        help="Log precomputed Gaussian splat PLYs as toggleable radius/opacity point-proxy layers.",
    )
    parser.add_argument(
        "--max-splats-per-session",
        type=int,
        default=100000,
        help="Maximum splat centers to log per session. Use a negative value for no cap.",
    )
    parser.add_argument(
        "--splat-radius-scale",
        type=float,
        default=2.5,
        help="Multiply decoded Gaussian splat radii for easier inspection in Rerun.",
    )
    parser.add_argument(
        "--splat-min-radius",
        type=float,
        default=0.01,
        help="Clamp splat proxy radii to at least this world-space size. Use 0 to disable.",
    )
    parser.add_argument(
        "--splat-max-radius",
        type=float,
        default=0.08,
        help="Clamp splat proxy radii to at most this world-space size. Use a negative value to disable.",
    )
    parser.add_argument(
        "--splat-opacity-scale",
        type=float,
        default=1.4,
        help="Multiply decoded splat opacity alpha for the Rerun point proxy.",
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
    max_images_per_session = None if args.all_images else args.max_images_per_session
    max_splats_per_session = (
        None if args.max_splats_per_session < 0 else args.max_splats_per_session
    )
    splat_max_radius = None if args.splat_max_radius < 0 else args.splat_max_radius

    cameras, images, points3d = read_model(str(model_path), ext="")
    rgb_lookup = _image_lookup(rgb_df)
    image_catalog, catalog_missing_images = _prepare_image_catalog(
        rgb_df,
        images,
        image_root,
        sequences,
        strict_images=args.strict_images,
    )
    image_records, missing_images = _prepare_image_records(
        rgb_df,
        images,
        rgb_lookup,
        image_root,
        sequences,
        max_images_per_session=max_images_per_session,
        strict_images=args.strict_images,
    )
    direct_point_id_path = (
        bool(args.inspect_point_id)
        and args.inspect_image is None
        and (args.inspect_lookup_only or not args.inspect_with_context)
    )
    point_observations, image_observations, point_sequences, skipped = _collect_observations(
        points3d,
        images,
        rgb_lookup,
        point_ids=args.inspect_point_id if direct_point_id_path else None,
    )

    if (args.inspect_image is None) != (args.inspect_pixel is None):
        raise ValueError("--inspect-image and --inspect-pixel must be provided together.")
    pixel_lookup_requested = args.inspect_image is not None
    if args.inspect_lookup_only and not pixel_lookup_requested and not args.inspect_point_id:
        raise ValueError(
            "--inspect-lookup-only requires --inspect-point-id or "
            "--inspect-image with --inspect-pixel."
        )
    if args.inspect_lookup_only and not pixel_lookup_requested:
        _print_point_id_correspondences(
            args.inspect_point_id,
            points3d,
            point_observations,
            image_catalog,
        )
        return

    focused_point_id_mode = bool(args.inspect_point_id) and not args.inspect_with_context
    needs_session_point_clouds = pixel_lookup_requested or not focused_point_id_mode
    session_point_clouds = {}
    if needs_session_point_clouds:
        session_point_clouds = _build_session_point_clouds(
            points3d,
            point_observations,
            point_sequences,
            sequences,
            alpha_tint=args.alpha_tint,
            track_summary_limit=args.point_metadata_observation_limit,
        )

    if args.inspect_image is not None and args.inspect_pixel is not None:
        target_record = _find_image_record(image_catalog, args.inspect_image)
        selected, candidates = _lookup_point_from_pixel(
            cameras,
            images,
            target_record,
            click_xy=(float(args.inspect_pixel[0]), float(args.inspect_pixel[1])),
            image_observations=image_observations,
            session_point_clouds=session_point_clouds,
            source_session=args.inspect_source_session,
            max_distance_px=args.inspect_pixel_max_distance,
        )
        _print_point_correspondences(
            selected,
            target_record,
            click_xy=(float(args.inspect_pixel[0]), float(args.inspect_pixel[1])),
            candidates=candidates,
            point_observations=point_observations,
            image_catalog=image_catalog,
        )
        if selected.point_id not in args.inspect_point_id:
            args.inspect_point_id.append(selected.point_id)
        if args.inspect_lookup_only:
            return

    focused_inspection = bool(args.inspect_point_id) and not args.inspect_with_context

    rr.init(args.application_id, spawn=args.output is None)
    if args.output:
        Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.output)

    _log_world_header(rr, dataset, subset)
    logged_images = defaultdict(int)
    reprojected_points = defaultdict(int)
    logged_tracks = 0

    if not focused_inspection:
        _log_session_points(
            rr,
            session_point_clouds,
            point_radius=args.point_radius,
        )
        _log_static_cameras(rr, cameras, images, image_records)
        if args.log_camera_images:
            _log_static_camera_images(
                rr,
                cameras,
                images,
                image_records,
                image_observations,
                session_point_clouds,
                sequences,
                log_reprojected_points=args.log_camera_reprojections,
                reprojected_point_radius=args.reprojected_point_radius,
                max_reprojected_points_per_session=args.max_reprojected_points_per_session,
            )
        logged_images, reprojected_points = _log_current_image_timeline(
            rr,
            cameras,
            images,
            image_records,
            image_observations,
            point_observations,
            image_catalog,
            session_point_clouds,
            sequences,
            log_reprojected_points=not args.no_reprojected_points,
            reprojected_point_radius=args.reprojected_point_radius,
            show_rays=args.show_rays,
            max_rays_per_session=args.max_rays_per_session,
            max_reprojected_points_per_session=args.max_reprojected_points_per_session,
            support_images_per_session=args.support_images_per_session,
            include_active_support_session=args.support_include_active_session,
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
    if args.show_splats and not args.no_auto_splats:
        splat_paths.extend(_default_splat_paths(splat_dir, sequences))
    _log_gaussian_splats(
        rr,
        splat_paths,
        splat_radius_scale=args.splat_radius_scale,
        splat_min_radius=args.splat_min_radius,
        splat_max_radius=splat_max_radius,
        splat_opacity_scale=args.splat_opacity_scale,
        max_splats_per_session=max_splats_per_session,
    )
    inspected_image_origins = _log_inspected_point_tracks(
        rr,
        cameras,
        images,
        points3d,
        point_observations,
        image_catalog,
        sequences,
        point_ids=args.inspect_point_id,
        max_observations_per_point=args.max_inspect_observations_per_point,
        image_view_limit=args.inspect_point_image_views,
        ray_radius=args.inspect_ray_radius,
    )
    _send_blueprint(
        rr,
        sequences,
        support_images_per_session=args.support_images_per_session,
        inspected_image_origins=inspected_image_origins,
        inspected_point_ids=args.inspect_point_id,
        focused_inspection=focused_inspection,
    )

    print(f"Logged sessions: {', '.join(sequences)}")
    print(f"Logged timeline images: {0 if focused_inspection else len(image_records)}")
    if splat_paths:
        print(f"Logged Gaussian splat assets: {len(splat_paths)}")
    if args.log_camera_images and not focused_inspection:
        print(f"Logged static camera images: {len(image_records)}")
    if inspected_image_origins:
        print(f"Logged inspected point image views: {len(inspected_image_origins)}")
    print(f"Logged images per session: {dict(logged_images)}")
    if reprojected_points:
        print(f"Logged reprojected 2D points by source session: {dict(reprojected_points)}")
    print(f"Logged point track documents: {logged_tracks}")
    if skipped:
        print(f"Skipped {skipped} point observations without image/session metadata.")
    if missing_images:
        print(f"Skipped {missing_images} missing image files.")
    if catalog_missing_images:
        print(f"Skipped {catalog_missing_images} missing catalog image files.")


if __name__ == "__main__":
    main()
