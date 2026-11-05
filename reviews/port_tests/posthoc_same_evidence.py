"""Same-evidence post-hoc baseline (R13.3 / R13.5 / R13.14): independent per-session reconstructions aligned after the
fact with the same cross-session evidence as a finished "Ours" (hybrid) run.

  1. Each session reconstructed alone, in its own VSLAM-LAB run with its own database (--survey, e.g.
     configs/exp_x5x6_ssk_subsets_posthoc.yaml: exhaustive SIFT, COLMAP incremental mapper, the session's fixed camera; NumRuns 3
     = three repeats, picked with --repeat). Nothing is taken from the joint run.
  2. The sessions' databases merged (`colmap database_merger`), then the Ours run's own cross-session matching on it
     (posthoc_cross_matching.py -> colmap_matcher.run_hybrid: the same VPR pairs from the same D.npy and threshold, the
     same learned features and matcher, COLMAP geometric verification). Built once per sequence and repeat, from that
     repeat's per-session databases (SIFT extraction is not bit-identical between runs):
     <VSLAM-LAB-Evaluation>/<survey exp>/<DATASET>/merged_<sequence>_r<k>/ for --repeat k (k = 0, 1, 2), matching the
     run queue's output folder posthoc_r<k> and the per-session model colmap_0000k
  3. Each session model moved into the merged database's ids (by image name) -> posthoc/session_<s>/0, also the input
     of posthoc_registration.py (ICP, BUFFER-X). 3D-3D correspondences from the verified cross-session matches: each
     matched keypoint takes its 3D point in its own session model (a learned keypoint, which a session does not
     triangulate, takes the 3D point of the nearest triangulated keypoint within NEAR_PX in the same image).
  4. One Sim(3) (rotation, translation, scale) per session onto a reference session (the one with most verified
     pairs): Umeyama inside RANSAC over the correspondences, refit on the inliers.
  5. Row "Sim(3)": sessions transformed and merged, each kept rigid  -> posthoc/sim3/
  6. Row "Sim(3) + BA": all matches (cross-session included) triangulated on the merged model with poses fixed, then
     bundle adjustment of all poses and points (intrinsics as in the run)  -> posthoc/sim3_ba/
  7. posthoc/summary.yaml: per-session runs and times, matching stats, registration per session, Sim(3) fits, runtime
     per stage, and the reprojection error of the annotated clicks (evaluate_rpe.score_clicks) for both rows and Ours.

  With --min_cross_inliers N, cross-session pairs with fewer than N verified inliers are deleted from posthoc/database.db
  after step 2 (the full datasets use 100, as their Ours re-map), so steps 4-6 use the same evidence as that Ours run.

usage: python posthoc_same_evidence.py <run folder> --survey <per-session exp> [--repeat k] [--min_cross_inliers N]
                                       [--no-ba] [--selftest]
  <run folder> = <VSLAM-LAB-Evaluation>/<exp>/<DATASET>/<sequence> of a finished hybrid run
  --selftest   check steps 3-4 instead: split Ours into sessions, move each by a known Sim(3), recover it
"""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "baselines" / "VSLAM_LAB_port"
sys.path.insert(0, str(PORT))
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION  # noqa: E402

from evaluate_rpe import score_clicks  # noqa: E402

PAIR_ID_FACTOR = 2147483647  # COLMAP pair_id = image_id1 * factor + image_id2 (image_id1 < image_id2)
MIN_INLIERS = 15             # COLMAP's TwoViewGeometry.min_num_inliers default
MAX_PAIRS_PER_SESSION_PAIR = 1000  # strongest verified pairs used to fit each Sim(3) (7 unknowns); BA uses every match
NEAR_PX = 2.0                # a keypoint without a 3D point borrows the nearest triangulated keypoint's within this
RANSAC_ITERATIONS = 2000
RANSAC_SCORE_SAMPLE = 20000  # correspondences scored per RANSAC hypothesis (all of them in the final refit)
INLIER_FRACTION = 0.02       # inlier threshold as a fraction of the reference session's spread of 3D points


def colmap(*args):
    subprocess.run(["pixi", "run", "--manifest-path", str(PORT / "pixi.toml"), "-e", "colmap", "colmap", *map(str, args)],
                   check=True, cwd=PORT)


# ---------------------------------------------------------------------------------------------------------------------
# Rotations (COLMAP quaternions are qw qx qy qz, poses are world-to-camera)
def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def rotation_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def umeyama(src, dst):
    """Least-squares similarity (s, R, t) with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


# ---------------------------------------------------------------------------------------------------------------------
# COLMAP text model: cameras / rigs untouched, poses in frames.txt and images.txt, points renumbered on merge.
# Large parts stay as raw text (each image's 2D points line, each 3D point's colour/error/track) so that reading and
# writing a full-dataset model stays fast.
class Model:
    def __init__(self):
        self.cameras, self.rigs, self.frames, self.images = {}, {}, {}, {}
        self.point_ids, self.xyz, self.point_rest = np.zeros(0, np.int64), np.zeros((0, 3)), []

    @staticmethod
    def read(folder: Path, poses_only: bool = False) -> "Model":
        m = Model()

        def rows(name):
            return [line.split() for line in open(folder / name) if line.strip() and not line.startswith("#")]
        m.cameras = {int(t[0]): t for t in rows("cameras.txt")}
        m.rigs = {int(t[0]): t for t in rows("rigs.txt")}
        for t in rows("frames.txt"):
            m.frames[int(t[0])] = {"rig": t[1], "q": np.array(t[2:6], float), "t": np.array(t[6:9], float), "tail": t[9:]}
        lines = [line.rstrip("\n") for line in open(folder / "images.txt") if not line.startswith("#")]
        for i in range(0, len(lines) - 1, 2):
            t = lines[i].split()
            if t:
                m.images[int(t[0])] = {"q": np.array(t[1:5], float), "t": np.array(t[5:8], float), "cam": t[8],
                                       "name": t[9], "pts": "" if poses_only else lines[i + 1]}
        if not poses_only:
            heads, rests = [], []
            for line in open(folder / "points3D.txt"):
                if line.strip() and not line.startswith("#"):
                    t = line.split(maxsplit=4)
                    heads.append(t[:4])
                    rests.append(t[4].rstrip("\n"))
            heads = np.array(heads, float).reshape(-1, 4)
            m.point_ids, m.xyz, m.point_rest = heads[:, 0].astype(np.int64), heads[:, 1:], rests
        return m

    def pose(self, image_id):
        img = self.images[image_id]
        return quat_to_rot(img["q"]), img["t"]

    def subset(self, image_ids) -> "Model":
        """Images (and their frames) in image_ids; the points stay, so the images' observations still resolve."""
        m = Model()
        m.cameras, m.rigs = dict(self.cameras), dict(self.rigs)
        m.point_ids, m.xyz, m.point_rest = self.point_ids, self.xyz, self.point_rest
        m.images = {i: dict(self.images[i]) for i in image_ids}
        m.frames = {f: dict(fr) for f, fr in self.frames.items() if any(int(d) in m.images for d in fr["tail"][3::3])}
        return m

    def keypoint_points(self, image_id, keypoint_idx):
        """3D point (NaN if none) behind each keypoint index of an image (database keypoint order); a keypoint
        without one takes the 3D point of the nearest triangulated keypoint within NEAR_PX."""
        cache = self.__dict__.setdefault("_observations", {})
        if image_id not in cache:
            obs = np.array(self.images[image_id]["pts"].split(), float).reshape(-1, 3)
            cache[image_id] = (obs[:, :2], obs[:, 2].astype(np.int64))
        xy, pids = cache[image_id]
        pid = pids[keypoint_idx].copy()
        missing = pid < 0
        if missing.any() and (pids >= 0).any():
            triangulated = np.flatnonzero(pids >= 0)
            dist, nearest = cKDTree(xy[triangulated]).query(xy[keypoint_idx[missing]], distance_upper_bound=NEAR_PX)
            found = np.isfinite(dist)
            pid[np.flatnonzero(missing)[found]] = pids[triangulated[nearest[found]]]
        out = np.full((len(pid), 3), np.nan)
        order = np.argsort(self.point_ids)
        ok = pid >= 0
        out[ok] = self.xyz[order[np.searchsorted(self.point_ids, pid[ok], sorter=order)]]
        return out

    def transformed(self, s, R, t) -> "Model":
        """Apply X' = s R X + t to the world frame (points, frame and image poses)."""
        m = Model()
        m.cameras, m.rigs, m.point_ids, m.point_rest = self.cameras, self.rigs, self.point_ids, self.point_rest

        def move(pose):
            Rc, tc = quat_to_rot(pose["q"]), pose["t"]
            out = dict(pose)
            out["q"], out["t"] = rot_to_quat(Rc @ R.T), s * tc - Rc @ R.T @ t
            return out
        m.frames = {k: move(v) for k, v in self.frames.items()}
        m.images = {k: move(v) for k, v in self.images.items()}
        m.xyz = s * self.xyz @ R.T + t
        return m

    def renumbered(self, ids: dict[str, tuple[int, int, int, int]]) -> "Model":
        """The model with the image, camera, frame and rig ids of another database (ids: image name -> (image_id,
        camera_id, frame_id, rig_id)), e.g. a per-session model moved into the merged database. Keypoint indices and
        3D point ids are unchanged."""
        m = Model()
        m.point_ids, m.xyz = self.point_ids, self.xyz
        image_of = {old: ids[img["name"]][0] for old, img in self.images.items()}
        camera_of = {img["cam"]: str(ids[img["name"]][1]) for img in self.images.values()}
        m.images = {image_of[old]: {**img, "cam": camera_of[img["cam"]]} for old, img in self.images.items()}
        rig_of = {}
        for frame in self.frames.values():            # tail: NUM_DATA_IDS, then (SENSOR_TYPE, SENSOR_ID, DATA_ID) triples
            tail = list(frame["tail"])
            data = [int(d) for d in tail[3::3]]
            _, _, new_frame, new_rig = ids[self.images[data[0]]["name"]]
            rig_of[frame["rig"]] = str(new_rig)
            tail[2::3] = [camera_of[c] for c in tail[2::3]]
            tail[3::3] = [str(image_of[d]) for d in data]
            m.frames[new_frame] = {**frame, "rig": str(new_rig), "tail": tail}
        for old, t in self.cameras.items():
            if str(old) in camera_of:
                m.cameras[int(camera_of[str(old)])] = [camera_of[str(old)], *t[1:]]
        for old, t in self.rigs.items():
            if str(old) in rig_of:
                if t[1] != "1":
                    raise SystemExit(f"rig {old} has {t[1]} sensors; only single-camera rigs are renumbered")
                m.rigs[int(rig_of[str(old)])] = [rig_of[str(old)], t[1], t[2], camera_of[t[3]], *t[4:]]
        m.point_rest = []
        for rest in self.point_rest:                  # "R G B ERROR IMAGE_ID POINT2D_IDX ..." -> new image ids
            t = rest.split()
            t[4::2] = [str(image_of[int(i)]) for i in t[4::2]]
            m.point_rest.append(" ".join(t))
        return m

    @staticmethod
    def merged(models) -> "Model":
        """One model from several sessions; each session's 3D point IDs are shifted past the previous ones."""
        out, offset, ids, xyz = Model(), 0, [], []
        for m in models:
            out.cameras.update(m.cameras)
            out.rigs.update(m.rigs)
            out.frames.update(m.frames)
            for k, img in m.images.items():
                tokens = img["pts"].split()
                if offset and tokens:
                    pids = np.array(tokens[2::3], np.int64)
                    tokens[2::3] = np.where(pids >= 0, pids + offset, -1).astype(str).tolist()
                out.images[k] = {**img, "pts": " ".join(tokens)}
            ids.append(m.point_ids + offset)
            xyz.append(m.xyz)
            out.point_rest += m.point_rest
            offset += int(m.point_ids.max()) if m.point_ids.size else 0
        out.point_ids, out.xyz = np.concatenate(ids), np.concatenate(xyz)
        return out

    def write(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)

        def fmt(a):
            return " ".join(f"{v:.17g}" for v in a)
        (folder / "cameras.txt").write_text("".join(" ".join(t) + "\n" for t in self.cameras.values()))
        (folder / "rigs.txt").write_text("".join(" ".join(t) + "\n" for t in self.rigs.values()))
        (folder / "frames.txt").write_text("".join(
            f"{k} {f['rig']} {fmt(f['q'])} {fmt(f['t'])} {' '.join(f['tail'])}\n" for k, f in self.frames.items()))
        (folder / "images.txt").write_text("".join(
            f"{k} {fmt(i['q'])} {fmt(i['t'])} {i['cam']} {i['name']}\n{i['pts']}\n" for k, i in self.images.items()))
        (folder / "points3D.txt").write_text("".join(
            f"{k} {fmt(x)} {rest}\n" for k, x, rest in zip(self.point_ids, self.xyz, self.point_rest)))


# ---------------------------------------------------------------------------------------------------------------------
# Cross-session evidence and Sim(3)
def cross_session_matches(database: Path, session_of: dict[int, str], registered: set[int]):
    """[(image_a, image_b, verified matches (n, 2) keypoint indices)] for the MAX_PAIRS_PER_SESSION_PAIR verified
    cross-session pairs with most matches per pair of sessions, both images registered."""
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    by_sessions = {}
    for pair_id, rows in db.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows >= ?", (MIN_INLIERS,)):
        b = pair_id % PAIR_ID_FACTOR
        a = (pair_id - b) // PAIR_ID_FACTOR
        if a in registered and b in registered and session_of[a] != session_of[b]:
            by_sessions.setdefault(tuple(sorted((session_of[a], session_of[b]))), []).append((rows, pair_id, a, b))
    out = []
    for pairs in by_sessions.values():
        for rows, pair_id, a, b in sorted(pairs, reverse=True)[:MAX_PAIRS_PER_SESSION_PAIR]:
            data = db.execute("SELECT data FROM two_view_geometries WHERE pair_id = ?", (pair_id,)).fetchone()[0]
            out.append((a, b, np.frombuffer(data, np.uint32).reshape(rows, 2).astype(np.int64)))
    return out


def filter_cross_session_pairs(database: Path, session_of: dict[int, str], min_inliers: int) -> dict:
    """Delete the cross-session two_view_geometries with fewer than min_inliers inliers (the same rule as the Ours re-map,
    Baselines/colmap/filter_cross_session_pairs.py), so neither the Sim(3) nor the triangulation sees them.
    Returns the kept and removed pairs per session pair."""
    kept, removed, drop = {}, {}, []
    with closing(sqlite3.connect(database)) as db:
        for pair_id, rows in db.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0").fetchall():
            b = pair_id % PAIR_ID_FACTOR
            a = (pair_id - b) // PAIR_ID_FACTOR
            if session_of.get(a) == session_of.get(b):
                continue
            key = "-".join(sorted((session_of[a], session_of[b])))
            if rows < min_inliers:
                removed[key] = removed.get(key, 0) + 1
                drop.append((pair_id,))
            else:
                kept[key] = kept.get(key, 0) + 1
        db.executemany("DELETE FROM two_view_geometries WHERE pair_id = ?", drop)
        db.commit()
    return {"min_inliers": min_inliers, "cross_session_pairs_kept": kept, "cross_session_pairs_removed": removed}


def estimate_sim3(model_s: Model, model_t: Model, matches, rng):
    """(s, R, t, info) with X_S = s R X_T + t from the 3D points behind the verified matches between S and T."""
    xs, xt = [], []
    for a, b, m in matches:
        if a in model_t.images:                       # orient every pair as (a in S, b in T)
            a, b, m = b, a, m[:, ::-1]
        xs.append(model_s.keypoint_points(a, m[:, 0]))
        xt.append(model_t.keypoint_points(b, m[:, 1]))
    xs, xt = np.concatenate(xs), np.concatenate(xt)
    ok = np.isfinite(xs).all(1) & np.isfinite(xt).all(1)
    xs, xt = np.unique(np.hstack([xs[ok], xt[ok]]), axis=0).reshape(-1, 2, 3).transpose(1, 0, 2)  # one per 3D-3D pair
    if len(xs) < 3:
        raise SystemExit(f"only {len(xs)} 3D-3D correspondences: cannot align")

    tau = INLIER_FRACTION * np.median(np.linalg.norm(xs - np.median(xs, 0), axis=1))
    score = rng.choice(len(xs), min(RANSAC_SCORE_SAMPLE, len(xs)), replace=False)
    best = (-1, None)
    for _ in range(RANSAC_ITERATIONS):
        s, R, t = umeyama(xt[sample := rng.choice(len(xs), 3, replace=False)], xs[sample])
        if not np.isfinite(s) or s <= 0:
            continue
        n_in = (np.linalg.norm(s * xt[score] @ R.T + t - xs[score], axis=1) < tau).sum()
        if n_in > best[0]:
            best = (n_in, (s, R, t))
    s, R, t = best[1]
    inliers = np.linalg.norm(s * xt @ R.T + t - xs, axis=1) < tau
    s, R, t = umeyama(xt[inliers], xs[inliers])
    info = {"pairs": len(matches), "correspondences": int(len(xs)), "inliers": int(inliers.sum()),
            "scale": float(s), "inlier_threshold": float(tau)}
    return s, R, t, info


def align_sessions(sessions: dict[str, Model], matches, rng):
    """Reference = session with most verified pairs; every other session aligned directly onto it."""
    pair_count = {name: sum((a in m.images) or (b in m.images) for a, b, _ in matches) for name, m in sessions.items()}
    ref = max(pair_count, key=pair_count.get)
    transforms = {ref: (1.0, np.eye(3), np.zeros(3), {"reference": True, "pairs": pair_count[ref]})}
    for name, m in sessions.items():
        if name != ref:
            between = [p for p in matches if (p[0] in m.images and p[1] in sessions[ref].images)
                       or (p[1] in m.images and p[0] in sessions[ref].images)]
            transforms[name] = estimate_sim3(sessions[ref], m, between, rng)
    return ref, transforms


# ---------------------------------------------------------------------------------------------------------------------
def merged_database(databases: list[Path], out_db: Path) -> float:
    """One database from the per-session runs' databases (`colmap database_merger`, pairwise): images, cameras,
    rigs / frames, SIFT features and within-session matches, with image ids renumbered. Returns the merge time in s."""
    out_db.parent.mkdir(parents=True, exist_ok=True)
    # COLMAP opens its inputs read-write (a WAL checkpoint updates their mtime), so it merges copies and the
    # per-session runs stay untouched. The backup API reads through the read-only connection, WAL contents included.
    # The copies are bookkeeping, not part of the method, so they are not timed
    copies = []
    for k, database in enumerate(databases):
        copy = out_db.with_name(f"input_{k}.db")
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as src, closing(sqlite3.connect(copy)) as dst:
            src.backup(dst)
        copies.append(copy)
    start = time.perf_counter()
    current, intermediates = copies[0], []
    for k, database in enumerate(copies[1:], 1):
        target = out_db if k == len(copies) - 1 else out_db.with_name(f"merge_{k}.db")
        colmap("database_merger", "--database_path1", current, "--database_path2", database,
               "--merged_database_path", target)
        intermediates.append(target)
        current = target
    elapsed = time.perf_counter() - start
    for path in intermediates[:-1] + copies:
        path.unlink()
    return elapsed


def cross_matching(database: Path, sequence_path: Path, run: Path):
    """The cross-session matching of the Ours run (same VPR pairs, features, matcher, verification), on `database`."""
    subprocess.run(["pixi", "run", "--manifest-path", str(PORT / "pixi.toml"), "-e", "colmap", "python",
                    str(Path(__file__).with_name("posthoc_cross_matching.py")), "--database", str(database),
                    "--sequence_path", str(sequence_path), "--rgb_csv", str(run / "rgb_exp.csv"),
                    "--run_report", str(run / "00000_run_report.yaml")], check=True, cwd=PORT)


def database_ids(database: Path) -> dict[str, tuple[int, int, int, int]]:
    """image name -> (image_id, camera_id, frame_id, rig_id) in a COLMAP 4.x database."""
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    rows = db.execute("SELECT i.name, i.image_id, i.camera_id, d.frame_id, f.rig_id FROM images i "
                      "JOIN frame_data d ON d.data_id = i.image_id AND d.sensor_type = 0 "
                      "JOIN frames f ON f.frame_id = d.frame_id").fetchall()
    return {Path(name).name: (image_id, camera_id, frame_id, rig_id) for name, image_id, camera_id, frame_id, rig_id in rows}


def pad_to_database_keypoints(model: Model, database: Path):
    """Append to each image's 2D points the database keypoints the model does not list (the learned keypoints the
    cross-session matcher added after the SIFT ones), without a 3D point, so model and database agree index for index."""
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    for image_id, img in model.images.items():
        rows, cols, data = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()
        xy = np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2]
        listed = len(img["pts"].split()) // 3
        if rows > listed:
            img["pts"] = " ".join([img["pts"], *(f"{x:.17g} {y:.17g} -1" for x, y in xy[listed:])]).strip()


def keypoint_counts(database: Path) -> dict[str, int]:
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    return {Path(n).name: r for n, r in db.execute("SELECT i.name, k.rows FROM images i JOIN keypoints k USING(image_id)")}


def best_submodel(session_dir: Path) -> Path:
    """The sub-model with most registered images in a session's mapper output (already converted to TXT)."""
    subs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return max(subs, key=lambda sub: len(Model.read(sub, poses_only=True).images))


def evaluate(model_dir, csvs):
    if not csvs:
        return None
    errors, skipped, clicks, _ = score_clicks(str(model_dir), [str(c) for c in csvs])
    return {"rpe_median_px": round(float(np.median(errors)), 3) if errors.size else None,
            "n_scored": int(errors.size), "n_clicks": clicks, "not_scored": skipped}


def selftest(run: Path, session_of, rng):
    joint = Model.read(run / "colmap_00000")
    sessions = {}
    truth = {}
    for name in sorted(set(session_of.values())):
        sub = joint.subset([i for i in joint.images if session_of[i] == name])
        angle = rng.uniform(10, 40)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = Rotation.from_rotvec(axis * np.radians(angle)).as_matrix()
        s, t = rng.uniform(0.3, 3.0), rng.normal(size=3) * 5
        sessions[name] = sub.transformed(s, R, t)
        truth[name] = (s, R, t)                       # session frame = s R joint + t
    matches = cross_session_matches(run / "colmap_00000" / "colmap_database.db", session_of, set(joint.images))
    ref, est = align_sessions(sessions, matches, rng)
    s_r, R_r, t_r = truth[ref]
    for name, (s, R, t, info) in est.items():
        if name == ref:
            continue
        s_n, R_n, t_n = truth[name]
        # true map from session `name` frame to the reference frame
        s_true, R_true = s_r / s_n, R_r @ R_n.T
        t_true = t_r - s_true * R_true @ t_n
        print(f"{name} -> {ref}: rotation error {rotation_angle_deg(R_true.T @ R):.3f} deg, "
              f"scale ratio {s / s_true:.4f}, translation error {np.linalg.norm(t - t_true) / s_r:.4f} (joint units), "
              f"pairs {info['pairs']}, inliers {info['inliers']} / {info['correspondences']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="<VSLAM-LAB-Evaluation>/<exp>/<DATASET>/<sequence> of a finished hybrid run")
    parser.add_argument("--survey", default=None, help="experiment of the independent per-session runs (exp_x5x6_ssk_subsets_posthoc)")
    parser.add_argument("--repeat", type=int, default=0, help="which per-session run (colmap_0000<k>) to align")
    parser.add_argument("--survey_sequences", nargs="*", default=None,
                        help="session=sequence, e.g. ssk16=ssk16-s05 (default: <session>-<suffix of the run's sequence>)")
    parser.add_argument("--min_cross_inliers", type=int, default=None,
                        help="drop cross-session pairs with fewer verified inliers (100 = the full-dataset Ours rule)")
    parser.add_argument("--no-ba", action="store_true", help="stop after the Sim(3) row")
    parser.add_argument("--selftest", action="store_true", help="recover known Sim(3)s from the joint model instead")
    args = parser.parse_args()

    run = args.run.resolve()
    dataset, sequence = run.parent.name, run.name
    rng = np.random.default_rng(0)
    rgb = pd.read_csv(run / "rgb_exp.csv")
    session_by_name = {Path(p).name: s for p, s in rgb[["path_rgb_0", "sequence_name"]].values}
    database = run / "colmap_00000" / "colmap_database.db"
    # immutable: the finished Ours database may sit in a read-only folder (imported RA-L runs link it), where a WAL-mode
    # database cannot get the -shm file even a read-only open needs
    db = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    # by file name: imported RA-L databases store "../../<session>/rgb_0/<frame>" (import_ral_eiffel.py)
    session_of = {i: session_by_name[Path(n).name] for i, n in db.execute("SELECT image_id, name FROM images")}
    if args.selftest:
        return selftest(run, session_of, rng)
    if not args.survey:
        raise SystemExit("--survey <experiment of the independent per-session runs> is required (see --help)")

    report = yaml.safe_load((run / "00000_run_report.yaml").read_text())
    refine = int(report["configuration"]["mapper"]["optimize_intrinsics"])
    sequence_path = VSLAMLAB_BENCHMARK / dataset / sequence
    rgb_path = sequence_path / Path(rgb["path_rgb_0"].iloc[0]).parent   # rgb_0, or refrax_0 for refraction-corrected runs
    csvs = sorted((ROOT / "reprojection" / dataset / sequence).glob("evaluation_points_*.csv"))
    timing, summary = {}, {"run": str(run), "survey": args.survey, "repeat": args.repeat, "rows": {}}

    # 1. The independent per-session runs: one VSLAM-LAB run per session, each with its own database
    names = sorted(set(session_by_name.values()))
    suffix = sequence.split("-", 1)[1]
    survey_sequence = dict(s.split("=") for s in args.survey_sequences) if args.survey_sequences else \
        {name: f"{name}-{suffix}" for name in names}
    survey_root = VSLAMLAB_EVALUATION / args.survey / dataset
    survey_runs = {name: survey_root / survey_sequence[name] for name in names}
    # Each repeat merges its own per-session databases: SIFT extraction is not bit-identical between runs, so a
    # repeat's models only match the keypoints of its own databases
    survey_dbs = [survey_runs[n] / f"colmap_{args.repeat:05d}" / "colmap_database.db" for n in names]
    summary["per_session_runs"] = {n: str(survey_runs[n]) for n in names}
    summary["per_session_colmap_s"] = {
        n: yaml.safe_load((survey_runs[n] / f"{args.repeat:05d}_run_report.yaml").read_text())["runtime_s"]["total_colmap_pipeline"]
        for n in names}

    # 2. Merged database + the Ours run's cross-session matching on it, built once per sequence and repeat
    # Named like the run queue's output folders: --repeat k -> per-session colmap_0000k, posthoc_r<k>, merged_<seq>_r<k>
    merged_dir = survey_root / f"merged_{sequence}_r{args.repeat}"
    summary["per_session_model"] = f"colmap_{args.repeat:05d}"
    summary["posthoc_folder"] = f"posthoc_r{args.repeat}"
    summary["merged_database"] = str(merged_dir)
    merged_db = merged_dir / "database.db"
    merge_time = merged_dir / "merge_time_s.json"    # kept with the database, so a re-run that reuses it still reports it
    if not (merged_dir / "hybrid_matcher_stats.json").exists() or not merge_time.exists():
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        elapsed = merged_database(survey_dbs, merged_db)
        start = time.perf_counter()
        cross_matching(merged_db, sequence_path, run)
        merge_time.write_text(json.dumps({"merge_and_cross_session_matching": elapsed + time.perf_counter() - start}))
    timing.update(json.loads(merge_time.read_text()))
    stats = json.loads((merged_dir / "hybrid_matcher_stats.json").read_text())
    summary["cross_session_matching"] = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    out = run / "posthoc"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    work_db = out / "database.db"
    shutil.copy2(merged_db, work_db)                  # the merged database is never modified
    ids = database_ids(work_db)
    session_of = {ids[n][0]: s for n, s in session_by_name.items() if n in ids}
    # Same cross-session rule as the Ours run, applied to the working copy only (the merged matching stays reusable)
    if args.min_cross_inliers:
        summary["cross_session_filter"] = filter_cross_session_pairs(work_db, session_of, args.min_cross_inliers)

    # 3. Each session's own model, moved into the merged database's ids
    sessions = {}
    for name in names:
        model = Model.read(survey_runs[name] / f"colmap_{args.repeat:05d}").renumbered(ids)
        pad_to_database_keypoints(model, work_db)
        foreign = {session_of[i] for i in model.images} - {name}   # independent: only its own images
        assert not foreign, f"session {name} model contains images of {foreign}"
        model.write(out / f"session_{name}" / "0")
        sessions[name] = model
    summary["sessions"] = {n: f"{len(m.images)} / {sum(1 for s in session_by_name.values() if s == n)}" for n, m in sessions.items()}

    # 4-5. One Sim(3) per session onto the reference session, then the rigidly merged "Sim(3)" row
    start = time.perf_counter()
    registered = set().union(*(m.images for m in sessions.values()))
    matches = cross_session_matches(work_db, session_of, registered)
    ref, transforms = align_sessions(sessions, matches, rng)
    timing["sim3_estimation"] = time.perf_counter() - start
    summary["reference_session"] = ref
    summary["sim3"] = {n: info for n, (_, _, _, info) in transforms.items()}

    start = time.perf_counter()                       # applying the Sim(3)s and writing the model is part of the method
    aligned = [sessions[n].transformed(s, R, t) for n, (s, R, t, _) in transforms.items()]
    Model.merged(aligned).write(out / "sim3")
    timing["sim3_apply_and_write"] = time.perf_counter() - start
    summary["rows"]["sim3"] = evaluate(out / "sim3", csvs)

    # 6. "Sim(3) + BA": re-triangulate every match on the merged model, then bundle adjustment
    if not args.no_ba:
        start = time.perf_counter()
        (out / "sim3_ba").mkdir()                     # COLMAP needs an existing output folder
        colmap("point_triangulator", "--database_path", work_db, "--image_path", rgb_path, "--input_path", out / "sim3",
               "--output_path", out / "sim3_ba", "--clear_points", 1, "--refine_intrinsics", 0)
        colmap("bundle_adjuster", "--input_path", out / "sim3_ba", "--output_path", out / "sim3_ba",
               "--BundleAdjustment.refine_focal_length", refine, "--BundleAdjustment.refine_principal_point", 0,
               "--BundleAdjustment.refine_extra_params", refine, "--BundleAdjustment.refine_sensor_from_rig", 0,
               # COLMAP's default of 100 iterations stopped before convergence on ssk-s05; give the baseline its best fit
               "--BundleAdjustmentCeres.max_num_iterations", 500)
        colmap("model_converter", "--input_path", out / "sim3_ba", "--output_path", out / "sim3_ba", "--output_type", "TXT")
        timing["triangulation_and_ba"] = time.perf_counter() - start
        summary["rows"]["sim3_ba"] = evaluate(out / "sim3_ba", csvs)

    # the rows above only need the run's database, report and rgb_exp.csv, not its model (a failed Ours map has none)
    if (run / "colmap_00000" / "images.txt").exists():
        summary["rows"]["ours"] = evaluate(run / "colmap_00000", csvs)
    summary["runtime_s"] = {k: round(v, 1) for k, v in timing.items()}
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
