"""Check a joint model's global geometry against the reference poses: Sim(3)-align its camera centres to the run's
groundtruth.csv (Sesoko: vehicle navigation; Eiffel Tower: the dataset's reference poses) and report the residuals
- of all cameras with one Sim(3),
- of each session aligned on its own,
- of the cameras split into two groups that each get their own Sim(3) (2-means on the residuals, then reassigned to
  the better-fitting group).
A model that holds two copies of the scene fits badly as a whole but well per copy; a folded model fits badly even per
session. The annotated reprojection error cannot show either: each click pair falls inside one copy.
Note: Sesoko's navigation of different years is offset by a few metres (2018: ~3.5 m), so a correct joint Sesoko model
fits at ~1-2 m overall and ~0.2-0.4 m per session.

usage: python check_reference_poses.py <run folder> [...]   (<run folder> = <evaluation>/<exp>/<DATASET>/<sequence>)
"""
import csv
import sys
from pathlib import Path

import numpy as np


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sim3(a, b):
    """Umeyama: s, R, t with b ~ s R a + t."""
    ma, mb = a.mean(0), b.mean(0)
    U, D, Vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (a - ma).var(0).sum()
    return s, R, mb - s * R @ ma


def residuals(a, b):
    s, R, t = sim3(a, b)
    return np.linalg.norm(b - (s * (R @ a.T).T + t), axis=1)


def check(run: Path) -> None:
    rows = list(csv.DictReader(open(run / "rgb_exp.csv")))
    ts_of = {Path(r["path_rgb_0"]).name: int(float(r["ts_rgb_0 (ns)"])) for r in rows}
    sess_of = {Path(r["path_rgb_0"]).name: r["sequence_name"] for r in rows}
    gt = {int(float(r["ts (ns)"])): np.array([float(r["tx (m)"]), float(r["ty (m)"]), float(r["tz (m)"])])
          for r in csv.DictReader(open(run / "groundtruth.csv"))}

    names, est = [], []
    with open(run / "colmap_00000" / "images.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) >= 10 and t[-1].lower().endswith((".png", ".jpg", ".jpeg")):
                if ts_of.get(Path(t[-1]).name) in gt:
                    names.append(Path(t[-1]).name)
                    est.append(-quat_to_R(*map(float, t[1:5])).T @ np.array(list(map(float, t[5:8]))))
                next(f)                                # the POINTS2D line
    est = np.array(est)
    ref = np.array([gt[ts_of[n]] for n in names])
    sessions = np.array([sess_of[n] for n in names])

    res = residuals(est, ref)
    print(f"{run}\n  cameras: {len(est)}   all, one Sim(3): median {np.median(res):.2f} m, p90 {np.percentile(res, 90):.2f} m, "
          f"> 5 m: {100 * (res > 5).mean():.1f}%")
    for s in sorted(set(sessions)):
        r = residuals(est[sessions == s], ref[sessions == s])
        print(f"  {s}, own Sim(3): median {np.median(r):.2f} m, p90 {np.percentile(r, 90):.2f} m")

    # two groups with their own Sim(3)
    s, R, t = sim3(est, ref)
    vec = ref - (s * (R @ est.T).T + t)
    centres = vec[np.random.default_rng(0).choice(len(vec), 2, replace=False)]
    for _ in range(50):
        lab = np.argmin(np.linalg.norm(vec[:, None] - centres[None], axis=2), axis=1)
        centres = np.array([vec[lab == k].mean(0) if (lab == k).any() else centres[k] for k in range(2)])
    for _ in range(10):
        fits = []
        for k in range(2):
            if (lab == k).sum() < 3:
                fits.append(np.full(len(est), np.inf))
                continue
            sk, Rk, tk = sim3(est[lab == k], ref[lab == k])
            fits.append(np.linalg.norm(ref - (sk * (Rk @ est.T).T + tk), axis=1))
        lab = np.argmin(np.array(fits), axis=0)
    for k in range(2):
        m = lab == k
        if m.sum() < 3:
            continue
        r = residuals(est[m], ref[m])
        comp = {str(x): int((sessions[m] == x).sum()) for x in sorted(set(sessions))}
        print(f"  group {k}: {m.sum()} cameras {comp}, own Sim(3): median {np.median(r):.2f} m, p90 {np.percentile(r, 90):.2f} m")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(Path(arg))
