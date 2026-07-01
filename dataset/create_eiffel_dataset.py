import os
import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm



EIFFEL_SEQUENCES = {
    "eff15": "https://www.seanoe.org/data/00810/92226/data/98240.zip",
    "eff16": "https://www.seanoe.org/data/00810/92226/data/98289.zip",
    "eff18": "https://www.seanoe.org/data/00810/92226/data/98314.zip",
    "eff20": "https://www.seanoe.org/data/00810/92226/data/98356.zip",
}

R = 6378137.0  # WGS84


# =======================================================
# UTILITIES
# =======================================================

def run(cmd):
    subprocess.run(cmd, check=True)


def download_zip(dataset_path: Path, url: str):
    dataset_path.mkdir(parents=True, exist_ok=True)
    zip_name = url.split("/")[-1]
    zip_path = dataset_path / zip_name

    if zip_path.exists():
        print(f"[✓] Already downloaded {zip_name}")
        return zip_path

    print(f"[↓] Downloading {zip_name}")
    run(["wget", "-c", url, "-P", str(dataset_path)])

    return zip_path


def unzip_and_normalize(zip_path: Path, dataset_path: Path):
    print(f"[↧] Unzipping {zip_path.name}")
    shutil.unpack_archive(str(zip_path), str(dataset_path))
    zip_path.unlink()

    # Find extracted folder (usually numeric or year-like)
    candidates = [p for p in dataset_path.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError("No extracted folder found after unzip.")

    # Heuristic: pick newest folder
    extracted = max(candidates, key=lambda p: p.stat().st_mtime)

    return extracted


def find_navigation_file(seq_path: Path):
    nav = seq_path / "navigation.txt"
    if nav.exists():
        return nav

    raise FileNotFoundError(f"Missing navigation.txt in {seq_path}")


def parse_navigation(nav_file: Path):
    df = pd.read_csv(nav_file, sep=r"\s+")

    # timestamps
    ts = pd.to_datetime(
        df["name"].str.replace(".png", "", regex=False),
        format="%Y%m%dT%H%M%S.%fZ",
        utc=True,
    ).astype("int64")

    df["timestamp_ns"] = ts
    return df


def ensure_rgb_folder(seq_path: Path, resolution=None):
    rgb_dir = seq_path / "rgb_0"
    img_dir = seq_path / "images"

    if rgb_dir.exists():
        return rgb_dir

    rgb_dir.mkdir(exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".png"])

    for img_path in tqdm(images, desc=f"rgb_0 {seq_path.name}"):
        img = Image.open(img_path)

        if resolution is not None:
            w, h = resolution
            scaled_h = int(np.sqrt(img.size[1] / img.size[0] * w * h))
            scaled_w = int(w * h / scaled_h)
            img = img.resize((scaled_w, scaled_h), Image.LANCZOS)

        img.save(rgb_dir / img_path.name)

    return rgb_dir


def validate_consistency(df, rgb_dir: Path):
    nav_set = set(df["name"])
    img_set = set([p.name for p in rgb_dir.iterdir() if p.suffix == ".png"])

    missing_imgs = nav_set - img_set
    extra_imgs = img_set - nav_set

    if missing_imgs:
        raise RuntimeError(f"Missing images in rgb_0: {list(missing_imgs)[:10]}")

    if extra_imgs:
        raise RuntimeError(f"Extra images not in navigation.txt: {list(extra_imgs)[:10]}")


# =======================================================
# GLOBAL FRAME (eff15 reference)
# =======================================================

def compute_enu(df, ref_df):
    ref = ref_df.iloc[0]

    lat0 = np.deg2rad(ref["lat"])
    lon0 = np.deg2rad(ref["lon"])
    alt0 = ref["alt"]

    lat = np.deg2rad(df["lat"].to_numpy())
    lon = np.deg2rad(df["lon"].to_numpy())
    alt = df["alt"].to_numpy()

    east = (lon - lon0) * np.cos(lat0) * R
    north = (lat - lat0) * R
    up = alt - alt0

    return east, north, up


def write_groundtruth(seq_path: Path, df, east, north, up):
    out = seq_path / "groundtruth.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

        for i, row in df.iterrows():
            w.writerow([
                int(row["timestamp_ns"]),
                east[i], north[i], up[i],
                0.0, 0.0, 0.0, 1.0
            ])

    print(f"[✓] groundtruth.csv -> {out}")


def write_rgb(seq_path: Path, df):
    out = seq_path / "rgb.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])

        for _, row in df.iterrows():
            w.writerow([
                int(row["timestamp_ns"]),
                f"rgb_0/{row['name']}"
            ])

    print(f"[✓] rgb.csv -> {out}")


# =======================================================
# CALIBRATION
# =======================================================

def write_calibration(seq_path: Path, fx=541.5, fy=541.5):
    rgb_dir = seq_path / "rgb_0"
    img = next(rgb_dir.glob("*.png"))
    im = Image.open(img)

    w, h = im.size
    cx, cy = w / 2, h / 2

    T_BS = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]

    yaml = f"""%YAML 1.2
---
cameras:
- {{cam_name: rgb_0,
   cam_type: rgb,
   cam_model: pinhole,
   distortion_type: radtan4,
   focal_length: [{fx}, {fy}],
   principal_point: [{cx}, {cy}],
   distortion_coefficients: [0, 0, 0, 0],
   image_dimension: [{w}, {h}],
   fps: 25.0,
   T_BS: [{", ".join(map(str, T_BS))}]
}}
"""

    (seq_path / "calibration.yaml").write_text(yaml)
    print(f"[✓] calibration.yaml")


# =======================================================
# PIPELINE PER SEQUENCE
# =======================================================

def process_sequence(seq_path: Path, ref_nav):
    print(f"\n=== Processing {seq_path.name} ===")

    nav_file = find_navigation_file(seq_path)
    df = parse_navigation(nav_file)

    rgb_dir = ensure_rgb_folder(seq_path)
    validate_consistency(df, rgb_dir)

    east, north, up = compute_enu(df, ref_nav)

    write_groundtruth(seq_path, df, east, north, up)
    write_rgb(seq_path, df)
    write_calibration(seq_path)


# =======================================================
# MAIN
# =======================================================

def main():
    dataset_root = Path("/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK/EIFFEL_TEST")
    dataset_root.mkdir(parents=True, exist_ok=True)

    # download + extract
    for seq, url in EIFFEL_SEQUENCES.items():
        zip_path = download_zip(dataset_root, url)
        extracted = unzip_and_normalize(zip_path, dataset_root)

        # rename to canonical folder name (eff15, eff16, etc.)
        target = dataset_root / seq
        if extracted != target:
            if target.exists():
                shutil.rmtree(target)
            extracted.rename(target)

    # reference frame (eff16)
    ref_nav = parse_navigation(find_navigation_file(dataset_root / "eff16"))

    # process all
    for seq in tqdm(EIFFEL_SEQUENCES.keys()):
        process_sequence(dataset_root / seq, ref_nav)

    print("\nDone.")


if __name__ == "__main__":
    main()