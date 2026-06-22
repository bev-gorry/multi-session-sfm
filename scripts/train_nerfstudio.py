#!/usr/bin/env python3
"""Run Nerfstudio training from this repository's experiment layout."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from experiment import load_exp_yaml, resolve_repo_path


REPO_ROOT = Path(__file__).resolve().parents[1]
NERFSTUDIO_MANIFEST = REPO_ROOT / "baselines" / "nerfstudio" / "pixi.toml"
VSLAM_EVAL_ROOT = REPO_ROOT / "baselines" / "VSLAM-LAB-Evaluation"


def load_exp(exp_yaml: Path) -> dict:
    exp = load_exp_yaml(exp_yaml)
    required_keys = ("exp_name", "dataset", "subset")
    missing = [key for key in required_keys if key not in exp]
    if missing:
        raise ValueError(f"{exp_yaml} is missing required keys: {', '.join(missing)}")
    return exp


def default_colmap_dir(exp: dict) -> Path:
    return VSLAM_EVAL_ROOT / exp["exp_name"] / exp["dataset"] / exp["subset"] / "colmap_00000"


def default_data_dir(exp: dict) -> Path:
    if "log_dir" in exp:
        return Path(exp["log_dir"])
    return REPO_ROOT / "baselines" / "VSLAM-LAB-Benchmark" / exp["dataset"] / exp["subset"]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a Nerfstudio model on a multi-session SFM output.")
    parser.add_argument("--exp_yaml", type=Path, default=REPO_ROOT / "arguments" / "exp_test.yaml")
    parser.add_argument("--data", type=Path, default=None, help="Nerfstudio data directory. Defaults to the exp YAML benchmark subset.")
    parser.add_argument("--images-path", type=Path, default=Path("rgb_0"), help="Image directory relative to --data.")
    parser.add_argument("--colmap-path", type=Path, default=None, help="COLMAP model directory. Defaults to the exp YAML VSLAM output.")
    parser.add_argument("--dataparser", default="colmap")
    parser.add_argument("--method", default="splatfacto")
    parser.add_argument("--vis", default="viewer")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    exp_yaml = resolve_repo_path(args.exp_yaml)
    exp = load_exp(exp_yaml)
    data_dir = args.data if args.data is not None else default_data_dir(exp)
    colmap_dir = args.colmap_path if args.colmap_path is not None else default_colmap_dir(exp) / "0"

    nerfstudio_args = [
        "--vis",
        args.vis,
        *extra_args,
        args.dataparser,
        "--data",
        str(data_dir),
        "--images-path",
        str(args.images_path),
        "--colmap-path",
        str(colmap_dir),
    ]
    if args.method == "splatfacto":
        command = [
            "pixi",
            "run",
            "--manifest-path",
            str(NERFSTUDIO_MANIFEST),
            "train-splatfacto-colmap",
            shlex.join(nerfstudio_args),
        ]
    else:
        command = [
            "pixi",
            "run",
            "--manifest-path",
            str(NERFSTUDIO_MANIFEST),
            "ns-train",
            args.method,
            *nerfstudio_args,
        ]

    if args.dry_run:
        print(" ".join(command))
        return

    subprocess.run(command, cwd=REPO_ROOT / "baselines" / "nerfstudio", check=True)


if __name__ == "__main__":
    main()
