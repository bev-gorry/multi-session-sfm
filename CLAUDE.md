# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for "Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change" — a pipeline that reconstructs joint 3D scenes from images captured across multiple sessions (years apart) with substantial appearance change, and evaluates cross-session reprojection accuracy. Baseline methods (VSLAM-LAB/COLMAP, ICP, BUFFER-X) are compared. This is not a library with a stable API — scripts are run individually as pipeline stages, mostly against experiment-specific YAML configs and hardcoded local dataset paths.

## Environment management (pixi)

This project uses [pixi](https://pixi.sh/latest/), not pip/conda directly. There are several environments (`pixi.toml`), each scoped to one pipeline stage because dependencies conflict (e.g. VPR-LAB needs `numpy<2`, others don't):

- `default` — general pipeline env (`pipeline` feature): opencv, pytorch-gpu, open3d, etc.
- `vslamlab` — running VSLAM-LAB (baselines/VSLAM_LAB submodule)
- `vprlab` — running VPR-methods-evaluation for visual place recognition / distance matrix
- `lightglue` — feature matching, rerun visualization, splat export (needs LightGlue installed in editable mode)
- `rerun-viewer` — lightweight env for just the Rerun 3D viewer, no torch

Run scripts as `pixi run -e <env> <task-or-python-command>`. Defined tasks (see `pixi.toml`) include `train-splatfacto`, `vslamlab`, `vpr-lab`, `rerun-viewer`, `export-session-splats`, `install` (LightGlue editable install), `submodule-update`.

Setup from scratch: `git submodule update --init --recursive`, then `pixi run -e lightglue install` (installs LightGlue in editable mode — required before other lightglue-env scripts work). `HUGGINGFACE_TOKEN` and other path constants must be set in `baselines/VSLAM_LAB/path_constants.py` before running experiments.

There is no test suite, linter, or CI config in this repo — don't invent commands for these.

## Submodules (`baselines/`)

Four git submodules, each a forked external project:
- `VSLAM_LAB` (fork of VSLAM-LAB, branch `main`) — runs the actual SLAM/SfM reconstruction (COLMAP/glomap) driven by experiment yaml files
- `LightGlue` — feature extraction/matching, installed editable into the `lightglue` env
- `VPR-methods-evaluation` (fork) — visual place recognition, produces the cross-session distance matrix
- `nerfstudio` (Tobias Fischer's fork) — Gaussian splat / NeRF training from COLMAP output

Treat these as vendored external code unless a task specifically involves modifying the fork itself.

## Pipeline architecture

The full workflow (see `RECREATE_EXPERIMENTS.md` for the authoritative step-by-step per baseline: OURS, ICP, BUFFER) is:

1. **Dataset prep** (`dataset/`) — build/inspect session-based datasets (currently Eiffel and Sesoko datasets), select common anchors/subsets across sessions.
2. **VPR distance matrix** — `pixi run -e vprlab vpr-lab <method> <exp_yaml>` computes pairwise visual distances between all images across sessions into `D.npy`.
3. **Distance matrix filtering** (`scripts/clean_distance_matrix.py`, `scripts/apply_distance_threshold.py`) — zero out diagonal/same-sequence entries, then threshold using `dist_threshold` from the experiment yaml. This filtered matrix gates which cross-session image pairs get matched, so it must run before VSLAM-LAB.
4. **VSLAM-LAB reconstruction** — `pixi run -e vslamlab vslamlab <exp_yaml_vslamlab> --overwrite` runs COLMAP/glomap (in submodule `baselines/VSLAM_LAB`) using the filtered distance matrix and a custom feature matcher for cross-session matching. Produces a joint COLMAP model per experiment.
5. **Baselines for comparison**: `open3d_icp_part1.py`/`part2.py` (ICP alignment, needs manually-set paths), or BUFFER-X (external repo, ply-based registration) — see RECREATE_EXPERIMENTS.md.
6. **Visualization/coloring** (`vis/color_pointcloud_*.py`, `vis/rerun_viewer.py`, `vis/export_session_splats.py`) — color/inspect the joint reconstruction. `rerun_viewer.py` is read-only/visualization: it never runs reconstruction itself, only renders the precomputed COLMAP model, images, point clouds, and optional Gaussian splat PLYs, grouped by session. Point correspondence between 2D observations and 3D points is threaded through using COLMAP `POINT3D_ID` as Rerun `class_id`/`keypoint_id`.
7. **Evaluation** (`eval/`) — human-in-the-loop reprojection accuracy pipeline, run in this order:
   - `find_annotation_image_pairs.py` selects spatially-distributed, low-visual-distance cross-session image pairs from the distance matrix → `evaluation_points_*.csv` per session-pair (e.g. 2016-2018)
   - `select_eval_points.py` (lightglue env, GUI) — human clicks keypoints on query image, corrects projected position on database image
   - `project_eval_points.py` (lightglue env) — projects query keypoints into database image via the COLMAP model, compares to clicked ground truth, produces per-session-pair `.npy` reprojection errors
   - `compute_rpe_from_projected_points.py` — concatenates all session-pair errors into one array, computes mean/median
   - `plot_rpe.py` — plots median reprojection error across methods (run only after all baselines are evaluated)

## Experiment configuration

Everything is driven by per-experiment YAML files in `arguments/`. There are two related yaml files per experiment:
- `exp_<name>.yaml` — pipeline config (`exp_name`, `dataset`, `subset`, `log_dir`, `rgb_list_db`/`rgb_list_q`, `dist_threshold`). Loaded via `experiment.load_exp_yaml()` (`experiment.py`), which resolves `log_dir`/`rgb_list_db`/`rgb_list_q` to absolute paths relative to the repo root or the yaml file's own location.
- `exp_<name>_vslamlab.yaml` — VSLAM-LAB-specific config (its own `Config`, `Module`, `Parameters` schema, consumed directly by the VSLAM_LAB submodule, not by `load_exp_yaml`).

Almost every script/task takes `--exp_yaml=arguments/exp_<name>.yaml` as its entry point — when adding a new script that needs experiment context, follow this convention rather than hardcoding paths.

## Known hardcoded paths

These are not read from experiment yaml and must be edited directly in source when the pipeline moves machines (tracked in `RECREATE_EXPERIMENTS.md`):
- `scripts/train_nerfstudio.py` — `VSLAM_EVAL_ROOT`
- `dataset/create_eiffel_dataset.py` (and other eiffel scripts) — `dataset_root`
- `baselines/VSLAM_LAB/path_constants.py` — `VSLAMLAB_BENCHMARK`, `VSLAMLAB_EVALUATION`, `HUGGINGFACE_TOKEN`
- `constants.py` — `OVERLEAF_PATH` (paper figure export target)

## Data/session model

Datasets (Sesoko, Eiffel) consist of multiple **sessions** (captures separated by months/years, named e.g. `ssk16`/`ssk17`/`ssk18` for years 2016/2017/2018) that are jointly reconstructed into one COLMAP model. Most downstream code (rerun viewer, pointcloud coloring, RPE evaluation) is organized around per-session grouping and cross-session comparison — expect session name/year to be a key dimension in most data structures, filenames, and plot legends.

## Coding preferences
- Python, concise, well-commented
- Always explain what a function does before writing it
- Prefer explicit over clever
