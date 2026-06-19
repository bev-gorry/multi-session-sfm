<p align="center">

  <h1 align="center"> Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change </h1>
  <!-- <h3 align="center">Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change</h3>  -->
  <p align="center">
    <a href="https://github.com/bev-gorry"><strong>Beverley Gorry</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=eq46ylAAAAAJ"><strong>Tobias Fischer</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en"><strong>Michael Milford</strong></a>
    .
    <a href="https://scholar.google.com/citations?user=SDtnGogAAAAJ&hl=en"><strong>Alejandro Fontan</strong></a>
  </p>

##

  <p align="center">
    <a href="assets/Gorry-B_LongTermMultiSession3D_2026.pdf" align="center">Paper</a> 
    | <a href="https://arxiv.org/abs/2602.20584" align="center">arXiv</a>
    <!-- | <a href="https://underloc.github.io/" align="center"> Project Page</a> -->
  </p>

This repository contains code for the paper "Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change."

## Getting Started

We use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, run the following command in your terminal:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

_After installation, restart your terminal or source your shell for the changes to take effect_. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/bev-gorry/multi-session-sfm.git && cd multi-session-sfm
```

## Setup

Initialize the included submodules:

```bash
git submodule update --init --recursive
```

**IMPORTANT:** Install LightGlue in editable mode:

```bash
pixi run -e lightglue install
```

<!-- Optionally, clone colmap and glomap (VSLAM-LAB will do this automatically).
```bash
pixi run -e colmap git-clone
pixi run -e glomap git-clone
``` -->

NOTE: Please update the following variables in _VSLAM_LAB/path_constants.py_

```
HUGGINGFACE_TOKEN
```

<!-- Clone and setup LightGlue:

```bash
pixi run git-clone-lightglue
pixi run install-lightglue
``` -->

<!-- NTS: Somewhere here I will also need to install colmap and glomap and make the VPR changes which are not on huggingface. -->

## IMPORTANT: Experiment Setup
Ensure that the necessary datasets and sequences are downloaded in VSLAM-LAB format **before** computing the distance matrix.
```bash
cd baselines/VSLAM_LAB
pixi run get-experiment-resources ../../arguments/exp_test_vslamlab.yaml
cd ../..
```

## VPR-LAB and Distance Matrix Filtering

Compute a distance matrix **D.npy** from a combined 'all' subset. List the VPR method and argument yaml file in succession.

```bash
pixi run -e vprlab vpr-lab megaloc ../../arguments/exp_test.yaml
```

Clean the distance matrix by setting diagonal values and same-sequence values to inf. Then apply the distance threshold which must be specified in your experiment yaml file.

```bash
pixi run -e lightglue python scripts/clean_distance_matrix.py --exp_yaml=arguments/exp_test.yaml
pixi run -e lightglue python scripts/apply_distance_threshold.py --exp_yaml=arguments/exp_test.yaml
```

<!-- WIP: This may change to include the threshold as a command rather than specifying it in the experiment yaml. We may also standardize the output: **D_brinary_0.6.npy** -> **D_binary.npy**.

```bash
pixi run python scripts/apply_distance_threshold.py --threshold=0.6
``` -->

## VSLAM-LAB

Create an experiment yaml file (specific to your VSLAM-Lab exp) and ensure that it points to the correct config file in VSLAM-Lab. Ensure that these variables in _VSLAM_LAB/path_constants.py_ are correct according to your own benchmark containing the distance matrix: VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION. Run VSLAM-Lab:

```bash
pixi run -e vslamlab vslamlab ../../arguments/exp_test_vslamlab.yaml --overwrite
```

## Nerfstudio Training

Nerfstudio is included as a git submodule at `baselines/nerfstudio`, using Tobias Fischer's fork. Train a Splatfacto model from the benchmark images and COLMAP output described by your experiment yaml:

```bash
pixi run train-splatfacto
```

By default this resolves `arguments/exp_test.yaml` to the benchmark image directory from `log_dir`, the VSLAM-LAB COLMAP model at `baselines/VSLAM-LAB-Evaluation/<exp_name>/<dataset>/<subset>/colmap_00000/0`, and runs Nerfstudio's COLMAP dataparser:

```bash
ns-train splatfacto --vis viewer colmap --data <benchmark-subset> --images-path rgb_0 --colmap-path <resolved-colmap-model>
```

To train on another COLMAP-backed dataset directory, pass the dataset root and COLMAP model explicitly:

```bash
pixi run train-splatfacto "--data=/path/to/monkey_output --images-path=images --colmap-path=/path/to/monkey_output/colmap/sparse/0 --vis=viewer"
```

## Coloring Pointclouds
Color each point in a pointcloud according to the year in which it is observed. This changes depending on the dataset.

```bash
pixi run python vis/color_pointcloud_sesoko.py --exp_yaml=arguments/exp_test.yaml
```

## Rerun Visualization
View the joint COLMAP reconstruction in [Rerun](https://rerun.io/), grouped by session/year. This viewer is visualization-only: it reads the precomputed COLMAP model, benchmark images, point clouds, and optional Gaussian splat PLYs. It does not run reconstruction, relocalisation, VPR, matching, or distance-matrix filtering.

Generate one COLMAP-initialized Gaussian Splat PLY per session:

```bash
pixi run -e lightglue export-session-splats "--exp_yaml=arguments/exp_test.yaml"
```

The exporter writes splats to `outputs/gaussian_splats/<exp_name>/<dataset>/<subset>/`. The lightweight viewer opens with an explicit Rerun layout:

- `3D reconstruction` shows the aligned session point clouds, camera frustums, the current camera, optional rays, and optional Gaussian splat centers.
- `Current image` is a single timeline-driven image panel. Scrub the `frame` timeline to step through images sorted by timestamp; the status panel shows the image timestamp, session, and per-session frame.
- `world/current_camera/image/observed_points` contains COLMAP 2D observations for the active image.
- `world/current_camera/image/reprojected_points/<session>` contains all visible 3D points from that source session reprojected into the active image. Toggle `ssk16`, `ssk17`, and `ssk18` to compare sessions.
- Selecting a 2D reprojection exposes point id, source session, target image, pixel, depth, 3D coordinates, a representative source image, every source-session observation image, and every image in the COLMAP track. A reprojected point is derived from a 3D point rather than one unique source image, so the representative source image is the source-session observation closest in timestamp to the target image.
- Selecting a 3D point cloud entity exposes point ids and compact observation summaries. Full point-track documents are opt-in.
- Hovering/selecting a 3D point links to matching 2D observations/reprojections using the COLMAP `POINT3D_ID`, encoded into Rerun `class_id`/`keypoint_id` correspondence fields.
- Static camera frustums are light by default. Add `--log-camera-images` when you want each logged frustum to contain its source image and COLMAP 2D observations, like the Rerun SfM demo.

Run the default lightweight viewer:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml"
```

By default this logs 25 sorted images per session, cross-session reprojection overlays for those images, all session point clouds, and no full point-track documents or splats. To log every image:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --all-images"
```

For the included Sesoko example, `--all-images` logs the complete three-session sequence (`ssk16`, `ssk17`, and `ssk18`: 740 images total). Use `--sessions` only when you intentionally want a subset.

To include Gaussian splat centers as toggleable point-style Rerun geometry:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --show-splats"
```

Splats are logged under `world/gaussian_splats/<session>`. They are decoded from the PLY centers and displayed as a radius/opacity point proxy. Rerun 0.33 does not expose a native anisotropic Gaussian-splat rasterizer, so this is useful for aligned 3D inspection but is still not a continuous 3DGS render. If they are hidden by the COLMAP points, temporarily disable `world/sessions/*/points3D`, increase their apparent footprint with `--splat-radius-scale`, `--splat-min-radius`, or `--splat-opacity-scale`, or remove the default cap with `--max-splats-per-session=-1`.

To make every logged camera frustum open to its image and participate in COLMAP point/image correspondence picking:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --all-images --log-camera-images"
```

This logs the complete Sesoko sequence images under `world/sessions/<session>/cameras/<image>/image`. It is heavier than the default timeline view, but it is the mode to use when you want the whole three-session sequence available from frustums.

The static frustum images contain the COLMAP observations by default. To also put all cross-session reprojection overlays under every logged frustum image, add `--log-camera-reprojections`; use `--max-reprojected-points-per-session` if you need to cap that heavier mode.

To draw current-frame camera-to-point rays:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --show-rays --max-rays-per-session=50"
```

To show cross-session support images for the active image, ranked by shared COLMAP point tracks:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --support-images-per-session=1"
```

The support image row logs each support camera/image under `world/support_images/<session>/slot_<n>/image` with the same `POINT3D_ID` correspondence fields as the 3D point cloud and current image overlays. Hovering/selecting a point can then link across the current image and visible support images from other sessions. Add `--support-include-active-session` if you also want same-session support images.

Rerun's automatic hover correspondence only works across entities that are already logged and visible. If a frustum has no `image` child, there is no 2D view for Rerun to highlight from that camera. Use `--log-camera-images` for frustum-to-image picking across the selected sequence, and combine it with `--all-images` when you want all images from all three sessions.

For the most reliable point-to-images workflow, select a pixel/reprojection, read its `point_id` from the selection panel, then rerun the viewer in inspected-point mode:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --inspect-point-id=1146"
```

This logs `world/inspected_points/pid_<POINT3D_ID>` with a highlighted 3D point, persistent camera-to-point rays, and every COLMAP observation image for that point across all sessions in the precomputed model. Use `--max-inspect-observations-per-point` to cap very long tracks, and `--inspect-point-image-views` to control how many observation images are placed directly in the blueprint.

Point-ID inspection is focused and lightweight by default: it does not load the full point clouds, timeline, or dense reprojection overlays, and places at most four observation images in the initial layout. Add `--inspect-with-context` only when the complete reconstruction context is needed.

If only the corresponding filenames are needed, print them without opening Rerun:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --inspect-point-id=1146 --inspect-lookup-only"
```

If extracting the `point_id` from Rerun is awkward, look it up directly from an image and clicked pixel:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --inspect-image=8315199.jpg --inspect-pixel 137.3 485.8 --inspect-source-session=ssk17"
```

The command finds the nearest reprojected point from `ssk17`, prints its `point_id` and every COLMAP observation image/pixel across all sessions, then automatically opens the inspected-point views and rays. Omit `--inspect-source-session` to compare the nearest observed/reprojected candidate from every session. The default lookup tolerance is 15 pixels; adjust it with `--inspect-pixel-max-distance`.

For a quick terminal lookup without opening Rerun, add `--inspect-lookup-only`:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --inspect-image=8315199.jpg --inspect-pixel 137.3 485.8 --inspect-source-session=ssk17 --inspect-lookup-only"
```

To write an `.rrd` file instead of spawning the viewer:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --output=outputs/session_view.rrd"
```

To provide splats from another pipeline such as Nerfstudio, pass them explicitly with `--splat-asset`, or point the viewer at a directory with `--splat-dir`:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --show-splats --splat-asset=/path/to/splat.ply"
```

To also log full point-track documents, optionally with duplicated support images, use an explicit cap:

```bash
pixi run -e rerun-viewer rerun-viewer "--exp_yaml=arguments/exp_test.yaml --max-track-docs=50 --log-track-images"
```

## Evaluation

Populate your own csv files with image pairs for evaluation, following the format provided.
* **select_eval_points.py:** The script will show the image pairs and clickable keypoints on the query image. User-clicked keypoints are projected onto the database image, where users can click on a projected point to move and correct its position. Must be run under the *lightglue* environment.
* **project_eval_points.py:** Iterates over the image pairs in the csv file and projects query keypoints onto the database image. Projections are compared against uv_groundtruth (in the csv files) and reprojection error is measured. Must be run under the *lightglue* environment. A *.npy* file is produced for each csv file, corresponding to a session combination.
* **compute_rpe_from_projected_points.py:** Concatenates reprojection errors from each session combination into one *.npy* file. Also computes the mean and median rpe.
* **plot_rpe.py:** Created a plot of median reprojection error for each method across the experiment subset. Inset shows a cropped plot of the top *x%*.

## Included Repositories

The following forked repositories are included in our repository:

- [COLMAP](https://colmap.github.io/)
- [LightGlue](https://github.com/cvg/LightGlue)
- [Nerfstudio](https://github.com/Tobias-Fischer/nerfstudio)
- [VPR-LAB](https://github.com/VSLAM-LAB/VPR-LAB)
- [VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB)

## Citation

Thanks for using our work. You can cite it as:

```bibtex
@misc{gorry2026multisession3dreconstructionappearancechange,
      title={Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change},
      author={Beverley Gorry and Tobias Fischer and Michael Milford and Alejandro Fontan},
      year={2026},
      eprint={2602.20584},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.20584},
}
```

We also encourage citing [COLMAP](https://colmap.github.io/), [LightGlue](https://github.com/cvg/LightGlue), [VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB), and [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation).

## Acknowledgements

This research was partially supported by funding from ARC Laureate Fellowship FL210100156 to MM and ARC DECRA Fellowship DE240100149 to TF. The authors acknowledge continued support from the Queensland University of Technology (QUT) through the Centre for Robotics.

We would particularly like to acknlowedge the authors of [COLMAP](https://colmap.github.io/), [LightGlue](https://github.com/cvg/LightGlue), [VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB), and [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation).
