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
View the joint COLMAP reconstruction in [Rerun](https://rerun.io/), grouped by session/year. The viewer logs one toggleable entity tree per session, including camera images, per-image 2D point observations, and the 3D points observed by that session. It also logs point-track documents grouped by session, so a selected `POINT3D_ID` can be looked up under `world/point_tracks/pid_<POINT3D_ID>`.

Generate one COLMAP-initialized Gaussian Splat PLY per session:

```bash
pixi run -e lightglue export-session-splats "--exp_yaml=arguments/exp_test.yaml"
```

The exporter writes splats to `outputs/gaussian_splats/<exp_name>/<dataset>/<subset>/`. The Rerun viewer automatically logs any `.ply` files from that directory under `world/gaussian_splats`, alongside the COLMAP point cloud and source images:

```bash
pixi run -e lightglue rerun-viewer "--exp_yaml=arguments/exp_test.yaml"
```

For large reconstructions, limit image logging while keeping all 3D points:

```bash
pixi run -e lightglue rerun-viewer "--exp_yaml=arguments/exp_test.yaml --max-images-per-session=25"
```

To write an `.rrd` file instead of spawning the viewer:

```bash
pixi run -e lightglue rerun-viewer "--exp_yaml=arguments/exp_test.yaml --output=outputs/session_view.rrd"
```

To provide splats from another pipeline such as Nerfstudio, pass them explicitly with `--splat-asset`, or point the viewer at a directory with `--splat-dir`:

```bash
pixi run -e lightglue rerun-viewer "--exp_yaml=arguments/exp_test.yaml --splat-asset=/path/to/splat.ply"
```

To also duplicate the actual projected images under each point-track entity, use `--log-track-images` with a cap:

```bash
pixi run -e lightglue rerun-viewer "--exp_yaml=arguments/exp_test.yaml --max-track-docs=50 --log-track-images"
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
