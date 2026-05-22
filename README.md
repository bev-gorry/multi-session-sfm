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

Clone VSLAM-LAB and VPR-LAB:

```bash
pixi run -e vslamlab git-clone
pixi run -e vprlab git-clone-vpr-methods
```

**IMPORTANT:** Clone and install Lightglue in VSLAM-LAB

```bash
cd baselines/VSLAM_LAB
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

## VPR-LAB and Distance Matrix Filtering

Compute a distance matrix **D.npy** from a combined 'all' subset:

```bash
pixi run -e vprlab vpr-lab
```

Clean the distance matrix by setting diagonal values and same-sequence values to inf. Then apply the distance threshold which must be specified in your experiment yaml file.

```bash
pixi run python scripts/clean_distance_matrix.py --exp_yaml=arguments/exp_test.yaml
pixi run python scripts/apply_distance_threshold.py --exp_yaml=arguments/exp_test.yaml
```

<!-- WIP: This may change to include the threshold as a command rather than specifying it in the experiment yaml. We may also standardize the output: **D_brinary_0.6.npy** -> **D_binary.npy**.

```bash
pixi run python scripts/apply_distance_threshold.py --threshold=0.6
``` -->

## VSLAM-LAB

Create an experiment yaml file (specific to your VSLAM-Lab exp) and ensure that it points to the correct config file in VSLAM-Lab. Ensure that these variables in _VSLAM_LAB/path_constants.py_ are correct according to your own benchmark containing the distance matrix: VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION. Run VSLAM-Lab:

```bash
pixi run vslamlab
```

## COMING NEXT

Evaluation scripts and visualizations.

## Included Repositories

The following forked repositories are included in our repository:

- [COLMAP](https://colmap.github.io/)
- [LightGlue](https://github.com/cvg/LightGlue)
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
