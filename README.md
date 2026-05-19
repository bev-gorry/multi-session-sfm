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

Clone and setup LightGlue:
```bash
pixi run git-clone-lightglue
pixi run install-lightglue
```

## VPR-Lab and Distance Matrix Filtering
Compute a distance matrix **D.npy** from a combined 'all' subset:
```bash
pixi run -e vprlab vpr-lab
```

Clean the distance matrix by setting diagonal values and same-sequence values to inf. Then apply the distance threshold which must be specified in your experiment yaml file.
```bash
pixi run scripts/clean_distance_matrix.py
pixi run python scripts/apply_distance_threshold.py
```
WIP: This may change to include the threshold as a command rather than specifying it in the experiment yaml. We may also standardize the output: **D_brinary_0.6.npy** -> **D_binary.npy**.
```bash
pixi run python scripts/apply_distance_threshold.py --threshold=0.6
```