# NadirFloorNet: Reconstructing Multi-Room Floorplans from Registered Panoramic Images

**NadirFloorNet** is a deep-learning framework for reconstructing complex indoor multi-room floorplans from a small set of spatially registered panoramic images.

This repository provides demo code, example data, and pretrained models related to the official paper:

**NadirFloorNet: Reconstructing Multi-Room Floorplans from a Small Set of Registered Panoramic Images**  
Giovanni Pintore, Uzair Shah, Marco Agus, Enrico Gobbetti  
Proc. CVPR, 2025

The method predicts indoor floorplans by combining single-panorama room-shape estimation with a multi-view floorplan reconstruction stage. Given a set of registered and vertically aligned equirectangular images, the pipeline first predicts clutter-free depth and local floor-projection representations, called **Nadir shapes**, for each panorama. These local predictions are then transformed into a common floorplan reference frame using the known camera registration. A transformer-based encoder-decoder architecture processes the resulting set of Nadir maps and predicts the final room polygons of the reconstructed multi-room floorplan.

Keywords: **NadirFloorNet**, **NadirFloorNet GitHub**, **multi-room floorplan reconstruction**, **indoor reconstruction**, **panoramic images**, **360 images**, **equirectangular images**, **registered panoramas**, **deep learning**, **computer vision**.

---

## Overview

The approach assumes as input a set of spatially registered and vertically aligned equirectangular panoramic images.

For each input image, a depth and room-shape prediction module estimates:

- an intermediate clutter-free depth map of the scene;
- a segmented floor projection, referred to as a **Nadir shape**, representing the predicted uncluttered room footprint.

We exploit camera registration to place all Nadir projections in the same reference floorplan coordinate system.

Given this joined representation, we adopt a transformer-based encoder-decoder architecture to process relationships between Nadir projections and predict the final room shapes, using a two-level query embedding representation, namely room polygons and room corners.

As a final result, the method outputs the predicted room polygons forming the reconstructed multi-room floorplan.

## Python Requirements
See the file [requirements.txt](requirements.txt)

## Pre-requisited
 * Compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)) and the differentiable rasterization module (from [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)):
 ```shell
 cd models/ops
 sh make.sh

 # unit test for deformable-attention modules (should see all checking is True)
 # python test.py

 cd ../../diff_ras
 python setup.py build develop
 ```
 
## Installation
We suggest to create a Python virtual environment and installing all the essential Python modules using pip. After cloning the repository, run:

```
# python -m venv .env
# source .env/bin/activate
# pip install -r requirements.txt
```

## Data

To test single image depth estimation and its floorplan footprint estimation we provide a panoramic indoor scene from [Structured3D](https://structured3d-dataset.org/) at [data/s3d_single/test](data/s3d_single/test) .
To test floorplan reconstruction we provide an exemple scene from [Structured3D](https://structured3d-dataset.org/) at [data/s3d_floor](data/s3d_floor), which includes as input 9 panoramic images from which an entire multi-room floor plan is reconstructed.

## Download Pretrained Models

Pretrained weights for a **pre-print/project version** of NadirFloorNet are available on Hugging Face:

[Download the pretrained weights from Hugging Face](https://huggingface.co/giannpint/NadirFloorNet-weights)

Please download the checkpoint files manually from the **Files and versions** tab of the Hugging Face repository and copy them into your local checkpoint directories.

Depending on the script being used, the checkpoints should be placed for example in:

```text
./checkpoints/
```

or:

```text
./nadirshape/ckpt/DEMO_RUNS/
./checkpoints/DEMO_RUNS/
```

> **Important note:**  
> The pretrained weights hosted on Hugging Face refer to a **pre-print/project version** developed within the PNRR ICSC project. They are **not** the weights used to produce the official quantitative and qualitative results reported in the published CVPR 2025 paper.

These weights are provided to support reproducibility, experimentation, and further research related to the pre-print/project version of the method.

## Usage
To test prediction of single image depth and its Nadir shape run (example): 
```
python eval_nadirshape.py --pth ./nadirshape/ckpt/DEMO_RUNS/s3d_depth/best_valid.pth --root_dir ./data/s3d_single/test/  
```    
    - `--pth` path to the trained model.
    - `--root_dir` path to the input equirectangular scene.
    - `--output_dir` path to the output results.

To generate Nadir maps from a set of omnidirectional images run (example): 
```
python generate_nadirmaps.py --pth ./nadirshape/ckpt/DEMO_RUNS/s3d_depth/best_valid.pth --data_dir ./data/s3d_floor  
```    
    - `--pth` path to the trained model.
    - `--data_dir` path to the input scene (registered images).
    - `--output_dir` path to the output results.

To predict a floorplan as a set of rooms polygons run (example): 
```
python eval_nadirfloor.py --pth checkpoints/DEMO_RUNS/nadirfloornet_s3d.pth --dataset_dir ./results/s3d_nadirmaps  
```    
    - `--pth` path to the trained model.
    - `--dataset_dir` path to the input scene (registered images).
    - `--output_dir` path to the output results.

## Acknowledgements
We acknowledge the support of the PNRR ICSC National Research Centre for High Performance Computing, Big Data and Quantum Computing (CN00000013), under the NRRP MUR program funded by the NextGenerationEU.

## Paper

If you use this code, data, or pretrained weights, please cite the official paper:

```bibtex
@inproceedings{Pintore:2025:NRM,
  author =        {Pintore, Giovanni and Shah, Uzair and Agus, Marco and
                   Gobbetti, Enrico},
  booktitle =     {Proc. CVPR},
  pages =         {1985--1994},
  title =         {{NadirFloorNet}: reconstructing multi-room floorplans
                   from a small set of registered panoramic images},
  year =          {2025},
  doi =           {10.1109/CVPRW67362.2025.00186},
}

 
