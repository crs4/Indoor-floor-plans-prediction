This repository provides work-in-progress code, data and pretrained models for a deep-learning approach for predicting complex indoor floor plans from registered omnidirectional images.

## Python Requirements
See the file `requirements.txt`

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

## Usage
To test prediction of single image depth and floorplan footprint run: 
```
python eval_nadirshape.py --pth ./nadirshape/ckpt/DEMO_RUNS/s3d_depth/best_valid.pth --root_dir ./data/s3d_single/test/  
```    
    - `--pth` path to the trained model.
    - `--root_dir` path to the input equirectangular scene.
    - `--output_dir` path to the output results.

## Acknowledgements
We acknowledge the support of the PNRR ICSC National Research Centre for High Performance Computing, Big Data and Quantum Computing (CN00000013), under the NRRP MUR program funded by the NextGenerationEU.
 
