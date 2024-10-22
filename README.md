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

## Acknowledgements
We acknowledge the support of the PNRR ICSC National Research Centre for High Performance Computing, Big Data and Quantum Computing (CN00000013), under the NRRP MUR program funded by the NextGenerationEU.
 
