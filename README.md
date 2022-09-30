# csc791-025

Oliver Fowler, Brian Park

## Setup
For setup on local machine or ARC cluster:
```sh
git clone --recurse git@github.com:briancpark/csc791-025.git
git submodule update --init --recursive # For any new submodules added to the repo
```

Packages will be installed along the way as the semester progresses. Make a conda environment:
```sh
conda create -n csc791 python=3.10
pip3 install -r requirements.txt
```

For M1, you'll want to install packages that are optimized. For NumPy, we can link Apple Accelerate LAPACK library: 
```
conda install "libblas=*=*accelerate"
```
The PyTorch project moves rapidly and still at it's early stages for support on M1. A lot of bugs arise, as well as a lot of fixes for them. For example, we already had to update from 1.12 to 1.13. We can update by running this:
```
conda update pytorch torchvision -c pytorch-nightly
```

## Project 1: DNN Pruning via NNI
Located in `proj1`




