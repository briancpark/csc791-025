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

## Project 1: DNN Pruning via NNI
Located in `proj1`
