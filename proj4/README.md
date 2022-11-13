# Project 4 (Optional): Distillation and Compilation

## PSC Installation and Setup

### Install Anaconda

Create a conda environment
    
```sh
conda create -n csc791 python=3.10
```

After creating conda environment, you can debug via interative sessions for 1 hour intervals. You need to activate an interactive session attached to a GPU to link the PyTorch libraries correctly to CUDA.
```sh
salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive -t 01:00:00
```

### Install dependencies
```sh
pip3 install -r requirements
````
For PyTorch, special installation is required to link to CUDA correctly:
```sh
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
```

### Running Project