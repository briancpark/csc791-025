# Project 3: DNN HPO via NNI

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

### Executing NNI Experiments
First run `model.py` to download the datasets needed before running the HPO experiments
```sh
python3 model.py no-vgg
python3 model.py vgg
```

Once done debugging and installing, run batch scripts on bridges-2. These experiments will take long, so interactive sessions won't cut it:
```sh
sbatch job-proj3
```

If on local cluster or different environment, you can just run it:
```sh
bash job-proj3
```

### Viewing NNI Experiments
Once done running experiments, you can view them individually:
```sh
python3 proj3.py hpo view <experiment-id>
```