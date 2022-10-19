# Project 3: DNN HPO via NNI

## PSC Installation and Setup

Install Anaconda

Create a conda environment
    
```sh
conda create -n csc791 python=3.10
```

Install dependencies



After installing, you can debug via interative sessions for 1 hour intervals
```sh
salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive -t 01:00:00
```

Once done debugging, run batch scripts.. These experiments will take long, so interative sessions won't cut it
```sh
sbatch job-proj3
```