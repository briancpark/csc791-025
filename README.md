# CSC 791-025

Oliver Fowler, Brian Park

CSC 791-025 is Real-Time AI & High-Performance Machine Learning at NCSU. We took this offering in Fall 2022 under Xipeng Shen. This repo contains our development setup and notes, as well as source code for our projects.

## Setup

For setup on local machine or ARC cluster:

```sh
git clone --recurse git@github.com:briancpark/csc791-025.git
git submodule update --init --recursive # For any new submodules added to the repo
```

Packages will be installed along the way as the semester progresses. Make a Conda environment:

```sh
conda create -n csc791 python=3.10
pip3 install -r requirements.txt
```

## Project 1: DNN Pruning via NNI

Applied pruning to DNNs. Explored various pruning techniques provided by NNI, such as $L1$ and $L2$ norm pruning, level pruner, FPGM pruner, and many more.

## Project 2: DNN Quantization via NNI

Applied and explored various quantization techniques to DNNs. That included naive quantiation from 32-bit single-precision floating point numbers to 16-bit half-precision floating point numbers. We also explored quantization-aware training, which is a technique that trains the model with quantization in mind. Others included DoReFa, Binurization, and many more.

## Project 3: DNN HPO via NNI

Applied Hyperparameter Optimization (HPO) to DNNs. This involved setting a search space of hyperparameters and letting NNI search via specific algorithms for convergence, such as grid search, random search, and evolutionary algorithms.

## Project 4: Distillation and Compilation

Applied distillation to DNNs. We distilled a large model (ResNet 101) into a smaller model (ResNet 18) wihtout sacrificing much accuracy. Then we compiled them via TVM to execute on CPUs and GPUs at a lower latency that removes the overhead from PyTorch and Python.

## Final Course Project: Accelerating Video Super Resolution for Mobile Device

Took a Super Resolution model, and applied all the techniques we learned in the course to accelerate it for mobile devices. This included pruning, quantization, distillation, HPO, and compilation. We found that pruning and HPO definitely helped. Compilation enabled the model to run natively on mobile devices.

We used [CoCoPIE's XGen](https://www.cocopie.ai/#/xgen) to compile and run the model on an Android device to achieve real-time super resolution inference.

A copy of the repo by itself is located in [here](https://github.com/briancpark/video-super-resolution).
