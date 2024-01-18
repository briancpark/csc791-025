# Project 2: DNN Quantization via NNI

## Installation and Setup

For ARC or local Linux/UNIX system, installation should be easy. We highly recommend using a Conda environment if possible

```sh
pip3 install -r requirements.txt
```

Sometimes, the CUDA won't be linked properly upon the first installation. For a lot of the nodes on ARC, they have CUDA 11.7. If that's the case please install nightly builds of PyTorch:

```sh
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```

## Running the Project

To simply run everything, just run this command:

```sh
python3 proj2.py
```

You can also run parts of the project individually. For example, training will take a very long time:

```sh
python3 proj2.py --train     # train DNNs
python3 proj2.py --quantize  # quantize DNNs
python3 proj2.py --figures   # plot figures for report
python3 proj2.py --benchmark # benchmark normal vs. quantize DNNs
```
