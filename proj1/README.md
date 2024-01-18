# Project 1: DNN Pruning via NNI

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

For M1 Mac, installation needs to be specialized.

For example, NNI needs to be built from [source](https://nni.readthedocs.io/en/stable/notes/build_from_source.html) as Microsoft does not release the wheels for ARM architecture yet:

```sh
git clone https://github.com/microsoft/nni.git
cd nni
pip install --upgrade setuptools pip wheel
python setup.py develop
```

For PyTorch installation, we can now utilize the [M1 GPU](https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c) thanks to the Metal Performance Shaders team! Should be included in the most recent releases!

To make code hardware agnostic and conveniently switch backends between different machines, just slap this line of code to automatically detect the backends:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```

## Running the Project

To simply run everything, just run this command:

```sh
python3 proj1.py
```

You can also run parts of the project individually. For example, training will take a very long time:

```sh
python3 proj1.py --train     # train DNNs
python3 proj1.py --prune     # prune DNNs
python3 proj1.py --figures   # plot figures for report
python3 proj1.py --benchmark # benchmark normal vs. pruned DNNs
```
