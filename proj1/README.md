# Project 1: DNN Pruning via NNI

## Installation
For ARC or local Linux/UNIX system, installation should be easy:
```
pip3 install -r requirements.txt
```

For M1 Mac, installation needs to be specialized:

For example, NNI needs to be build from [source](https://nni.readthedocs.io/en/stable/notes/build_from_source.html) as Microsoft does not release the wheels for ARM architecture:
```sh
git clone https://github.com/microsoft/nni.git
cd nni
pip install --upgrade setuptools pip wheel
python setup.py develop
```

For PyTorch installation, we can now utilize the [M1 GPU](https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c) thanks to the Metal Performance Shaders team! Should be included in the most recent releases!

To make code conveniently switch backends between different machines, just slap this line of code to automatically detect the backends:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```

```sh
cd examples/mnist
pip3 install -r requirements.txt
```


To run jupyter notebook on ARC cluster with VSCode extension make sure to run these commands:
```sh
jupyter-notebook --ip=cXX *.ipynb
module load cuda
```

