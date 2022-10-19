import torch
import sys
import os
import nni
import torch.nn as nn
from nni.experiment import Experiment


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device.type.upper())

# Set Random Seed
torch.manual_seed(42)

# Setup directory and environment variables
arc_env = os.path.exists("/mnt/beegfs/" + os.environ["USER"])
os.system("mkdir -p figures")
os.system("mkdir -p models")
os.system("mkdir -p logs")

if torch.cuda.is_available():
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def hpo(device, tuner, batch="", use_vgg=False):
    name = "VGG-19 HPO " + tuner
    if batch == "batch":
        name += " Batch"
    experiment = Experiment("local")
    experiment.config.trial_code_directory = "."
    experiment.config.experiment_working_directory = "experiments"
    experiment.config.experiment_name = name
    if use_vgg:
        experiment.config.trial_command = "python model.py vgg"
    else:
        experiment.config.trial_command = "python model.py n"

    search_space = {
        "features": {"_type": "choice", "_value": [128, 256, 512, 1024]},
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
        "momentum": {"_type": "uniform", "_value": [0, 1]},
    }
    if batch == "batch":
        search_space["batch_size"] = {
            "_type": "choice",
            "_value": [1, 4, 32, 64, 128, 256],
        }

    experiment.config.search_space = search_space
    experiment.config.tuner.name = tuner
    experiment.config.tuner.class_args["optimize_mode"] = "maximize"
    if tuner == "Hyperband":
        experiment.config.tuner.class_args = {
            "optimize_mode": "maximize",
            "R": 60,
            "eta": 3,
        }

    experiment.config.max_trial_number = 10
    if use_vgg:
        experiment.config.trial_concurrency = 100
    else:
        experiment.config.trial_concurrency = 10

    experiment.run(8080)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        nni.experiment.Experiment.view("godeaxlj")

    elif sys.argv[1] == "hpo":
        if sys.argv[3] == "vgg":
            use_vgg = True
        else:
            use_vgg = False
        hpo(device, sys.argv[2], sys.argv[3], use_vgg)
    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py train")
