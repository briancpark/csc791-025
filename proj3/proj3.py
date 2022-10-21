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

if torch.cuda.is_available():
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def hpo(device, tuner, advanced=False, use_vgg=False):
    name = "VGG-11 HPO " if use_vgg else "MLP HPO "
    name += tuner
    name += " Advanced" if advanced else " Default"

    experiment = Experiment("local")
    experiment.config.trial_code_directory = "."
    experiment.config.experiment_working_directory = "experiments"
    experiment.config.experiment_name = name
    if use_vgg:
        experiment.config.trial_command = "python model.py vgg"
    else:
        experiment.config.trial_command = "python model.py n"

    search_space = {
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
        "momentum": {"_type": "uniform", "_value": [0, 1]},
        "batch_size": {
            "_type": "choice",
            "_value": [1, 4, 32, 64, 128, 256],
        },
    }
    if not use_vgg:
        search_space["features"] = {"_type": "choice", "_value": [128, 256, 512, 1024]}

    experiment.config.search_space = search_space
    experiment.config.tuner.name = tuner
    experiment.config.tuner.class_args["optimize_mode"] = "maximize"

    if advanced:
        if tuner == "TPE":
            config.tuner.class_args = {
                "seed": 42,
                "tpe_args": {
                    "constant_liar_type": "mean",
                    "n_startup_jobs": 10,
                    "n_ei_candidates": 20,
                    "linear_forgetting": 50,
                    "prior_weight": 0.9,
                    "gamma": 0.1,
                },
            }
        if tuner == "Evolution":
            config.tuner.class_args = {"population_size": 50}
        if tuner == "Hyperband":
            experiment.config.tuner.class_args = {
                "optimize_mode": "maximize",
                "R": 60,
                "eta": 3,
                "exec_mode": "parallelism",
            }
    else:
        # Default to minimal config
        if tuner == "Hyperband":
            experiment.config.tuner.class_args = {
                "optimize_mode": "maximize",
                "R": 100,
                "eta": 25,
                "exec_mode": "parallelism",
            }

    experiment.config.max_trial_number = 20
    if use_vgg:
        experiment.config.trial_concurrency = 5
    else:
        experiment.config.trial_concurrency = 20

    experiment.run(8080)


if __name__ == "__main__":
    if sys.argv[1] == "view":
        nni.experiment.Experiment.view(sys.argv[2])

    elif sys.argv[1] == "hpo":
        use_vgg = sys.argv[3] == "vgg"
        advanced = sys.argv[4] == "advanced"

        hpo(device, sys.argv[2], advanced, use_vgg)
    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py train")
