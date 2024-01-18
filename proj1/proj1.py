"""Project 1: DNN Pruning via NNI"""

import os
import glob
import time
import argparse
import torch

import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision import models
from nni.compression.pytorch.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner,
    ADMMPruner,
)
from nni.compression.pytorch.speedup import ModelSpeedup
from torchviz import make_dot
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=redefined-outer-name,invalid-name,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-locals,not-callable
# pylint: disable=too-many-statements,pointless-exception-statement,protected-access

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

# Set other global parameters
batch_size = 64
test_batch_size = 1000
epochs = 180
lr = 1.0
gamma = 0.7
seed = 1
save_model = False
criterion = nn.CrossEntropyLoss()
TRIALS = 25
num_cpus = int(os.cpu_count() / 2)
sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Setup directory and environment variables
arc_env = os.path.exists("/mnt/beegfs/" + os.environ["USER"])
os.system("mkdir -p figures")
os.system("mkdir -p models")

if torch.cuda.is_available():
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag
    # defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class Net(nn.Module):
    """Very simple CNN for MNIST"""

    def __init__(self):
        """Initialize the model"""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def touch():
    """For benchmarking purposes, we need to add synchronization primitives"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train(model, device, train_loader, optimizer, criterion, epoch):
    """Train the model"""
    model.train()

    t_iter = tqdm(
        train_loader, position=1, desc=str(epoch), leave=False, colour="yellow"
    )

    for _, (data, target) in enumerate(t_iter):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        t_iter.set_description(f"Loss: {loss.item():.4f}", refresh=False)


def test(model, device, test_loader, criterion):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_dataset_length = len(test_loader.dataset)
    test_loss /= test_dataset_length
    accuracy = 100.0 * correct / test_dataset_length
    return test_loss, correct, test_dataset_length, accuracy


def load_data(train_kwargs, test_kwargs, mnist=True):
    """Load the data"""
    if arc_env:
        data_dir = "/mnt/beegfs/" + os.environ["USER"] + "/data/"
    else:
        data_dir = "data"

    if mnist:
        # The transformations were copied from the PyTorch MNIST example
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset1 = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST(data_dir, train=False, transform=transform)
    else:
        # The transformations were copied from
        # https://www.programcreek.com/python/example/105099/torchvision.datasets.CIFAR100
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                    std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
                ),
            ]
        )

        dataset1 = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        dataset2 = datasets.CIFAR10(data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def train_models(resnet=False, retrain=False):
    """ "Train the models"""
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # If we're using NVIDIA, we can apply some more software/hardware
    # optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet101.pt"
        model = models.resnet101().to(device)
    else:
        model_save_path = "models/mnist_cnn.pt"
        model = Net().to(device)

    # If we were in the middle of training, reload the model.
    if os.path.exists(model_save_path) or retrain:
        model = torch.load(model_save_path, map_location=device)

    if resnet:
        # CIFAR-10 ResNet-101
        # Reccomended hyperparameters are here:
        # https://discuss.pytorch.org/t/resnet-with-cifar10-only-reaches-86-accuracy-expecting-90/135051
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135])
        criterion = nn.CrossEntropyLoss()
    else:
        # MNIST CNN
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        criterion = F.nll_loss

    for epoch in tqdm(
        range(1, epochs + 1),
        position=0,
        desc="Epochs      ",
        leave=False,
        colour="green",
    ):
        train(model, device, train_loader, optimizer, criterion, epoch)
        _, _, _, accuracy = test(model, device, test_loader, criterion)
        tqdm.write(f"Accuracy: {accuracy:.4f}")

        scheduler.step()

        # Update the weights and save the model
        torch.save(model, model_save_path)

    test_loss, correct, test_dataset_length, accuracy = test(
        model, device, test_loader, criterion
    )
    print(
        f"Average test loss: {test_loss:.4f}, \
        Accuracy: {correct}/{test_dataset_length} ({accuracy:.0f}%)"
    )


def prune_helper(
    model, opt_pruner, train_loader, test_loader, sparsity, resnet, opt_pruner_name
):
    """Prune helper function for distributed pruning"""
    if resnet:
        config_list = [
            {
                "sparsity_per_layer": sparsity,
                "op_types": [
                    "Conv2d",
                ],
            },
            {"exclude": True, "op_names": ["fc"]},
        ]
        retrain_epochs = 20
    else:
        config_list = [
            {"sparsity_per_layer": sparsity, "op_types": ["Conv2d", "Linear"]},
            {"exclude": True, "op_names": ["fc2"]},
        ]
        retrain_epochs = 3
    print(model)

    pruner = opt_pruner(model, config_list)

    print(model)

    pruned_model_save_dir = "models/" + opt_pruner_name
    os.system("mkdir -p " + pruned_model_save_dir)

    # compress the model and generate the masks
    model, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(f"{name} sparsity: {mask['weight'].sum() / mask['weight'].numel():.2f}")

    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    if resnet:
        rand_tensor = torch.rand(64, 3, 28, 28).to(device)
    else:
        rand_tensor = torch.rand(64, 1, 28, 28).to(device)

    ModelSpeedup(model, rand_tensor, masks).speedup_model()

    optimizer = SGD(model.parameters(), 1e-2)
    for epoch in tqdm(
        range(1, retrain_epochs + 1),
        position=0,
        desc="Epochs      ",
        leave=False,
        colour="green",
    ):
        train(model, device, train_loader, optimizer, criterion, epoch)

    test_loss, correct, test_dataset_length, accuracy = test(
        model, device, test_loader, criterion
    )
    tqdm.write(
        f"Average test loss: {test_loss:.4f}, \
        Accuracy: {correct}/{test_dataset_length} ({accuracy:.0f}%)"
    )

    # Save the models with their respective sparsities and pruning methods
    if resnet:
        torch.save(
            model,
            f"models/{opt_pruner_name}/cifar10_resnet101_pruned_sparsity_{int(sparsity * 100)}.pt",
        )
    else:
        torch.save(
            model,
            f"models/{opt_pruner_name}/mnist_cnn_pruned_sparsity_{int(sparsity * 100)}.pt",
        )

    return accuracy


def prune(device, resnet=False, use_ray=False):
    """
    Prune the models
    if use_ray is True, use Ray to parallelize the pruning process
    """
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet101.pt"
    else:
        model_save_path = "models/mnist_cnn.pt"

    model = torch.load(model_save_path, map_location=device)

    if device.type == "cuda":
        cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = model.to(device)

    oids = []

    if use_ray:
        import ray

        # Tune these parameters to liking depending on hardware environment
        ray.init(num_gpus=1, include_dashboard=False, num_cpus=num_cpus)
        r = ray.remote(num_gpus=0.25)
        remote_fn = r(prune_helper)

    opt_pruners = {
        "LevelPruner": LevelPruner,
        "L1NormPruner": L1NormPruner,  # Used in benchmark
        "L2NormPruner": L2NormPruner,
        "FPGMPruner": FPGMPruner,
        "SlimPruner": SlimPruner,
        "ActivationAPoZRankPruner": ActivationAPoZRankPruner,
        "ActivationMeanRankPruner": ActivationMeanRankPruner,
        "TaylorFOWeightPruner": TaylorFOWeightPruner,
        "ADMMPruner": ADMMPruner,
    }

    for opt_pruner_name, opt_pruner in opt_pruners.items():
        for sparsity in sparsities:
            if use_ray:
                oid = remote_fn.remote(
                    model,
                    opt_pruner,
                    train_loader,
                    test_loader,
                    sparsity,
                    resnet,
                    opt_pruner_name,
                )
            else:
                oid = prune_helper(
                    model,
                    opt_pruner,
                    train_loader,
                    test_loader,
                    sparsity,
                    resnet,
                    opt_pruner_name,
                )
            oids.append(oid)

    if use_ray:
        results = ray.get(oids)
        print(results)

    print(oids)


def figures(device):
    """Plot the figures"""
    cnn_model = Net().to(device)
    resnet18_model = models.resnet18().to(device)

    cnn_yhat = cnn_model(torch.rand(1, 1, 28, 28).to(device))
    resnet18_yhat = resnet18_model(torch.rand(1, 3, 224, 224).to(device))

    make_dot(cnn_yhat, params=dict(list(cnn_model.named_parameters()))).render(
        "figures/mnist_cnn", format="png"
    )

    make_dot(
        resnet18_yhat, params=dict(list(resnet18_model.named_parameters()))
    ).render("figures/resnet18", format="png")


def benchmark(device, pruner_name, resnet=False):
    """Benchmark the pruned models"""
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # If we're using NVIDIA, we can apply some more software/hardware
    # optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    pruned_model_save_dir = "models/" + pruner_name

    if resnet:
        model_save_path = "models/cifar10_resnet101.pt"
        pruned_model_save_path = (
            pruned_model_save_dir + "/cifar10_resnet101_pruned_sparsity_*"
        )

    else:
        model_save_path = "models/mnist_cnn.pt"
        pruned_model_save_path = pruned_model_save_dir + "/mnist_cnn_pruned_sparsity_*"

    model = torch.load(model_save_path, map_location=device)

    # Warmup, CUDA typically has overhead on the first run
    for _ in range(5):
        test(model, device, test_loader, criterion)

    times = []
    for _ in range(TRIALS):
        tik = time.time()
        touch()
        test_loss, correct, test_dataset_length, accuracy = test(
            model, device, test_loader, criterion
        )
        touch()
        tok = time.time()
        total_time = tok - tik
        times.append(total_time)
        _, _, _, baseline_train_accuracy = test(model, device, train_loader, criterion)
        print(
            f"Average test loss: {test_loss:.4f}, \
            Train Accuracy ({baseline_train_accuracy:.0f}%), \
            Val Accuracy: {correct}/{test_dataset_length} ({accuracy:.0f}%)"
        )

    baseline_time = sum(times) / len(times)
    baseline_accuracy = accuracy
    print("Average time: ", baseline_time)

    # Benchmark the pruned models
    exec_times = []
    exec_stds = []
    accuracies = []
    train_accuracies = []

    model_fns = sorted(glob.glob(pruned_model_save_path))

    for model_fn in tqdm(model_fns):
        model = torch.load(model_fn, map_location=device)
        model = model.to(device)

        times = []
        accuracies_subtrials = []
        training_accuracy_subtrials = []

        for _ in range(TRIALS):
            tik = time.time()
            touch()
            test_loss, correct, test_dataset_length, accuracy = test(
                model, device, test_loader, criterion
            )
            touch()
            tok = time.time()
            total_time = tok - tik
            times.append(total_time)
            _, _, _, train_accuracy = test(model, device, train_loader, criterion)
            tqdm.write(
                f"Average test loss: {test_loss:.4f}, \
                Train Accuracy ({train_accuracy:.0f}%), \
                Val Accuracy: {correct}/{test_dataset_length} ({accuracy:.0f}%)"
            )
            accuracies_subtrials.append(accuracy)
            training_accuracy_subtrials.append(train_accuracy)

        exec_time = sum(times) / len(times)
        exec_time_std = np.std(times)
        accuracies_avg = sum(accuracies_subtrials) / len(accuracies_subtrials)
        training_accuracy_avg = sum(training_accuracy_subtrials) / len(
            training_accuracy_subtrials
        )

        exec_times.append(exec_time)
        exec_stds.append(exec_time_std)
        accuracies.append(accuracies_avg)
        train_accuracies.append(training_accuracy_avg)

        tqdm.write(f"Average accuracy: {accuracies_avg:.4f}")

    # Convert to numpy arrays for vectorization computation
    accuracies = np.array(accuracies)
    exec_times = np.array(exec_times)
    exec_stds = np.array(exec_stds)

    # Plot the results
    plt.figure(figsize=(10, 10), dpi=400)
    ax1 = plt.subplot()
    p0 = ax1.plot(sparsities, exec_times, label="Inference Time")
    ax1.fill_between(
        sparsities,
        exec_times - exec_stds,
        exec_times + exec_stds,
        color="blue",
        alpha=0.2,
    )
    p1 = ax1.plot(
        sparsities,
        [baseline_time for _ in sparsities],
        color="blue",
        linestyle="--",
        label="Baseline Inference Time (No Pruning)",
    )
    plt.xlabel("Pruning (%)")
    plt.ylabel("Inference Time (s)")

    ax2 = ax1.twinx()
    p2 = ax2.plot(sparsities, accuracies, label="Validation Accuracy", color="green")
    p3 = ax2.plot(
        sparsities,
        [baseline_accuracy for _ in sparsities],
        color="green",
        linestyle="--",
        label="Baseline Validation Accuracy (No Pruning)",
    )
    p4 = ax2.plot(
        sparsities, train_accuracies, label="Training Accuracy", color="orange"
    )
    p5 = ax2.plot(
        sparsities,
        [baseline_train_accuracy for _ in sparsities],
        color="orange",
        linestyle="--",
        label="Baseline Training Accuracy (No Pruning)",
    )

    plt.ylabel("Accuracy (%)")
    leg = p0 + p1 + p2 + p3 + p4 + p5
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc="lower left")
    if resnet:
        plt.title("CIFAR-10 ResNet-101 Pruning Benchmark (Inference Time and Accuracy)")
        plt.savefig("figures/resnet101_benchmark.png")
    else:
        plt.title("MNIST CNN Pruning Benchmark (Inference Time and Accuracy)")
        plt.savefig("figures/mnist_cnn_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--prune", action="store_true", help="Prune models")
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark models")

    args = parser.parse_args()

    if args.train:
        train_models(resnet=True, retrain=False)

    elif args.prune:
        if device.type == "mps":
            UserWarning("Cannot perform pruning with NNI on MPS mode, fallback to CPU")
            device = torch.device("cpu")
        prune(device, resnet=True)

    elif args.figures:
        if device.type == "mps":
            UserWarning("Cannot generate graphs on MPS mode, fallback to CPU")
            device = torch.device("cpu")
        figures(device)

    elif args.benchmark:
        benchmark(device, "L1NormPruner", resnet=True)

    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py --train")
