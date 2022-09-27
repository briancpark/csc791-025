import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from tqdm import tqdm
from torchviz import make_dot, make_dot_from_trace
import sys
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device.type.upper())

batch_size = 64
test_batch_size = 1000
epochs = 100
lr = 1.0
gamma = 0.7
seed = 1
save_model = False
criterion = nn.CrossEntropyLoss()
TRIALS = 5

arc_env = os.path.exists("/mnt/beegfs/$USER")
os.system("mkdir -p figures")
os.system("mkdir -p models")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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


# For benchmarking purposes, we need to add synchonizaiton primitives
def touch():
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()

    t_iter = tqdm(
        train_loader, position=1, desc=str(epoch), leave=False, colour="yellow"
    )

    for batch_idx, (data, target) in enumerate(t_iter):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        t_iter.set_description("Loss: %.4f" % loss.item(), refresh=False)


def test(model, device, test_loader, criterion):
    touch()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(
                output,
                target  # , reduction="sum"
                # TODO: Figure out the significance of reduction argument.
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    touch()

    test_dataset_length = len(test_loader.dataset)
    test_loss /= test_dataset_length
    accuracy = 100.0 * correct / test_dataset_length
    return test_loss, correct, test_dataset_length, accuracy


def load_data(train_kwargs, test_kwargs, mnist=True):
    if arc_env:
        dir = "/mnt/beegfs/$USER/data/"
    else:
        dir = "data"

    if mnist:
        # The transformations were copied from the PyTorch MNIST example
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset1 = datasets.MNIST(dir, train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST(dir, train=False, transform=transform)
    else:
        # The transformations were copied from https://www.programcreek.com/python/example/105099/torchvision.datasets.CIFAR100
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

        dataset1 = datasets.CIFAR10(dir, train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10(dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def train_models(resnet=False, retrain=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet18.pt"
        model = models.resnet101().to(device)
    else:
        model_save_path = "models/mnist_cnn.pt"
        model = Net().to(device)

    # If we were in the middle of training, reload the model.
    if os.path.exists(model_save_path) or retrain:
        model = torch.load(model_save_path)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    criterion = F.nll_loss

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    # If we're using NVIDIA, we can apply some more software/hardware optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        # torch.backends.cudnn.benchmark = True

    for epoch in tqdm(
        range(1, epochs + 1),
        position=0,
        desc="Epochs      ",
        leave=False,
        colour="green",
    ):
        train(model, device, train_loader, optimizer, criterion, epoch)
        _, _, _, accuracy = test(model, device, test_loader, criterion)
        tqdm.write("Accuracy: %.4f" % accuracy)
        scheduler.step()

        ### Update the weights and save the model
        torch.save(model, model_save_path)

    test_loss, correct, test_dataset_length, accuracy = test(
        model, device, test_loader, criterion
    )
    print(
        "Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, test_dataset_length, accuracy
        )
    )


def prune_helper(model, train_loader, test_loader, sparsity, resnet):
    config_list = [
        # {"sparsity_per_layer": sparsity, "op_types": ["Conv2d", "Linear"]},
        # {"exclude": True, "op_names": ["fc2"]},
        {"sparsity_per_layer": sparsity, "op_types": ["Conv2d",]},
        {"exclude": True, "op_names": ["fc"]},
    ]
    print(model)
    pruner = L1NormPruner(model, config_list)
    print(model)

    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(
            name,
            " sparsity : ",
            "{:.2}".format(mask["weight"].sum() / mask["weight"].numel()),
        )

    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(64, 3, 28, 28).to(device), masks).speedup_model()
    # ModelSpeedup(model, torch.rand(64, 1, 28, 28).to(device), masks).speedup_model()

    optimizer = SGD(model.parameters(), 1e-2)
    for epoch in tqdm(
        range(1, 4),
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
        "Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, test_dataset_length, accuracy
        )
    )

    if resnet:
        torch.save(
            model, "models/cifar10_resnet18_pruned_sparsity_{0}.pt".format(int(sparsity * 100))
        )
    else:
        torch.save(
            model, "models/mnist_cnn_pruned_sparsity_{0}.pt".format(int(sparsity * 100))
        )


def prune(device, resnet=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet18.pt"
    else:
        model_save_path = "models/mnist_cnn.pt"

    model = torch.load(model_save_path)

    if device.type == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        # torch.backends.cudnn.benchmark = True
        # TODO: this is causing weird issues during the pruning phase

    model = model.to(device)

    oids = []
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # if device.type == "cuda":
    #     import ray

    #     ray.init(num_gpus=1)
    #     r = ray.remote(num_gpus)
    #     prune_helper = r(prune_helper).remote

    for sparsity in sparsities:
        oid = prune_helper(model, train_loader, test_loader, sparsity, resnet)
        oids.append(oid)

    # if device.type == "cuda":
    #     ray.wait(oids)


def figures(device):
    if device.type == "mps":
        UserWarning("Cannot generate graphs on MPS mode, fallback to CPU")
        device = torch.device("cpu")

    cnn_model = Net().to(device)
    resnet18_model = models.resnet18().to(device)

    cnn_yhat = cnn_model(torch.rand(1, 1, 28, 28))
    resnet18_yhat = resnet18_model(torch.rand(1, 3, 224, 224))

    make_dot(cnn_yhat, params=dict(list(cnn_model.named_parameters()))).render(
        "figures/mnist_cnn", format="png"
    )

    make_dot(
        resnet18_yhat, params=dict(list(resnet18_model.named_parameters()))
    ).render("figures/resnet18", format="png")


def benchmark(device, resnet=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet18.pt"
    else:
        model_save_path = "models/mnist_cnn.pt"

    model = torch.load(model_save_path)

    ### Warmup, CUDA typically has overhead on the first run
    for _ in range(5):
        test(model, device, test_loader, criterion)

    times = []
    for _ in range(TRIALS):
        tik = time.time()
        test_loss, correct, test_dataset_length, accuracy = test(
            model, device, test_loader, criterion
        )
        tok = time.time()
        total_time = tok - tik
        times.append(total_time)
        print(
            "Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, test_dataset_length, accuracy
            )
        )

    baseline_time = sum(times) / len(times)
    baseline_std_time = np.std(times)
    print("Average time: ", baseline_time)

    # Benchmark the pruned models
    exec_times = []
    exec_stds = []
    accuracies = []
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]#, 0.7, 0.8, 0.9

    i = 0
    model_fns = glob.glob("models/mnist_cnn_pruned_sparsity_*")
    for model_fn in tqdm(model_fns):
        model = torch.load(model_fn)
        if i == 6:
            break
        i += 1
        model = model.to(device)

        times = []
        accuracies_subtrials = []
        for _ in range(TRIALS):
            tik = time.time()
            test_loss, correct, test_dataset_length, accuracy = test(
                model, device, test_loader, criterion
            )
            tok = time.time()
            total_time = tok - tik
            times.append(total_time)
            tqdm.write(
                "Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    test_loss, correct, test_dataset_length, accuracy
                )
            )
            accuracies_subtrials.append(accuracy)

        exec_time = sum(times) / len(times)
        exec_time_std = np.std(times)
        accuracies_avg = sum(accuracies_subtrials) / len(accuracies_subtrials)

        exec_times.append(exec_time)
        exec_stds.append(exec_time_std)
        accuracies.append(accuracies_avg)

        tqdm.write("Average time: {0}".format(sum(times) / len(times)))

    # Convert to numpy arrays for vectorization computation
    accuracies = np.array(accuracies)
    exec_times = np.array(exec_times)
    exec_stds = np.array(exec_stds)
    baseline_time = np.array(baseline_time)

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
        color="r",
        linestyle="--",
        label="Baseline (No Pruning)",
    )
    plt.xlabel("Sparsity")
    plt.xlabel("Inference Time (s)")

    ax2 = ax1.twinx()
    p2 = ax2.plot(sparsities, accuracies, label="Accuracy", color="green")

    plt.ylabel("Accuracy (%)")
    leg = p0 + p1 + p2
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.title("MNIST CNN Pruning Benchmark (Inference Time and Accuracy)")
    plt.savefig("figures/mnist_cnn_benchmark.png")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_models()
        prune(device)
        figures(device)
        benchmark(device)
    elif sys.argv[1] == "train":
        train_models(resnet=True, retrain=False)
    elif sys.argv[1] == "prune":
        if device.type == "mps":
            UserWarning("Cannot perform pruning with NNI on MPS mode, fallback to CPU")
            device = torch.device("cpu")
        prune(device, resnet=True)
    elif sys.argv[1] == "figures":
        figures(device)
    elif sys.argv[1] == "benchmark":
        benchmark(device, resnet=True)
    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py train")
