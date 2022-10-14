import torch
import sys
import os
import csv
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchviz import make_dot
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
from pathlib import Path

from nni.algorithms.compression.pytorch.quantization import (
    NaiveQuantizer,
    QAT_Quantizer,
    DoReFaQuantizer,
    BNNQuantizer,
    ObserverQuantizer,
)

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
TRIALS = 5
num_cpus = int(os.cpu_count() / 2)

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


# For benchmarking purposes, we need to add synchronization primitives
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
    if arc_env:
        dir = "/mnt/beegfs/" + os.environ["USER"] + "/data/"
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


def train_models(resnet=False, quantizer="", retrain=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # If we're using NVIDIA, we can apply some more software/hardware optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        if quantizer:
            model_save_path = "models/" + quantizer + "/cifar10_resnet101_quantized.pt"
            model = torch.load(model_save_path, map_location=device)
        else:
            model_save_path = "models/cifar10_resnet101.pt"
            model = models.resnet101().to(device)
        logger_fn = "logs/cifar10_resnet101_" + quantizer + ".csv"
    else:
        if quantizer:
            model_save_path = "models/" + quantizer + "/mnist_cnn_quantized.pt"
            model = torch.load(model_save_path, map_location=device)
        else:
            model_save_path = "models/mnist_cnn.pt"
            model = Net().to(device)
        logger_fn = "logs/mnist_cnn_" + quantizer + ".csv"

    # If we were in the middle of training, reload the model.
    if os.path.exists(model_save_path) or retrain:
        model = torch.load(model_save_path, map_location=device)

    if resnet:
        # CIFAR-10 ResNet-101
        # Reccomended hyperparameters are here: https://discuss.pytorch.org/t/resnet-with-cifar10-only-reaches-86-accuracy-expecting-90/135051
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135])
        criterion = nn.CrossEntropyLoss()
    else:
        # MNIST CNN
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        criterion = F.nll_loss

    # For logging purposes
    if not os.path.exists(logger_fn):
        with open(logger_fn, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "train_loss",
                    "train_correct",
                    "train_dataset_length",
                    "train_accuracy",
                    "test_loss",
                    "test_correct",
                    "test_dataset_length",
                    "test_accuracy",
                ]
            )

    for epoch in tqdm(
        range(1, epochs + 1),
        position=0,
        desc="Epochs      ",
        leave=False,
        colour="green",
    ):
        train(model, device, train_loader, optimizer, criterion, epoch)
        train_loss, train_correct, train_dataset_length, train_accuracy = test(
            model, device, train_loader, criterion
        )
        test_loss, test_correct, test_dataset_length, test_accuracy = test(
            model, device, test_loader, criterion
        )

        # For logging purposes
        with open(logger_fn, "a") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    train_loss,
                    train_correct,
                    train_dataset_length,
                    train_accuracy,
                    test_loss,
                    test_correct,
                    test_dataset_length,
                    test_accuracy,
                ]
            )
        tqdm.write("Accuracy: %.4f" % test_accuracy)
        scheduler.step()

        ### Update the weights and save the model
        torch.save(model, model_save_path)
        if test_accuracy > 99.0:
            break

    test_loss, correct, test_dataset_length, accuracy = test(
        model, device, test_loader, criterion
    )
    print(
        "Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, test_dataset_length, accuracy
        )
    )


def hpo(device):
    model = models.resnet101().to(device)


    print("Starting HPO")
    return
    # train_kwargs = {"batch_size": batch_size}
    # test_kwargs = {"batch_size": test_batch_size}
    #
    # train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)
    #
    # if device.type == "cuda":
    #     cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)
    #
    # os.system("mkdir -p models/" + quantizer)
    #
    # if quantizer == "ObserverQuantizer":
    #     # Observer Quantizer is the only one that is done post-training
    #     if resnet:
    #         model = torch.load("models/cifar10_resnet101.pt", map_location=device)
    #     else:
    #         model = torch.load("models/mnist_cnn.pt", map_location=device)
    #
    #     config_list = [
    #         {
    #             "quant_types": ["weight", "input"],
    #             "quant_bits": {"weight": 8, "input": 8},
    #             "op_types": ["Conv2d", "Linear"],
    #         }
    #     ]
    #     quantizer = ObserverQuantizer(model, config_list)
    #
    #     def calibration(model, train_loader):
    #         model.eval()
    #
    #         with torch.no_grad():
    #             for data, _ in train_loader:
    #                 model(data)
    #
    #     calibration(model, train_loader)
    #     quantizer.compress()
    #
    #     if resnet:
    #         torch.save(model, "models/ObserverQuantizer/cifar10_resnet101_quantized.pt")
    #     else:
    #         torch.save(model, "models/ObserverQuantizer/mnist_cnn_quantized.pt")
    #
    #     return
    #
    # if resnet:
    #     model = models.resnet101().to(device)
    #     model_save_path = "models/" + quantizer + "/cifar10_resnet101"
    # else:
    #     model = Net().to(device)
    #     model_save_path = "models/" + quantizer + "/mnist_cnn"
    #
    # print(model)
    #
    # op_types = ["Conv2d", "Linear"]
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    #
    # if quantizer == "NaiveQuantizer":
    #     quant_bits = 8
    #
    #     config_list = [
    #         {
    #             "quant_types": ["weight"],
    #             "quant_bits": {
    #                 "weight": quant_bits,
    #             },
    #             "op_types": op_types,
    #         }
    #     ]
    #
    #     quantizer = NaiveQuantizer(model, config_list)
    #
    # elif quantizer == "BNNQuantizer":
    #     quant_bits = 1
    #
    #     config_list = [
    #         {
    #             "quant_types": ["weight"],
    #             "quant_bits": {"weight": quant_bits},
    #             "op_types": op_types,
    #         }
    #     ]
    #
    #     quantizer = BNNQuantizer(model, config_list, optimizer)
    #
    # elif quantizer == "DoReFaQuantizer":
    #     quant_bits = 8
    #
    #     config_list = [
    #         {
    #             "quant_types": ["weight"],
    #             "quant_bits": {"weight": quant_bits},
    #             "op_types": op_types,
    #         }
    #     ]
    #
    #     quantizer = DoReFaQuantizer(model, config_list, optimizer)
    #
    # elif quantizer == "QAT_Quantizer":
    #     if resnet:
    #         dummy_input = torch.rand(1, 3, 224, 224).to(device)
    #     else:
    #         dummy_input = torch.rand(1, 1, 28, 28).to(device)
    #
    #     quant_bits = 8
    #
    #     config_list = [
    #         {
    #             "quant_types": ["weight"],
    #             "quant_bits": {"weight": quant_bits},
    #             "op_types": op_types,
    #         }
    #     ]
    #
    #     quantizer = QAT_Quantizer(
    #         model, config_list, optimizer, dummy_input=dummy_input
    #     )
    #
    # quantizer.compress()
    #
    # print(model)
    #
    # quantized_model_save_path = model_save_path + "_quantized.pt"
    #
    # torch.save(model, quantized_model_save_path)


def figures(device):
    # Plot the computational graphs of the models
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

    # For BNN, print the number of ones and negative ones in the weights
    model_fns = [
        "models/BNNQuantizer/cifar10_resnet101_quantized.pt",
        "models/BNNQuantizer/mnist_cnn_quantized.pt",
    ]

    for model_fn in model_fns:
        model = torch.load(model_fn, map_location=device)
        layers = [module for module in model.modules()]

        ones, neg_ones = 0, 0
        for layer in layers:
            if layer.__class__.__name__ == "QuantizerModuleWrapper":
                ones += torch.sum(layer.module.weight.data == 1).item()
                neg_ones += torch.sum(layer.module.weight.data == -1).item()

        print("Number of 1s: {}, Number of -1s: {}".format(ones, neg_ones))
        # Plot the accuracy and loss graphs
        files = os.listdir("logs")

    for file in files:
        fn = "logs/" + str(Path(file))
        save_fn = "figures/" + Path(file).stem + ".png"
        quantizer = Path(file).stem.split("_")[-1]
        df = pd.read_csv(fn)
        epochs = list(range(1, len(df) + 1))

        plt.figure(figsize=(10, 10), dpi=400)
        ax1 = plt.subplot()
        p0 = ax1.plot(
            epochs, df[["train_accuracy"]], label="Training Accuracy", color="blue"
        )
        p1 = ax1.plot(
            epochs, df[["test_accuracy"]], label="Validation Accuracy", color="green"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")

        ax2 = ax1.twinx()
        p2 = ax2.plot(epochs, df[["train_loss"]], label="Training Loss", color="orange")
        p3 = ax2.plot(epochs, df[["test_loss"]], label="Validation Loss", color="red")
        plt.ylabel("Loss")

        leg = p0 + p1 + p2 + p3
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc="center right")

        model_type = str(Path(file)).split("_")[0]
        if model_type == "cifar10":
            model_title = "CIFAR-10 ResNet-101"
        else:
            model_title = "MNIST CNN"

        plt.title(
            model_title
            + " Quantization with "
            + quantizer
            + " Training and Validation Accuracy/Loss Over Epochs"
        )
        plt.savefig(save_fn)
        plt.clf()


def benchmark(device, resnet=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # If we're using NVIDIA, we can apply some more software/hardware optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": num_cpus, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = load_data(train_kwargs, test_kwargs, mnist=not resnet)

    if resnet:
        model_save_path = "models/cifar10_resnet101.pt"
    else:
        model_save_path = "models/mnist_cnn.pt"

    model = torch.load(model_save_path, map_location=device)

    ### Warmup, CUDA typically has overhead on the first run
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
            "Average test loss: {:.4f}, Train Accuracy ({:.0f}%), Val Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                baseline_train_accuracy,
                correct,
                test_dataset_length,
                accuracy,
            )
        )

    baseline_time = sum(times) / len(times)
    baseline_std_time = np.std(times)
    baseline_accuracy = accuracy
    print("Average time: ", baseline_time)

    # Benchmark the pruned models
    exec_times = []
    exec_stds = []
    accuracies = []
    train_accuracies = []

    # Based on our training session, we pick the ones that actually trained properly
    if resnet:
        model_names = ["DoReFaQuantizer", "BNNQuantizer"]
        model_fns = [
            "models/DoReFaQuantizer/cifar10_resnet101_quantized.pt",
            "models/BNNQuantizer/cifar10_resnet101_quantized.pt",
        ]
    else:
        model_names = ["DoReFaQuantizer", "NaiveQuantizer", "QAT_Quantizer"]
        model_fns = [
            "models/DoReFaQuantizer/mnist_cnn_quantized.pt",
            "models/NaiveQuantizer/mnist_cnn_quantized.pt",
            "models/QAT_Quantizer/mnist_cnn_quantized.pt",
        ]

    for model_fn in tqdm(model_fns):
        model = torch.load(model_fn, map_location=device)

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
                "Average test loss: {:.4f}, Train Accuracy ({:.0f}%), Val Accuracy: {}/{} ({:.0f}%)".format(
                    test_loss, train_accuracy, correct, test_dataset_length, accuracy
                )
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

        tqdm.write("Average time: {0}".format(sum(times) / len(times)))

    # Convert to numpy arrays for vectorization computation
    accuracies = np.array(accuracies)
    exec_times = np.array(exec_times)
    exec_stds = np.array(exec_stds)

    # Plot the results
    plt.figure(figsize=(10, 10), dpi=400)
    barWidth = 0.25
    br1 = np.arange(len(model_names))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, exec_times, width=barWidth, label="Execution Time")
    plt.bar(br2, accuracies, width=barWidth, label="Accuracy")
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)

    if resnet:
        plt.title(
            "CIFAR-10 ResNet-101 Quantization Benchmark (Inference Time and Accuracy)"
        )
        plt.savefig("figures/resnet101_benchmark.png")
    else:
        plt.title("MNIST CNN Pruning Benchmark (Inference Time and Accuracy)")
        plt.savefig("figures/mnist_cnn_benchmark.png")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_models(resnet=True)
        train_models(resnet=False)

    elif sys.argv[1] == "train":
        train_models(resnet=True, quantizer="NaiveQuantizer")
        # train_models(resnet=False, quantizer="BNNQuantizer")
        # train_models(resnet=False, quantizer="QAT_Quantizer")
        # train_models(resnet=False, quantizer="DoReFaQuantizer")

    elif sys.argv[1] == "hpo":
        hpo(device)
    elif sys.argv[1] == "figures":
        figures(device)

    elif sys.argv[1] == "benchmark":
        benchmark(device, resnet=False)

    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py train")
