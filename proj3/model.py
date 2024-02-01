"""PyTorch model for training on FashionMNIST dataset"""

import argparse
import os
import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor


# pylint: disable=redefined-outer-name,invalid-name,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-locals,not-callable,too-many-branches
# pylint: disable=too-many-statements,pointless-exception-statement,protected-access

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device.type.upper())

# Set Random Seed
torch.manual_seed(42)

params = {
    "features": 512,
    "lr": 0.001,
    "momentum": 0,
    "batch_size": 64,
}

if torch.cuda.is_available():
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


arc_env = os.path.exists("/mnt/beegfs/" + os.environ["USER"])
a100_env = os.path.exists("/mnt/local/" + os.environ["USER"])

if a100_env:
    data_dir = "/mnt/local/" + os.environ["USER"] + "/data/"
elif arc_env:
    data_dir = "/mnt/beegfs/" + os.environ["USER"] + "/data/"
else:
    data_dir = "data"


class MLP(nn.Module):
    """MLP Model"""

    def __init__(self):
        """Initialize the model"""
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, params["features"]),
            nn.ReLU(),
            nn.Linear(params["features"], params["features"]),
            nn.ReLU(),
            nn.Linear(params["features"], 10),
        )

    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    """Training loop"""
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    """Test loop"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--vgg", action="store_true", help="Train vgg")

    args = parser.parse_args()

    use_vgg = args.vgg

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    if use_vgg:
        training_data = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=ToTensor()
        )
        test_data = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=ToTensor()
        )
    else:
        training_data = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=ToTensor()
        )
        test_data = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=ToTensor()
        )

    batch_size = params["batch_size"]

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if use_vgg:
        model = models.vgg11().to(device)
    else:
        model = MLP().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=params["lr"], momentum=params["momentum"]
    )

    if use_vgg:
        epochs = 20
    else:
        epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)
