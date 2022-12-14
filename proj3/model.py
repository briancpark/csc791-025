import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import os
import sys

if sys.argv[1] == "vgg":
    use_vgg = True
else:
    use_vgg = False

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


optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

arc_env = os.path.exists("/mnt/beegfs/" + os.environ["USER"])
a100_env = os.path.exists("/mnt/local/" + os.environ["USER"])

if a100_env:
    dir = "/mnt/local/" + os.environ["USER"] + "/data/"
elif arc_env:
    dir = "/mnt/beegfs/" + os.environ["USER"] + "/data/"
else:
    dir = "data"

if use_vgg:
    training_data = datasets.CIFAR10(
        root=dir, train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.CIFAR10(
        root=dir, train=False, download=True, transform=ToTensor()
    )
else:
    training_data = datasets.FashionMNIST(
        root=dir, train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=dir, train=False, download=True, transform=ToTensor()
    )

batch_size = params["batch_size"]

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, params["features"]),
            nn.ReLU(),
            nn.Linear(params["features"], params["features"]),
            nn.ReLU(),
            nn.Linear(params["features"], 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if use_vgg:
    model = models.vgg11().to(device)
else:
    model = MLP().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=params["lr"], momentum=params["momentum"]
)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
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
