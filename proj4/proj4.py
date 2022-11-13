import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet152
import os
import csv
import sys
from tqdm import tqdm
import torch.nn.functional as F
from nni.compression.pytorch.pruning import *
from nni.compression.pytorch.speedup import ModelSpeedup

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
    "lr": 0.001,
    "momentum": 0,
}

num_gpus = int(
    os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read()
)

batch_size = 512 * num_gpus
test_batch_size = 1024 * num_gpus
pin_memory = True
kd_T = 4


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    t_iter = tqdm(dataloader)

    for batch, (X, y) in enumerate(dt_iter):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def kd_train(train_loader, model_s, model_t, optimizer):
    cri_kd = DistillKL(kd_T)

    model_s.train()
    model_t.eval()
    size = len(train_loader.dataset)
    t_iter = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(t_iter):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_s = model_s(data)
        y_t = model_t(data)

        loss_cri = F.cross_entropy(y_s, target)
        loss_kd = cri_kd(y_s, y_t)
        # total loss
        loss = loss_cri + loss_kd
        loss.backward()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss


def knowledge_dist():

    if torch.cuda.is_available():
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    dir = "/ocean/datasets/community/imagenet"

    traindir = os.path.join(dir, "train")
    valdir = os.path.join(dir, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    training_data = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_data = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_dataloader = Subset(training_data, range(50000))
    test_dataloader = Subset(test_data, range(10000))

    train_dataloader = DataLoader(
        train_dataloader, batch_size=batch_size, pin_memory=pin_memory, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataloader, batch_size=test_batch_size, pin_memory=pin_memory, shuffle=True
    )

    t_model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1").to(device)
    s_model = torch.load("models/student_model.pt", map_location=device).to(device)

    t_model = torch.nn.DataParallel(t_model)
    s_model = torch.nn.DataParallel(s_model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        s_model.parameters(), lr=params["lr"], momentum=params["momentum"]
    )

    with open("kd.log", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "epoch",
                "test_accuracy",
                "test_loss",
            ]
        )

    epochs = 10
    for t in range(epochs):
        if os.path.exists("models/student_model_kd.pt"):
            s_model = torch.load("models/student_model_kd.pt", map_location=device).to(
                device
            )
        print(f"Epoch {t+1}\n-------------------------------")
        kd_train(train_dataloader, s_model, t_model, optimizer)
        test_accuracy, test_loss = test(test_dataloader, s_model, loss_fn)
        print(
            "Accuracy: {:0.2e}%, Test Loss: {:0.2e}".format(
                test_accuracy * 100, test_loss
            )
        )
        torch.save(s_model, "models/student_model_kd.pt")
        with open("kd.log", "a") as fh:
            writer = csv.writer(fh)
            writer.writerow([t, test_accuracy, test_loss])


def prune():
    s_model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1").to(device)

    config_list = [
        {
            "sparsity_per_layer": 0.1,
            "op_types": ["Conv2d"],
        },
    ]

    pruner = L1NormPruner(s_model, config_list)
    s_model, masks = pruner.compress()
    pruner._unwrap_model()
    random_input = torch.randn(batch_size, 3, 224, 224).to(device)

    ModelSpeedup(s_model, random_input, masks, device).speedup_model()

    torch.save(s_model, "models/student_model.pt")


def tvm():
    pass


if __name__ == "__main__":
    prune()
    knowledge_dist()
