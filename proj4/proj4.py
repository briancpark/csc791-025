import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.models import densenet121, densenet201, DenseNet201_Weights
import os
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
    "features": 512,
    "lr": 0.001,
    "momentum": 0,
    "batch_size": 64,
}


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def kd_train(train_loader, model_s, model_t, optimizer):
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_s = model_s(data)
        y_t = model_t(data)
        loss_cri = F.cross_entropy(y_s, target)

    # kd loss
    p_s = F.log_softmax(y_s / kd_T, dim=1)
    p_t = F.softmax(y_t / kd_T, dim=1)
    loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

    # total loss
    loss = loss_cir + loss_kd
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
    return correct


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

    batch_size = params["batch_size"]

    train_dataloader = DataLoader(training_data, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    t_model = densenet201(weights="DenseNet201_Weights.IMAGENET1K_V1").to(device)
    s_model = torch.load("models/student_model.pt", map_location=device).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        s_model.parameters(), lr=params["lr"], momentum=params["momentum"]
    )

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train(train_dataloader, t_model, loss_fn, optimizer)
        kd_train(train_dataloader, s_model, t_model, optimizer)
        # accuracy = test(test_dataloader, t_model, loss_fn)
        accuracy = test(test_dataloader, s_model, loss_fn)
        print("Accuracy:", accuracy)


def prune():
    s_model = densenet121().to(device)

    config_list = [
        {
            "sparsity_per_layer": 0.1,
            "op_types": ["Conv2d", "Linear"],
        },
    ]
    retrain_epochs = 20

    pruner = L1NormPruner(s_model, config_list)
    s_model, masks = pruner.compress()
    pruner._unwrap_model()
    random_input = torch.randn(64, 3, 256, 256).to(device)

    ModelSpeedup(s_model, random_input, masks, device).speedup_model()

    torch.save(s_model, "models/student_model.pt")


if __name__ == "__main__":
    # prune()
    knowledge_dist()
