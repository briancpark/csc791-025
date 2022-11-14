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
import torch.onnx
from nni.compression.pytorch.pruning import *
from nni.compression.pytorch.speedup import ModelSpeedup

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if not os.path.exists("logs"):
    os.mkdir("logs")

print("Using device:", device.type.upper())

# Set Random Seed
torch.manual_seed(42)

params = {
    "lr": 0.001,
    "momentum": 0.7,
}

batch_size = 1024
test_batch_size = 2048
pin_memory = True


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

    for batch, (X, y) in enumerate(t_iter):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def kd_train(train_loader, model_s, model_t, optimizer, kd_T):
    cri_kd = DistillKL(kd_T)

    model_s.train()
    model_t.eval()
    size = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
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

    t_iter = tqdm(dataloader)

    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in t_iter:
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

    dataset1 = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10("data", train=False, transform=transform)

    train_dataloader = DataLoader(
        dataset1, pin_memory=pin_memory, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset2, pin_memory=pin_memory, batch_size=test_batch_size, shuffle=True
    )

    temperatures = [1, 5, 10, 15, 20]

    for kd_T in temperatures:
        t_model = torch.load("models/cifar10_resnet101.pt", map_location=device)
        s_model = torch.load("models/student_model.pt", map_location=device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            s_model.parameters(), lr=params["lr"], momentum=params["momentum"]
        )

        with open("logs/kd_{kd_T}.log", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "epoch",
                    "student_train_accuracy",
                    "student_train_loss",
                    "student_test_accuracy",
                    "student_test_loss",
                ]
            )

        epochs = 200

        t_train_accuracy, t_train_loss = test(train_dataloader, t_model, loss_fn)
        t_test_accuracy, t_test_loss = test(test_dataloader, t_model, loss_fn)
        print(
            "Teacher model train accuracy: ",
            t_train_accuracy,
            "Teacher model train loss: ",
            t_train_loss,
        )
        print(
            "Teacher model test accuracy: ",
            t_test_accuracy,
            "Teacher model test loss: ",
            t_test_loss,
        )
        for t in range(epochs):
            if os.path.exists(f"models/student_model_kd_{kd_T}.pt"):
                s_model = torch.load(
                    f"models/student_model_kd_{kd_T}.pt", map_location=device
                ).to(device)
            print(f"Epoch {t+1}\n-------------------------------")
            kd_train(train_dataloader, s_model, t_model, optimizer, kd_T)
            s_train_accuracy, s_train_loss = test(train_dataloader, t_model, loss_fn)
            s_test_accuracy, s_test_loss = test(test_dataloader, s_model, loss_fn)

            print(
                f"Student Train Accuracy: {s_train_accuracy * 100}%, Student Train Loss: {s_train_loss}"
            )
            print(
                f"Student Test Accuracy: {s_test_accuracy * 100}%, Student Test Loss: {s_test_loss}"
            )
            torch.save(s_model, f"models/student_model_kd_{kd_T}.pt")
            with open(f"logs/kd_{kd_T}.log", "a") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [t, s_train_accuracy, s_train_loss, s_test_accuracy, s_test_loss]
                )


def prune():
    s_model = resnet18().to(device)

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


def convert_torch_to_onnx():
    x = torch.randn(1, 3, 224, 224)
    resnet18_model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    resnet18_kd_model = torch.load("models/student_model_kd.pt").module.to("cpu")

    torch.onnx.export(
        resnet18_model,
        x,
        "resnet18.onnx",
        export_params=True,
        input_names=["data"],
        output_names=["output"],
    )

    torch.onnx.export(
        resnet18_kd_model,
        x,
        "resnet18_kd.onnx",
        export_params=True,
        input_names=["data"],
        output_names=["output"],
    )


def pre_process():
    from tvm.contrib.download import download_testdata
    from PIL import Image
    import numpy as np

    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # ONNX expects NCHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

    # Normalize according to ImageNet
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_stddev = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :] / 255 - imagenet_mean[i]
        ) / imagenet_stddev[i]

    # Add batch dimension
    img_data = np.expand_dims(norm_img_data, axis=0)

    # Save to .npz (outputs imagenet_cat.npz)
    np.savez("imagenet_cat", data=img_data)


def post_process():
    import os.path
    import numpy as np

    from scipy.special import softmax

    from tvm.contrib.download import download_testdata

    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    output_file = "predictions.npz"

    # Open the output and read the output tensor
    if os.path.exists(output_file):
        with np.load(output_file) as data:
            scores = softmax(data["output_0"])
            scores = np.squeeze(scores)
            ranks = np.argsort(scores)[::-1]

            for rank in ranks[0:5]:
                print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


if __name__ == "__main__":
    # prune()
    knowledge_dist()
    # convert_torch_to_onnx()
    # pre_process()
    # post_process()
