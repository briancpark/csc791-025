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
from torchvision.models import densenet201
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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
    "momentum": 0,
}

if torch.cuda.is_available():
    num_gpus = int(
        os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read()
    )
else:
    num_gpus = 1  # Placeholder

batch_size = 1024 * num_gpus
test_batch_size = 2048 * num_gpus
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


def train(dataloader, model, loss_fn, optimizer, distributed=False, rank=-1):
    size = len(dataloader.dataset)
    model.train()

    if rank != -1:
        device = rank

    if distributed:
        t_iter = dataloader
    else:
        t_iter = tqdm(dataloader)

    for batch, (X, y) in enumerate(t_iter):
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


def test(dataloader, model, loss_fn, distributed=False, rank=-1):
    size = len(dataloader.dataset)
    if rank != -1:
        device = rank
    if distributed:
        t_iter = dataloader
    else:
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


### DISTRIBUTED TRAINING


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, batch_size=batch_size, pin_memory=True, num_workers=0):
    # https://blog.jovian.ai/image-classification-of-cifar100-dataset-using-pytorch-8b7145242df1
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*stats)]
    )

    train_data = datasets.CIFAR100(
        download=True, root="data", transform=train_transform
    )
    test_data = datasets.CIFAR100(root="data", train=False, transform=test_transform)

    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def distributed_training(rank, world_size):
    setup(rank, world_size)
    train_dataloader, test_dataloader = prepare(rank, world_size)

    model = densenet201().to(rank)
    dpp_model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )

    optimizer = torch.optim.SGD(
        dpp_model.parameters(), lr=params["lr"], momentum=params["momentum"]
    )
    loss_fn = nn.CrossEntropyLoss()

    with open("logs/densenet201.log", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "epoch",
                "train_accuracy",
                "train_loss",
                "test_accuracy",
                "test_loss",
            ]
        )

    epochs = 200
    for t in range(epochs):
        train(
            train_dataloader, dpp_model, loss_fn, optimizer, distributed=True, rank=rank
        )
        test_accuracy, test_loss = test(
            test_dataloader, dpp_model, loss_fn, distributed=True, rank=rank
        )
        train_accuracy, train_loss = test(
            train_dataloader, dpp_model, loss_fn, distributed=True, rank=rank
        )

        # On main rank, print and save intermediate results
        if rank == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Accuracy: {train_accuracy * 100}%, Train Loss: {train_loss}")
            print(f"Accuracy: {test_accuracy * 100}%, Test Loss: {test_loss}")

            with open("logs/densenet201.log", "a") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [t, train_accuracy, train_loss, test_accuracy, test_loss]
                )

            # Save checkpoint
            torch.save(dpp_model.state_dict(), "models/densenet201.pt")

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()

        if test_accuracy > 0.9:
            cleanup()
            break

    cleanup()


def distributed():
    import torch.multiprocessing as mp

    world_size = num_gpus
    mp.spawn(distributed_training, args=(world_size,), nprocs=world_size)


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
    distributed()
    # prune()
    # knowledge_dist()
    # convert_torch_to_onnx()
    # pre_process()
    # post_process()
