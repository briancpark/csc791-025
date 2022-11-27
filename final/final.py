from __future__ import print_function

import argparse
import csv
import os
import tarfile
import time
from math import log10
from os import listdir
from os import makedirs, remove
from os.path import exists, basename
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from nni.compression.pytorch.pruning import *
from nni.compression.pytorch.speedup import ModelSpeedup
from six.moves import urllib
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize
from torchvision.transforms import ToTensor
from torchviz import make_dot
from tqdm import tqdm

from model import (
    FMEN,
    RDN,
    SuperResolutionByteDance,
    SuperResolutionTwitter,
    VDSR,
    WDSR,
    IMDN,
    RFDN,
)

# ycbcr optimization on or off
model_config = {
    "FMEN": False,
    "VDSR": True,
    "RDN": True,
    "SuperResolutionByteDance": True,
    "SuperResolutionTwitter": True,
    "WDSR": True,
    "IMDN": True,
    "RFDN": True,
}

model_prune_config = {
    "FMEN": [],
    "VDSR": [],
    "RDN": [],  # OOM
    "SuperResolutionByteDance": ["fea_conv", "upsampler.0", "LR_conv"],  # Not working
    "SuperResolutionTwitter": ["conv4"],
    "WDSR": ["body.17", "skip.0", "body.0"],  # Other error
    "IMDN": ["RM.0"],  # TypeError
    "RFDN": ["c.0", "LR_conv", "upsampler.0", "fea_conv"],  # mask conflict
}

### Inference Variables
USE_EXTERNAL_STORAGE = True if os.environ.get("PROJECT") else False


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device.type.upper())

if not os.path.exists("logs"):
    os.mkdir("logs")


### COLOR CONVERSIONS
"""
Y  = R *  0.29900 + G *  0.58700 + B *  0.11400
Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128
Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128

R  = Y +                       + (Cr - 128) *  1.40200
G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
B  = Y + (Cb - 128) *  1.77200

SOURCE: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/ConvertYCbCr.c
"""


def rgb_to_ycbcr(image):
    with torch.no_grad():
        if image.max() < 1.0:
            image = image * 255.0

        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        delta = 128.0
        y = (r * 0.29900) + (g * 0.58700) + (b * 0.11400)
        cb = (r * -0.16874) + (g * -0.33126) + (b * 0.50000) + delta
        cr = (r * 0.50000) + (g * -0.41869) + (b * -0.08131) + delta
        out = torch.stack([y, cb, cr], -3)

        out = out / 255.0
        return out


def ycbcr_to_rgb(image):
    with torch.no_grad():
        if image.max() < 1:
            image = image * 255.0

        y = image[..., 0, :, :]
        cb = image[..., 1, :, :]
        cr = image[..., 2, :, :]

        delta = 128.0
        r = y + (cr - delta) * 1.40200
        g = y + ((cb - delta) * -0.34414) + ((cr - delta) * -0.71414)
        b = y + (cb - delta) * 1.77200
        out = torch.stack([r, g, b], -3)
        return out


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, ycbcr=True):
    if ycbcr:
        img = read_image(filepath)
        y = rgb_to_ycbcr(img)[0:1]
        return y
    else:
        # TODO: convert to PyTorch
        img = Image.open(filepath).convert("RGB")
        return img


class DatasetFromFolder(data.Dataset):
    def __init__(
        self, image_dir, input_transform=None, target_transform=None, ycbcr=True
    ):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
        ]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.ycbcr = ycbcr

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index], ycbcr=self.ycbcr)

        # target = input.copy()
        target = input.clone()

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


def download_bsd300(dest="data"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, "wb") as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose(
        [
            CenterCrop(crop_size),
            Resize(crop_size // upscale_factor),
        ]
    )


def target_transform(crop_size):
    return Compose(
        [
            CenterCrop(crop_size),
        ]
    )


def get_training_set(upscale_factor, ycbcr=True):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(
        train_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
        ycbcr=ycbcr,
    )


def get_test_set(upscale_factor, ycbcr=True):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
        ycbcr=ycbcr,
    )


def train(data_loader, model, criterion, optimizer, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(
        "===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
            epoch, epoch_loss / len(data_loader)
        )
    )

    return epoch_loss / len(data_loader)


def test(data_loader, model, criterion):
    avg_psnr = 0
    with torch.no_grad():
        for batch in data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(data_loader)))

    return avg_psnr / len(data_loader)


def checkpoint(epoch, model, upscale_factor, prefix="original", sparsity=0):
    if sparsity:
        if USE_EXTERNAL_STORAGE:
            PROJECT_DIR = os.environ.get("PROJECT")
            os.makedirs(
                "{}/models/{}/{}/{}/{}".format(
                    PROJECT_DIR,
                    model.__class__.__name__,
                    prefix,
                    upscale_factor,
                    sparsity,
                ),
                exist_ok=True,
            )
            model_out_path = "{}/models/{}/{}/{}/{}/model_epoch_{}.pth".format(
                PROJECT_DIR,
                model.__class__.__name__,
                prefix,
                upscale_factor,
                sparsity,
                epoch,
            )
        else:
            os.makedirs(
                "models/{}/{}/{}/{}".format(
                    model.__class__.__name__, prefix, upscale_factor, sparsity
                ),
                exist_ok=True,
            )
            model_out_path = "models/{}/{}/{}/{}/model_epoch_{}.pth".format(
                model.__class__.__name__, prefix, upscale_factor, sparsity, epoch
            )
    else:
        if USE_EXTERNAL_STORAGE:
            PROJECT_DIR = os.environ.get("PROJECT")
            os.makedirs(
                "{}/models/{}/{}/{}".format(
                    PROJECT_DIR, model.__class__.__name__, prefix, upscale_factor
                ),
                exist_ok=True,
            )
            model_out_path = "{}/models/{}/{}/{}/model_epoch_{}.pth".format(
                PROJECT_DIR, model.__class__.__name__, prefix, upscale_factor, epoch
            )
        else:
            os.makedirs(
                "models/{}/{}/{}".format(
                    model.__class__.__name__, prefix, upscale_factor
                ),
                exist_ok=True,
            )
            model_out_path = "models/{}/{}/{}/model_epoch_{}.pth".format(
                model.__class__.__name__, prefix, upscale_factor, epoch
            )
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# Function helper for inference
def super_resolution(model, img, upscale_factor):
    ycbcr = model_config[model.__class__.__name__]

    if ycbcr:
        ycbcr_img = rgb_to_ycbcr(img)

        # Deconstruct ycbcr_img
        input = ycbcr_img[0].unsqueeze(0)

        # Retain Cb and Cr channels for reconstruction later
        cb = ycbcr_img[1].unsqueeze(0)
        cr = ycbcr_img[2].unsqueeze(0)

        upscale = Resize(
            (img.shape[-2] * upscale_factor, img.shape[-1] * upscale_factor),
            interpolation=InterpolationMode.BICUBIC,
        )
        out_img_cb = upscale(cb)
        out_img_cr = upscale(cr)

        # Upscale Y channel
        input = input.unsqueeze(0)
        # input MUST be formatted as size (1, 1, H, W)
        out = model(input)

        out_img_y = out * 255.0
        out_img_y = out_img_y.clip(0, 255)

        out_img_cb = out_img_cb * 255.0
        out_img_cb = out_img_cb.clip(0, 255).unsqueeze(0)

        out_img_cr = out_img_cr * 255.0
        out_img_cr = out_img_cr.clip(0, 255).unsqueeze(0)

        output = torch.stack([out_img_y, out_img_cb, out_img_cr], -3)
        output = ycbcr_to_rgb(output)

        output = output.clip(0, 255).squeeze(0)

        output = output.type(torch.uint8)
        output = output / 255.0
        return output
    else:
        # TODO: Implement for RGB
        pass


def inference(model_path, upscale_factor, sparsity, pruner="original"):
    # This performs inference on the test dataset

    model = torch.load(model_path, map_location=device)
    ycbcr = model_config[model.__class__.__name__]

    test_dir = "data/BSDS300/images/test"

    # Create directories and set filename
    if pruner == "original":
        if USE_EXTERNAL_STORAGE:
            PROJECT_DIR = os.environ.get("PROJECT")
            os.makedirs(
                "{}/results/{}/{}/{}".format(
                    PROJECT_DIR, model.__class__.__name__, pruner, upscale_factor
                ),
                exist_ok=True,
            )
            output_dir = "{}/results/{}/{}/{}".format(
                PROJECT_DIR, model.__class__.__name__, pruner, upscale_factor
            )
        else:
            os.makedirs(
                "results/{}/{}/{}".format(
                    model.__class__.__name__, pruner, upscale_factor
                ),
                exist_ok=True,
            )
            output_dir = "results/{}/{}/{}".format(
                model.__class__.__name__, pruner, upscale_factor
            )
    else:
        if USE_EXTERNAL_STORAGE:
            PROJECT_DIR = os.environ.get("PROJECT")
            os.makedirs(
                "{}/results/{}/{}/{}/{}".format(
                    PROJECT_DIR,
                    model.__class__.__name__,
                    pruner,
                    upscale_factor,
                    sparsity,
                ),
                exist_ok=True,
            )
            output_dir = "{}/results/{}/{}/{}/{}".format(
                PROJECT_DIR, model.__class__.__name__, pruner, upscale_factor, sparsity
            )
        else:
            os.makedirs(
                "results/{}/{}/{}/{}".format(
                    model.__class__.__name__, pruner, upscale_factor, sparsity
                ),
                exist_ok=True,
            )
            output_dir = "results/{}/{}/{}/{}".format(
                model.__class__.__name__, pruner, upscale_factor, sparsity
            )

    if sparsity:
        output_dir = output_dir + "/sparsity_{}".format(sparsity)
        os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(test_dir)):
        filename = os.path.join(test_dir, img_name)
        output_filename = os.path.join(output_dir, img_name)

        if ycbcr:
            img = read_image(filename).to(device)
            out = super_resolution(model, img, upscale_factor)
            save_image(out, output_filename)
        else:
            # TODO: clean up this mass
            img = Image.open(input_image)
            img_to_tensor = ToTensor()
            input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0])

            input = input.to(device)

            out = model(input)

            # post process
            out = out.cpu()
            out_img = out[0].detach().numpy()
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            out_img = np.transpose(out_img, (1, 2, 0))
            out_img = np.uint8(out_img)
            out_img = Image.fromarray(out_img, mode="RGB")
            out_img.save(output_filename)
            print("output image saved to", output_filename)


def training(
    upscale_factor,
    batch_size,
    test_batch_size,
    epochs,
    logging,
    model_name,
):
    if model_name == "FMEN":  # Gradient error
        model = FMEN(upscale_factor=upscale_factor).to(device)  # DOESN'T WORK
    elif model_name == "VDSR":
        model = VDSR().to(device)  # DOESN'T WORK

    elif model_name == "RFDN":
        model = RFDN(upscale_factor=upscale_factor).to(device)
    elif model_name == "IMDN":
        model = IMDN(upscale_factor=upscale_factor).to(device)
    elif model_name == "RDN":
        model = RDN(upscale_factor=upscale_factor).to(device)
    elif model_name == "SuperResolutionByteDance":
        model = SuperResolutionByteDance(upscale_factor=upscale_factor).to(device)
    elif model_name == "SuperResolutionTwitter":
        model = SuperResolutionTwitter(upscale_factor=upscale_factor).to(device)
    elif model_name == "WDSR":  # XGen baseline model
        model = WDSR(upscale_factor=upscale_factor).to(device)
    else:
        print("Invalid model name")
        return

    ycbcr = model_config[model_name]

    train_set = get_training_set(upscale_factor, ycbcr)
    test_set = get_test_set(upscale_factor, ycbcr)

    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
    )

    criterion = model.criterion
    optimizer = model.optimizer
    scheduler = model.scheduler

    # Initialize logging
    if logging:
        with open(
            f"logs/original_{upscale_factor}_{model.__class__.__name__ }.csv",
            "w",
            newline="",
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_psnr", "test_psnr", "train_loss"])

    for epoch in range(1, epochs + 1):
        train_loss = train(training_data_loader, model, criterion, optimizer, epoch)
        train_psnr = test(training_data_loader, model, criterion)
        test_psnr = test(testing_data_loader, model, criterion)
        checkpoint(epoch, model, upscale_factor)
        scheduler.step()

        if logging:
            with open(
                f"logs/original_{upscale_factor}_{model.__class__.__name__ }.csv", "a"
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_psnr, test_psnr, train_loss])


def visualize(upscale_factor):
    models = [
        FMEN,
        RDN,
        SuperResolutionByteDance,
        SuperResolutionTwitter,
        VDSR,
        WDSR,
        IMDN,
        RFDN,
    ]

    for model in models:
        model = model(upscale_factor=upscale_factor)
        ycbcr = model_config[model.__class__.__name__]
        channels = 1 if ycbcr else 3
        input = torch.randn(1, channels, 300, 300)
        output = model(input)

        make_dot(output, params=dict(list(model.named_parameters()))).render(
            f"figures/{model.__class__.__name__}", format="png"
        )


def quantization(upscale_factor):
    from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)

    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
    )

    model = SuperResolutionTwitter(upscale_factor=upscale_factor).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    config_list = [
        {
            "quant_types": ["input", "weight"],
            "quant_bits": {"input": 8, "weight": 8},
            "op_types": ["Conv2d"],
        },
        {"quant_types": ["output"], "quant_bits": {"output": 8}, "op_types": ["ReLU"]},
    ]

    dummy_input = torch.rand(32, 1, 28, 28).to(device)
    quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input)
    quantizer.compress()

    # Initialize logging
    if logging:
        with open(f"logs/{model.__class__.__name__}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_psnr", "test_psnr", "train_loss"])

    for epoch in range(1, 3 + 1):
        train_loss = train(training_data_loader, model, criterion, optimizer, epoch)
        train_psnr = test(training_data_loader, model, criterion)
        test_psnr = test(testing_data_loader, model, criterion)
        checkpoint(epoch, model, upscale_factor)
        scheduler.step()

        if logging:
            with open(f"logs/{model.__class__.__name__}.csv", "a") as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_psnr, test_psnr, train_loss])

    model_path = "logs/mnist_model.pth"
    calibration_path = "logs/mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)

    print("calibration_config: ", calibration_config)

    from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

    input_shape = (32, 1, 28, 28)
    engine = ModelSpeedupTensorRT(
        model, input_shape, config=calibration_config, batchsize=32
    )
    engine.compress()
    test_trt(engine)


def prune(
    upscale_factor,
    model_path,
    sparsity,
    batch_size,
    test_batch_size,
    step_size,
    gamma,
    finetune_epochs,
    trials,
    logging,
    pruner,
):
    opt_pruners = {
        "LevelPruner": LevelPruner,
        "L1NormPruner": L1NormPruner,
        "L2NormPruner": L2NormPruner,
        "FPGMPruner": FPGMPruner,
        "ActivationAPoZRankPruner": ActivationAPoZRankPruner,
        "ActivationMeanRankPruner": ActivationMeanRankPruner,
        "TaylorFOWeightPruner": TaylorFOWeightPruner,
        "ADMMPruner": ADMMPruner,
    }

    original_times = []
    pruned_times = []

    train_set = get_training_set(upscale_factor)
    test_set = get_test_set(upscale_factor)

    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()

    model = torch.load(model_path, map_location=device)
    test(testing_data_loader, model, criterion)
    fake_input = torch.randn(1, 1, 300, 300).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(trials):
        torch.cuda.synchronize()
        start.record()
        fake_output = model(fake_input)
        end.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        original_times.append(elapsed_time_ms)

    print(f"Original model inference time: {np.mean(original_times)} ms")

    config_list = [
        {"sparsity_per_layer": sparsity, "op_types": ["Conv2d"]},
        {"exclude": True, "op_names": model_prune_config[model.__class__.__name__]},
    ]

    pruner = opt_pruners[pruner](model, config_list)
    _, masks = pruner.compress()
    for name, mask in masks.items():
        print(
            name,
            " sparsity : ",
            "{:.2}".format(mask["weight"].sum() / mask["weight"].numel()),
        )
    pruner._unwrap_model()

    ModelSpeedup(model, torch.rand(1, 1, 300, 300).to(device), masks).speedup_model()

    for _ in range(trials):
        torch.cuda.synchronize()
        start.record()
        fake_output = model(fake_input)
        end.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        pruned_times.append(elapsed_time_ms)
    print(f"Pruned model inference time: {np.mean(pruned_times)} ms")
    test(testing_data_loader, model, criterion)

    optimizer = optim.SGD(model.parameters(), 1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if logging:
        with open(
            f"logs/{pruner.__class__.__name__}_{upscale_factor}_{model.__class__.__name__ }.csv",
            "w",
            newline="",
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_psnr", "test_psnr", "train_loss"])

    # Fine tune the weights
    for epoch in range(1, 1 + finetune_epochs):
        train_loss = train(training_data_loader, model, criterion, optimizer, epoch)
        train_psnr = test(training_data_loader, model, criterion)
        test_psnr = test(testing_data_loader, model, criterion)
        checkpoint(
            epoch,
            model,
            upscale_factor,
            prefix=pruner.__class__.__name__,
            sparsity=sparsity,
        )
        scheduler.step()

        if logging:
            with open(
                f"logs/{pruner.__class__.__name__}_{upscale_factor}_{model.__class__.__name__ }.csv",
                "a",
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_psnr, test_psnr, train_loss])


def benchmark(upscale_factor, model_path):
    ### Warm Up CUDA runtime
    A = torch.randn(2048, 2048).to(device)
    B = torch.randn(2048, 2048).to(device)
    for _ in range(100):
        A @ B

    model = torch.load(model_path, map_location=device)

    ycbcr = model_config[model.__class__.__name__]
    channels = 1 if ycbcr else 3

    train_set = get_training_set(upscale_factor, ycbcr)
    test_set = get_test_set(upscale_factor, ycbcr)

    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=False,
    )

    testing_data_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
    )

    inference_times = []

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for data_loader in [training_data_loader, testing_data_loader]:
            for data, _ in testing_data_loader:
                data = data.to(device)

                start.record()
                _ = model(data)
                end.record()

                torch.cuda.synchronize()

                inference_times.append(start.elapsed_time(end) / 1000)
    else:
        for data_loader in [training_data_loader, testing_data_loader]:
            for data, _ in testing_data_loader:
                data = data.to(device)

                tik = time.perf_counter()
                _ = model(data)
                tok = time.perf_counter()

                inference_times.append(tok - tik)

    inference_times = np.array(inference_times[5:])

    print("Benchmark Dataset")
    print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
    print(f"Average FPS: {1 / np.mean(inference_times):.4f}")

    # Benchmark Video from 360p to 1440p
    high_resolution_x, high_resolution_y = 2560, 1440
    low_resolution_x, low_resolution_y = (
        high_resolution_x // upscale_factor,
        high_resolution_y // upscale_factor,
    )

    input = torch.randn(1, channels, low_resolution_x, low_resolution_y).to(device)
    inference_times = []

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(1000):

            start.record()
            _ = model(input)
            end.record()

            torch.cuda.synchronize()

            inference_times.append(start.elapsed_time(end) / 1000)
    else:
        for _ in range(1000):

            tik = time.perf_counter()
            _ = model(input)
            tok = time.perf_counter()
            inference_times.append(tok - tik)

    inference_times = np.array(inference_times[5:])

    print(f"Benchmark Video from {low_resolution_y}p to {high_resolution_y}p")
    print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
    print(f"Average FPS: {1 / np.mean(inference_times):.4f}")

    if True:
        # Run TensorRT benchmark
        import torch_tensorrt

        input_ = torch.randn(1, 1, 640, 360).to(device)
        trt_ts_module = torch.jit.load(f"tensorrt_models/{model.__class__.__name__}.ts")

        inference_times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(1000):

            start.record()
            result = trt_ts_module(input_)
            end.record()

            torch.cuda.synchronize()

            inference_times.append(start.elapsed_time(end) / 1000)
        inference_times = np.array(inference_times[5:])
        print(
            f"TensorRT Benchmark Video from {low_resolution_y}p to {high_resolution_y}p"
        )
        print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        print(f"Average FPS: {1 / np.mean(inference_times):.4f}")

    # Run ONNX Runtime benchmark
    if True:
        import onnxruntime

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
        so = onnxruntime.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        ort_session = onnxruntime.InferenceSession(
            f"onnx_models/{model.__class__.__name__}.onnx",
            providers=providers,
            sess_options=so,
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}

        inference_times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(1000):

            start.record()
            ort_outs = ort_session.run(None, ort_inputs)
            end.record()

            torch.cuda.synchronize()

            inference_times.append(start.elapsed_time(end) / 1000)
        inference_times = np.array(inference_times[5:])
        print(f"ORT Benchmark Video from {low_resolution_y}p to {high_resolution_y}p")
        print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        print(f"Average FPS: {1 / np.mean(inference_times):.4f}")

    # Run TVM
    if False:
        from tvm.driver import tvmc

        package = tvmc.TVMCPackage(
            package_path=f"tvm_models/{model.__class__.__name__}.tar"
        )

        inference_times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(1000):
            start.record()
            result = tvmc.run(package, device="cuda")
            end.record()

            torch.cuda.synchronize()

            inference_times.append(start.elapsed_time(end) / 1000)

        inference_times = np.array(inference_times[5:])
        print(f"ORT Benchmark Video from {low_resolution_y}p to {high_resolution_y}p")
        print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        print(f"Average FPS: {1 / np.mean(inference_times):.4f}")


def demo(upscale_factor, model_path, frame_path):
    ycbcr = True  # TODO:(bcp) add ycbcr option
    channels = 1 if ycbcr else 3

    model = torch.load(model_path, map_location=device)

    sr_frame_path = frame_path + "_sr"

    images = []
    sr_images = []

    for frame in tqdm(os.listdir(frame_path)):
        fp = os.path.join(frame_path, frame)
        img = read_image(fp).to(device)
        out = super_resolution(model, img, upscale_factor)
        save_image(out, sr_frame_path + "/" + frame)


def convert_to_onnx(model_path):
    # Pinning to opset 9, as DepthToSpace is broken in XGen for blocksize != 4
    opset_version = 9

    if not os.path.exists("onnx_models"):
        os.mkdir("onnx_models")

    model = torch.load(model_path, map_location=device).cpu()

    ycbcr = model_config[model.__class__.__name__]
    channels = 1 if ycbcr else 3

    input = torch.randn(1, channels, 640, 360)

    torch.onnx.export(
        model,
        input,
        f"onnx_models/{model.__class__.__name__}.onnx",
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,  # XGen supports 11 or 9
        export_params=True,
    )


def convert_to_coreml(model_path):
    # https://coremltools.readme.io/docs/pytorch-conversion
    import coremltools as ct

    if not os.path.exists("coreml_models"):
        os.mkdir("coreml_models")

    torch_model = torch.load(model_path, map_location=device).cpu()
    # Set the model in evaluation mode.
    torch_model.eval()

    ycbcr = model_config[torch_model.__class__.__name__]
    channels = 1 if ycbcr else 3

    # Trace the model with random data.
    input = torch.rand(1, channels, 640, 360)
    traced_model = torch.jit.trace(torch_model, input)
    out = traced_model(input)

    # Using image_input in the inputs parameter:
    # Convert to Core ML program using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input.shape)],
    )

    # Save the converted model.
    model.save(f"coreml_models/{model.__class__.__name__}.mlpackage")


def convert_to_tensorrt(model_path):
    # https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html
    import torch_tensorrt

    if not os.path.exists("tensorrt_models"):
        os.mkdir("tensorrt_models")

    # torch module needs to be in eval (not training) mode
    model = torch.load(model_path, map_location=device).cpu()

    ycbcr = model_config[model.__class__.__name__]
    channels = 1 if ycbcr else 3

    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, channels, 640, 360],
            opt_shape=[1, channels, 640, 360],
            max_shape=[1, channels, 640, 360],
            dtype=torch.float,  # torch.half
        )
    ]
    enabled_precisions = {torch.float}  # torch.half, Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    input = torch.randn(1, channels, 640, 360).to(device)
    result = trt_ts_module(input)
    torch.jit.save(trt_ts_module, f"tensorrt_models/{model.__class__.__name__}.ts")

    # Deployment application
    trt_ts_module = torch.jit.load(f"tensorrt_models/{model.__class__.__name__}.ts")
    input = input.to(device)
    result = trt_ts_module(input)


def convert_to_tvm(model_path):
    # https://tvm.apache.org/docs/tutorial/tvmc_python.html
    from tvm.driver import tvmc

    torch_model = torch.load(model_path, map_location=device).cpu()
    onnx_model_path = f"onnx_models/{torch_model.__class__.__name__}.onnx"

    if not os.path.exists("tvm_models"):
        os.mkdir("tvm_models")

    model = tvmc.load(onnx_model_path)  # Step 1: Load

    tvmc.tune(
        model, target="cuda", enable_autoscheduler=True
    )  # Step 1.5: Optional Tune

    package = tvmc.compile(
        model,
        target="cuda",
        package_path=f"tvm_models/{torch_model.__class__.__name__}.tar",
    )  # Step 2: Compile

    result = tvmc.run(package, device="cuda")  # Step 3: Run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument(
        "--model_path", type=str, default="models/original/model_epoch_99.pth"
    )
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--logging", action="store_true", default=True)
    parser.add_argument("--pruner", type=str, default="original")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="SuperResolutionTwitter")
    parser.add_argument(
        "--frame_path",
        type=str,
        default="/ocean/projects/cis220070p/bpark1/demo/frames/macross",
    )

    args = parser.parse_args()
    if args.mode == "all":
        training(
            args.upscale_factor,
            args.batch_size,
            args.test_batch_size,
            args.epochs,
            args.logging,
            args.model_name,
        )
        inference()
        visualize(args.upscale_factor)
        prune()
        benchmark()
    elif args.mode == "training":
        training(
            args.upscale_factor,
            args.batch_size,
            args.test_batch_size,
            args.epochs,
            args.logging,
            args.model_name,
        )
    elif args.mode == "inference":
        inference(args.model_path, args.upscale_factor, args.sparsity, args.pruner)
    elif args.mode == "visualize":
        visualize(args.upscale_factor)
    elif args.mode == "prune":
        prune(
            args.upscale_factor,
            args.model_path,
            args.sparsity,
            args.batch_size,
            args.test_batch_size,
            args.step_size,
            args.gamma,
            args.finetune_epochs,
            args.trials,
            args.logging,
            args.pruner,
        )
    elif args.mode == "quantization":
        quantization(args)
    elif args.mode == "benchmark":
        benchmark(args.upscale_factor, args.model_path)
    elif args.mode == "demo":
        demo(args.upscale_factor, args.model_path, args.frame_path)
    elif args.mode == "onnx":
        convert_to_onnx(args.model_path)
    elif args.mode == "coreml":
        convert_to_coreml(args.model_path)
    elif args.mode == "tensorrt":
        convert_to_tensorrt(args.model_path)
    elif args.mode == "tvm":
        convert_to_tvm(args.model_path)
    # elif args.mode == "quant":
    #     from onnxmltools.utils.float16_converter import (
    #         convert_float_to_float16_model_path,
    #     )
    #     from onnxmltools.utils import save_model

    #     new_onnx_model = convert_float_to_float16_model_path(
    #         "onnx_models/SuperResolutionTwitter_9.onnx", keep_io_types=False
    #     )
    #     save_model(new_onnx_model, "onnx_models/SRTfp16.onnx")
