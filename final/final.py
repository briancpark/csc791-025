from __future__ import print_function
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import exists, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data
from os import listdir
from os.path import join
import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np
import sys
import csv
import time
from torchviz import make_dot
from model import SuperResolutionTwitter

### Inference Variables
input_image = "data/BSDS300/images/test/16077.jpg"
model_path = "models/model_epoch_48.pth"
output_filename = "out.png"

### Training Variables
upscale_factor = 3
batch_size = 64
test_batch_size = 32
epochs = 1000
lr = 0.001
logging = True

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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("YCbCr")
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
        ]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
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
            ToTensor(),
        ]
    )


def target_transform(crop_size):
    return Compose(
        [
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(
        train_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )


def train(data_loader, model, criterion, optimizer, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
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


def checkpoint(epoch, model):
    model_out_path = "models/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def inference():
    img = Image.open(input_image).convert("YCbCr")
    y, cb, cr = img.split()

    model = torch.load(model_path)
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    model = model.to(device)
    input = input.to(device)

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode="L")

    out_img_cb = cb.resize(out_img_y.size, Image.Resampling.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.Resampling.BICUBIC)
    out_img = Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert("RGB")

    out_img.save(output_filename)
    print("output image saved to ", output_filename)


def training():
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

    model = SuperResolutionTwitter(upscale_factor=upscale_factor).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Initialize logging
    if logging:
        with open(f"logs/{model.__class__.__name__ }.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_psnr", "test_psnr", "train_loss"])

    for epoch in range(1, epochs + 1):
        train_loss = train(training_data_loader, model, criterion, optimizer, epoch)
        train_psnr = test(training_data_loader, model, criterion)
        test_psnr = test(testing_data_loader, model, criterion)
        checkpoint(epoch, model)
        scheduler.step()

        if logging:
            with open(f"logs/{model.__class__.__name__ }.csv", "a") as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_psnr, test_psnr, train_loss])


def visualize():
    model = SuperResolutionTwitter(upscale_factor=3)
    input = torch.randn(1, 1, 300, 300)
    output = model(input)

    make_dot(output, params=dict(list(model.named_parameters()))).render(
        f"figures/{model.__class__.__name__}", format="png"
    )

def quantization():
    from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

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

    model = SuperResolutionTwitter(upscale_factor=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    config_list = [{
        'quant_types': ['input', 'weight'],
        'quant_bits': {'input': 8, 'weight': 8},
        'op_types': ['Conv2d']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output': 8},
        'op_types': ['ReLU']
    }]


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
        checkpoint(epoch, model)
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
    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
    engine.compress()
    test_trt(engine)
    

def prune():
    pass


def benchmark():
    ### Warm Up CUDA runtime
    A = torch.randn(2048, 2048).to(device)
    B = torch.randn(2048, 2048).to(device)
    for _ in range(100):
        A @ B

    train_set = get_training_set(upscale_factor)
    test_set = get_test_set(upscale_factor)

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

    model = torch.load(model_path, map_location=device)

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

    print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
    print(f"Average FPS: {1 / np.mean(inference_times):.4f}")


def convert_to_onnx():
    if not os.path.exists("onnx_models"):
        os.mkdir("onnx_models")

    model = torch.load(model_path).cpu()
    x = torch.randn(1, 1, 300, 300)
    torch.onnx.export(
        model,
        x,
        f"onnx_models/{model.__class__.__name__}.onnx",
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,  # XGen supports 11 or 9
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        training()
        inference()
        visualize()
        prune()
        benchmark()
    elif sys.argv[1] == "training":
        training()
    elif sys.argv[1] == "inference":
        inference()
    elif sys.argv[1] == "visualize":
        visualize()
    elif sys.argv[1] == "prune":
        prune()
    elif sys.argv[1] == "quantization":
        quantization()
    elif sys.argv[1] == "benchmark":
        benchmark()
    elif sys.argv[1] == "onnx":
        convert_to_onnx()
    else:
        print("Invalid argument")
        print("Example usage: python3 final.py training")
