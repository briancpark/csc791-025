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

from model import SuperResolutionTwitter

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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


def checkpoint(epoch):
    model_out_path = "models/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    pass


def inference():
    input_image = "data/BSDS300/images/test/16077.jpg"
    model = "models/model_epoch_48.pth"
    output_filename = "out.png"

    img = Image.open(input_image).convert("YCbCr")
    y, cb, cr = img.split()

    model = torch.load(model)
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
    upscale_factor = 3
    threads = os.cpu_count()
    batch_size = 64
    test_batch_size = 32
    epochs = 100
    lr = 0.01

    train_set = get_training_set(upscale_factor)
    test_set = get_test_set(upscale_factor)

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=threads,
        batch_size=batch_size,
        shuffle=True,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=threads,
        batch_size=test_batch_size,
        shuffle=False,
    )

    model = SuperResolutionTwitter(upscale_factor=upscale_factor).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(1, epochs + 1):
        train(training_data_loader, model, criterion, optimizer, epoch)
        test(training_data_loader, model, criterion)
        test(testing_data_loader, model, criterion)
        checkpoint(epoch)
        scheduler.step()


if __name__ == "__main__":
    # training()
    inference()
