from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)


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


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(
            "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss.item()
            )
        )

    print(
        "===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
            epoch, epoch_loss / len(training_data_loader)
        )
    )


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "models/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
    parser.add_argument(
        "--upscale_factor", type=int, default=3, help="super resolution upscale factor"
    )
    parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
    parser.add_argument(
        "--testBatchSize", type=int, default=128, help="testing batch size"
    )
    parser.add_argument(
        "--nEpochs", type=int, default=100, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning Rate. Default=0.01"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument(
        "--mps", action="store_true", default=False, help="enables macOS GPU training"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="number of threads for data loader to use",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="random seed to use. Default=123"
    )
    opt = parser.parse_args()

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if not opt.mps and torch.backends.mps.is_available():
        raise Exception("Found mps device, please run with --mps to enable macOS GPU")

    torch.manual_seed(opt.seed)
    use_mps = opt.mps and torch.backends.mps.is_available()

    if opt.cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("===> Loading datasets")
    train_set = get_training_set(opt.upscale_factor)
    test_set = get_test_set(opt.upscale_factor)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.testBatchSize,
        shuffle=False,
    )

    print("===> Building model")
    model = Net(upscale_factor=opt.upscale_factor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )  # TODO: (bcp) Add scheduler.

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
        checkpoint(epoch)
