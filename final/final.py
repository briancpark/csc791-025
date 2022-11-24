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
from PIL import Image
from nni.compression.pytorch.pruning import *
from nni.compression.pytorch.speedup import ModelSpeedup
from six.moves import urllib
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize
from torchvision.transforms import ToTensor
from torchviz import make_dot

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


def checkpoint(epoch, model, upscale_factor, prefix="original"):
    if USE_EXTERNAL_STORAGE:
        PROJECT_DIR = os.environ.get("PROJECT")
        os.makedirs(
            "{}/models/{}/{}".format(PROJECT_DIR, prefix, upscale_factor),
            exist_ok=True,
        )
        model_out_path = "{}/models/{}/{}/model_epoch_{}.pth".format(
            PROJECT_DIR, prefix, upscale_factor, epoch
        )
    else:
        os.makedirs("models/{}".format(prefix), exist_ok=True)
        model_out_path = "models/{}/{}model_epoch_{}.pth".format(
            prefix, upscale_factor, epoch
        )
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def inference(model_path, upscale_factor, sparsity, pruner="original"):
    # TODO (bcp): Parameterize this
    input_image = "data/BSDS300/images/test/16077.jpg"
    # input_image = "data/BSDS300/images/test/385039.jpg"
    if pruner == "original":
        output_filename = f"figures/out_{upscale_factor}_{pruner}.png"
    else:
        output_filename = f"figures/out_{upscale_factor}_{pruner}_{sparsity}.png"

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


def training(
    upscale_factor, batch_size, test_batch_size, epochs, lr, step_size, gamma, logging
):
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
    model = SuperResolutionTwitter(upscale_factor=upscale_factor)
    input = torch.randn(1, 1, 300, 300)
    output = model(input)

    make_dot(output, params=dict(list(model.named_parameters()))).render(
        f"figures/{model.__class__.__name__}", format="png"
    )

    input = torch.randn(1, 1, 300, 300)
    model = RDN(upscale_factor=upscale_factor)
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
        # {"exclude": True, "op_names": ["conv4"]},
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
        checkpoint(epoch, model, upscale_factor, prefix=pruner.__class__.__name__)
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


def convert_to_onnx(model_path):
    # Pinning to opset 9, as DepthToSpace is broken in XGen for blocksize != 4
    opset_version = 9

    if not os.path.exists("onnx_models"):
        os.mkdir("onnx_models")

    model = torch.load(model_path, map_location=device).cpu()

    x = torch.randn(1, 3, 300, 300)
    torch.onnx.export(
        model,
        x,
        f"onnx_models/{model.__class__.__name__}_{opset_version}.onnx",
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

    # Trace the model with random data.
    example_input = torch.rand(1, 1, 300, 300)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    # Using image_input in the inputs parameter:
    # Convert to Core ML program using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=example_input.shape)],
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

    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1, 300, 300],
            opt_shape=[1, 1, 300, 300],
            max_shape=[1, 1, 300, 300],
            dtype=torch.float,  # torch.half
        )
    ]
    enabled_precisions = {torch.float}  # torch.half, Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    input_data = torch.randn(1, 1, 300, 300).to(device)
    result = trt_ts_module(input_data)
    torch.jit.save(trt_ts_module, f"tensorrt_models/{model.__class__.__name__}.ts")

    # Deployment application
    trt_ts_module = torch.jit.load(f"tensorrt_models/{model.__class__.__name__}.ts")
    input_data = input_data.to(device)
    result = trt_ts_module(input_data)


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

    args = parser.parse_args()
    if args.mode == "all":
        training(
            args.upscale_factor,
            args.batch_size,
            args.test_batch_size,
            args.epochs,
            args.lr,
            args.step_size,
            args.gamma,
            args.logging,
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
            args.lr,
            args.step_size,
            args.gamma,
            args.logging,
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
    elif args.mode == "onnx":
        convert_to_onnx(args.model_path)
    elif args.mode == "coreml":
        convert_to_coreml(args.model_path)
    elif args.mode == "tensorrt":
        convert_to_tensorrt(args.model_path)
    elif args.mode == "quant":
        from onnxmltools.utils.float16_converter import (
            convert_float_to_float16_model_path,
        )
        from onnxmltools.utils import save_model

        new_onnx_model = convert_float_to_float16_model_path(
            "onnx_models/SuperResolutionTwitter_9.onnx", keep_io_types=False
        )
        save_model(new_onnx_model, "onnx_models/SRTfp16.onnx")
        # /ocean/projects/cis220070p/bpark1/models/original/4/model_epoch_1000.pt
    elif args.mode == "tester":
        from model import SuperResolutionTwitter

        # from model import RDN, WDSR, VDSR

        # BNECHMARK TWITTER
        input = torch.randn(1, 1, 300, 300).to(device)
        model = SuperResolutionTwitter(upscale_factor=4).to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        original_times = []
        for _ in range(1000):
            torch.cuda.synchronize()
            start.record()
            output = model(input)
            end.record()

            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            original_times.append(elapsed_time_ms)
        print(f"Original: {np.mean(original_times[5:])}")

        # # BNECHMARK RDN
        # model = RDN(upscale_factor=4).to(device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # original_times = []
        # for _ in range(1000):
        #     torch.cuda.synchronize()
        #     start.record()
        #     output = model(input)
        #     end.record()

        #     torch.cuda.synchronize()
        #     elapsed_time_ms = start.elapsed_time(end)
        #     original_times.append(elapsed_time_ms)
        # print(f"Original: {np.mean(original_times[5:])}")

        # # BENCHMARK WDSR
        # input = torch.randn(1, 3, 300, 300).to(device)
        # model = WDSR(upscale_factor=4).to(device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # original_times = []
        # for _ in range(1000):
        #     torch.cuda.synchronize()
        #     start.record()
        #     output = model(input)
        #     end.record()

        #     torch.cuda.synchronize()
        #     elapsed_time_ms = start.elapsed_time(end)
        #     original_times.append(elapsed_time_ms)
        # print(f"Original: {np.mean(original_times[5:])}")

        # # BENCHMARK VDSR
        # input = torch.randn(1, 1, 300, 300).to(device)
        # model = VDSR().to(device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # original_times = []
        # for _ in range(1000):
        #     torch.cuda.synchronize()
        #     start.record()
        #     output = model(input)
        #     end.record()

        #     torch.cuda.synchronize()
        #     elapsed_time_ms = start.elapsed_time(end)
        #     original_times.append(elapsed_time_ms)
        # print(f"Original: {np.mean(original_times[5:])}")

        # input = torch.randn(1, 3, 300, 300).to(device)
        # # from model import RLFN_cut
        # # model = RLFN_cut(in_nc=3, out_nc=3).to(device)
        # from model import FMEN
        # model = FMEN().to(device)
        # model.eval()

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # original_times = []
        # for _ in range(1000):
        #     torch.cuda.synchronize()
        #     start.record()
        #     output = model(input)
        #     end.record()

        #     torch.cuda.synchronize()
        #     elapsed_time_ms = start.elapsed_time(end)
        #     original_times.append(elapsed_time_ms)
        # print(f"Original: {np.mean(original_times[5:])}")
