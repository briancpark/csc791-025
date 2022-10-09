import torch
import torch.nn.functional as F
from torch.optim import SGD

from scripts.compression_mnist_model import (
    TorchModel,
    trainer,
    evaluator,
    device,
    test_trt,
)

# define the model
model = TorchModel().to(device)

# define the optimizer and criterion for pre-training

optimizer = SGD(model.parameters(), 1e-2)
criterion = F.nll_loss

# pre-train and evaluate the model on MNIST dataset
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

config_list = [
    {
        "quant_types": ["input", "weight"],
        "quant_bits": {"input": 8, "weight": 8},
        "op_types": ["Conv2d"],
    },
    {"quant_types": ["output"], "quant_bits": {"output": 8}, "op_types": ["ReLU"]},
    {
        "quant_types": ["input", "weight"],
        "quant_bits": {"input": 8, "weight": 8},
        "op_names": ["fc1", "fc2"],
    },
]

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

dummy_input = torch.rand(32, 1, 28, 28).to(device)
quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input)
quantizer.compress()

for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

model_path = "./log/mnist_model.pth"
calibration_path = "./log/mnist_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)

print("calibration_config: ", calibration_config)

from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

input_shape = (32, 1, 28, 28)
engine = ModelSpeedupTensorRT(
    model, input_shape, config=calibration_config, batchsize=32
)
engine.compress()
test_trt(engine)
