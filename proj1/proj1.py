import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from tqdm import tqdm
from torchviz import make_dot, make_dot_from_trace
import sys

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device.type.upper())

batch_size = 64
test_batch_size = 1000
epochs = 14
lr = 1.0
gamma = 0.7
seed = 1
save_model = False
arc_env = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()

    t_iter = tqdm(train_loader, position=1, desc=str(epoch), leave=False, colour='yellow')

    for batch_idx, (data, target) in enumerate(t_iter):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        t_iter.set_description("Loss: %.4f" % loss.item(), refresh=False)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_dataset_length = len(test_loader.dataset)
    test_loss /= test_dataset_length
    accuracy = 100. * correct / test_dataset_length
    print('Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, test_dataset_length, accuracy))


def train_models():
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # If we're using NVIDIA, we can apply some more software/hardware optimizations if available
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True


    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if arc_env:
        dir = "/mnt/beegfs/$USER/data/"
    else:
        dir = "data"

    dataset1 = datasets.MNIST(dir, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    criterion = F.nll_loss


    # Unfortunatley, doesn't work on Mac
    if device.type != "mps":
        batch = next(iter(train_loader))
        yhat = model(batch[0])
        make_dot(yhat, params=dict(list(model.named_parameters()))).render(
            "mnist_cnn", format="png"
        )

    for epoch in tqdm(range(1, epochs + 1), position=0, desc="Epochs      ", leave=False, colour='green'):
        train(model, device, train_loader, optimizer, criterion, epoch)
        scheduler.step()


    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    test(model, device, test_loader)


if __name__ == "__main__":
    # try sys.args[2]:
    #     arc_env = True
    if sys.argv[1] == "train":
        train_models()
    elif sys.argv[1] == "prune":
        import time
        for i in tqdm(range(5), position=0, desc="i", leave=False, colour='green'):
            for j in tqdm(range(10), position=1, desc="j", leave=False, colour='red'):
                time.sleep(0.1)
    #     assignment()
    # elif sys.argv[1] == "preprocess":
    #     preprocess()
    # elif sys.argv[1] == "tune":
    #     tune()
    # elif sys.argv[1] == "kaggle":
    #     kaggle()
    else:
        print("Invalid argument")
        print("Example usage: python3 proj1.py train")
