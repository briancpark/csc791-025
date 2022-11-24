from __future__ import print_function

import torch
import math
from torch import nn as nn
from torch.nn import init as init


class SuperResolutionTwitter(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

    Wenzhe Shi, Jose Caballero, Ferenc Husza, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang

    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(SuperResolutionTwitter, self).__init__()

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


class RDN(nn.Module):
    """
    Residual Dense Network for Image Super-Resolution

    Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, Yun Fu

    https://arxiv.org/abs/1802.08797
    """

    def __init__(self,upscale_factor, channel=1, growth_rate=64, rdb_number=3):
        super(RDN, self).__init__()
        self.SFF1 = nn.Conv2d(
            in_channels=channel, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.SFF2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.RDB1 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB2 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB3 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.GFF1 = nn.Conv2d(
            in_channels=64 * 3, out_channels=64, kernel_size=1, padding=0
        )
        self.GFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.upconv = nn.Conv2d(
            in_channels=64,
            out_channels=(64 * upscale_factor * upscale_factor),
            kernel_size=3,
            padding=1,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=channel, kernel_size=3, padding=1
        )

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 75, 100], gamma=0.5
        )

    def forward(self, x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1, f_2, f_3), 1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_
        f_upconv = self.upconv(f_DF)
        f_upscale = self.pixelshuffle(f_upconv)
        f_conv2 = self.conv2(f_upscale)
        return f_conv2


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate):
        super(RDB, self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers, input_dim, growth_rate)
        self.conv1x1 = nn.Conv2d(
            in_channels=input_dim + nb_layers * growth_rate,
            out_channels=growth_rate,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _make_layer(self, nb_layers, input_dim, growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out + x
