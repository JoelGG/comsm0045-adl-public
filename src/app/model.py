import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.inception import InceptionA


class ShallowNet(nn.Module):
    def __init__(self, class_count):
        """Implementation of Shallow CNN architecture from Schindler et al.

        Args:
            class_count (int): Number of classes needing to be classified
        """
        super().__init__()
        self.class_count = class_count
        self.left = ConvBlock(1, 16, (10, 23), (1, 20))
        self.right = ConvBlock(1, 16, (21, 20), (20, 1))
        self.fc1 = nn.Linear(in_features=10240, out_features=200)
        self.lrlu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(in_features=200, out_features=class_count)

    def forward(self, u):
        x_left = self.left(u)
        x_left = torch.flatten(x_left, start_dim=1)

        x_right = self.right(u)
        x_right = torch.flatten(x_right, start_dim=1)

        x = torch.cat((x_left, x_right), dim=1)
        x = self.fc1(x)
        x = self.lrlu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepNet(nn.Module):
    def __init__(self, class_count):
        """Implementation of Deep CNN architecture from Schindler et al.

        Args:
            class_count (int): Number of classes needing to be classified
        """
        super().__init__()
        self.class_count = class_count
        self.left1 = ConvBlock(1, 16, (10, 23), (2, 2))
        self.left2 = ConvBlock(16, 32, (5, 11), (2, 2))
        self.left3 = ConvBlock(32, 64, (3, 5), (2, 2))
        self.left4 = ConvBlock(64, 128, (2, 4), (1, 5))
        self.right1 = ConvBlock(1, 16, (21, 10), (2, 2))
        self.right2 = ConvBlock(16, 32, (10, 5), (2, 2))
        self.right3 = ConvBlock(32, 64, (5, 3), (2, 2))
        self.right4 = ConvBlock(64, 128, (4, 2), (5, 1))

        self.fc1 = nn.Linear(in_features=5120, out_features=200)
        self.lrlu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=200, out_features=class_count)

    def forward(self, u):
        x_left = self.left1(u)
        x_left = self.left2(x_left)
        x_left = self.left3(x_left)
        x_left = self.left4(x_left)
        x_left = torch.flatten(x_left, start_dim=1)

        x_right = self.right1(u)
        x_right = self.right2(x_right)
        x_right = self.right3(x_right)
        x_right = self.right4(x_right)
        x_right = torch.flatten(x_right, start_dim=1)

        x = torch.cat((x_left, x_right), dim=1)
        x = self.fc1(x)
        x = self.lrlu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel):
        """Module for use in implementations of models from Schindler et al. with LeakyReLU negative_slope of 0.3. This module implements
        the Convolution -> Leaky ReLU -> Max Pool process repeated in both reference architectures.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels  (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            kernel_size (int or tuple): The size of the window to take a max over for the max pooling kernel
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(pool_kernel),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class BBNN(nn.Module):
    def __init__(self, class_count):
        """Implementation of Bottom-up Broadcast Neural Network from Liu et al.

        Args:
            class_count (int): Number of classes needing to be classified
        """
        super().__init__()
        self.decision1 = BasicConv2d(1, 32, kernel_size=3)  # 80 x 80 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1))  # 20 x 80 x 32
        self.broadcast_module = BroadcastModule(in_channels=32)  # 20 x 80 x 416
        self.transition = TransitionLayers(in_channels=416)  # 20 x 80 x 32
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=32, out_features=class_count)

    def forward(self, x):
        x = self.decision1(x)
        x = self.pool1(x)
        x = self.broadcast_module(x)
        x = self.transition(x)
        x = self.bn1(x)
        x = self.global_average_pool(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class BroadcastModule(nn.Module):
    def __init__(self, in_channels):
        """Implementation of Broadcast Module from Liu et al.

        Args:
            in_channels (int): Number of input channels
        """
        super().__init__()
        self.inception_a = Inception(in_channels=in_channels, out_channels=in_channels)
        self.inception_b = Inception(
            in_channels=in_channels * 5, out_channels=in_channels
        )
        self.inception_c = Inception(
            in_channels=in_channels * 9, out_channels=in_channels
        )

    def forward(self, x):
        out = self.inception_a(x)
        x = torch.cat((x, out), dim=1)
        out = self.inception_b(x)
        x = torch.cat((x, out), dim=1)
        out = self.inception_c(x)
        x = torch.cat((x, out), dim=1)

        return x


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Implementation of Inception module used by Liu et al.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(out_channels, out_channels, kernel_size=3)

        self.branch5x5_1 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(out_channels, out_channels, kernel_size=5)

        self.branch_pool = BasicConv2d(in_channels, out_channels, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class TransitionLayers(nn.Module):
    def __init__(self, in_channels):
        """Implementation of Transition layers used by Liu et al. to reduce the number of feature channels"""
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels // 13, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """Module for use in implementations of models from Liu et al. This module implements
        the Batch Normalisation -> Convolution -> ReLU process frequently used in the BBNN.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels  (int): Number of channels produced by the convolution
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, padding="same", **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return F.relu(x, inplace=True)
