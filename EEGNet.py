# EEGNet.py (修复版)
import torch
import torch.nn as nn
import numpy as np


class EEGNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, time_points=250, dropout_rate=0.5):
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Block 2: Spatial Convolution (Depthwise)
        # D=2, F2=16. F1=8
        self.conv2 = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # === Block 3 & 4: Separable Convolution (拆分成单独的层以匹配权重文件) ===
        # 对应权重中的 Missing key(s): "conv3.weight" (Depthwise Temporal Conv)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)

        # 对应权重中的 Missing key(s): "conv4.weight", "batchnorm3.weight" (Pointwise Conv)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        # =========================================================================

        # 针对 250 时间点的输入，计算展平后的维度
        # Time dimension after Block 2: ceil(250 / 4) = 63
        # Time dimension after Block 4: ceil(63 / 8) = 7
        final_time_points = 7  # 修正：16 * 7 = 112 (解决 128 vs 112 的问题)
        self.flatten_size = 16 * final_time_points
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Channels, Time) -> (B, 1, 22, 250)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)  # (B, 16, 1, 63)
        x = self.dropout1(x)

        # === 前向传播中新增 Block 3 & 4 逻辑 ===
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)  # (B, 16, 1, 7)
        x = self.dropout2(x)
        # ======================================

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x