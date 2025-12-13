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

        # === Block 3 & 4: Separable Convolution (匹配权重文件结构) ===
        # 对应权重中的 conv3 (Depthwise Temporal Conv)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)

        # 对应权重中的 conv4 和 batchnorm3 (Pointwise Conv)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        # =========================================================

        # 计算展平后的维度: 16 channels * 7 time points = 112
        final_time_points = 7
        self.flatten_size = 16 * final_time_points
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Channels, Time) -> (B, 1, 22, 250)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x) # (B, 16, 1, 63)
        x = self.dropout1(x)

        # Block 3 & 4 逻辑
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x) # (B, 16, 1, 7)
        x = self.dropout2(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x

class EEGNet_feature(nn.Module):
    def __init__(self, in_channels, n_dim):
        super(EEGNet_feature, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(in_channels, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(0.25)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(0.25)
        )
        # 暂时将fc层注释，先动态计算维度；或在forward中动态计算
        self.fc = None # 初始化时不定义，forward中动态创建（或先占位）
        self.n_dim = n_dim # 保存特征维度

    def forward(self, x, simsiam=False):
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        # 动态计算flatten后的维度
        flatten_dim = x.size(1) * x.size(2) * x.size(3)
        x = x.view(x.size(0), -1)
        # 若fc层未创建，动态创建（仅第一次forward时执行）
        if self.fc is None:
            self.fc = nn.Linear(flatten_dim, self.n_dim).to(x.device)
        x = self.fc(x)
        return x

class EEGNet_class(nn.Module):
    def __init__(self, n_dim, num_classes):
        super(EEGNet_class, self).__init__()
        self.fc1 = nn.Linear(n_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x