# model.py（新增/修改，确保EEGNet支持动态分类数）
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, num_classes, channels, time_points, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        # 完全复制原EEGNet结构，仅改分类头输出数
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # 计算展平维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_points)
            dummy = self.first_conv(dummy)
            dummy = self.depthwise_conv(dummy)
            dummy = self.separable_conv(dummy)
            self.flatten_size = dummy.view(1, -1).size(1)

        # 核心修改：分类头输出数=num_classes
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x