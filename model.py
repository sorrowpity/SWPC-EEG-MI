import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, time_points=1000, dropout_rate=0.5):
        super(EEGNet, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Block 2: DepthwiseConv2D
        self.conv2 = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: SeparableConv2D
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # 计算全连接层输入维度
        # 输出维度计算：Time_points / 4 / 8 = Time_points / 32
        final_time_points = time_points // 32
        self.flatten_size = 16 * final_time_points

        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Channels, Time)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x

    def extract_feature(self, x):
        # 前向传播到dropout2之后、全连接层之前（与forward逻辑完全对齐）
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        # 展平后返回特征（不经过全连接层）
        feature = x.view(-1, self.flatten_size)
        return feature