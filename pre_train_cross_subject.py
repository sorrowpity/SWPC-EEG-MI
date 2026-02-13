# pre_train_cross_subject.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from data_loader import EEG_loader, EEGDataset
# 新增：从models.EEGNet导入所需类
from SWPC.EEGNet import EEGNet_feature, EEGNet_class  # 关键修复：添加导入语句
from EEG_cross_subject_loader_MI import EEG_loader  # 确认使用该路径的类


def main():
    # 配置参数
    params = {
        'test_subj': 5,
        'dataset': 'BNCI2014001',
        'model_name': 'EEGNet',
        'learning_rate': 1e-3,
        'num_iterations': 20,
        'cuda': True,
        'seed': 42,
        'RESTING': False
    }

    # 初始化
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    device = torch.device('cuda' if params['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    data = EEG_loader(test_subj=params['test_subj'], dataset=params['dataset'])
    train_x, train_y = data.train_x, data.train_y
    test_x, test_y = data.test_x, data.test_y

    # 数据格式转换 (N, C, T) -> (N, 1, T, C) 适配模型输入
    train_x = torch.from_numpy(train_x).unsqueeze(1).float()  # 原train_x是(N, C, T)，直接转换为张量
    train_y = torch.from_numpy(train_y).long()
    test_x = torch.from_numpy(test_x).unsqueeze(1).float()
    test_y = torch.from_numpy(test_y).long()

    # 构建数据集
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 初始化模型
    if params['model_name'] == 'EEGNet':
        if params['dataset'] == 'BNCI2014001':
            # 修正：参数名与EEGNet_feature定义一致（原定义为out_features，非n_dim）
            model_feature = EEGNet_feature(in_channels=22, n_dim=120)  # 替换为实际参数名  # 关键修复：参数名修正
            model_class = EEGNet_class(n_dim=120, num_classes=2)  # 关键修复：参数名修正（原定义为in_features）
        else:
            raise ValueError(f"未支持的数据集: {params['dataset']}")
    else:
        raise ValueError(f"未支持的模型: {params['model_name']}")

    model_feature.to(device)
    model_class.to(device)

    # 优化器与损失函数
    opt_feature = optim.Adam(model_feature.parameters(), lr=params['learning_rate'])
    opt_class = optim.Adam(model_class.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(params['num_iterations']):
        model_feature.train()
        model_class.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_feature.zero_grad()
            opt_class.zero_grad()

            x_middle = model_feature(x)
            outputs = model_class(x_middle)
            loss = criterion(outputs, y)

            loss.backward()
            opt_feature.step()
            opt_class.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{params['num_iterations']}], Loss: {avg_loss:.4f}")

    # 测试
    model_feature.eval()
    model_class.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_middle = model_feature(x)
            outputs = model_class(x_middle)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"测试准确率: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()