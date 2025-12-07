import torch
import argparse
from torch.utils.data import DataLoader
from data_loader import EEG_loader, EEGDataset
from SWPC.EEGNet import EEGNet_feature
from loss import SimCLR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BNCI2014001')
    parser.add_argument('--test_subj', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（带数据增强）
    data = EEG_loader(test_subj=args.test_subj, dataset=args.dataset)
    train_dataset = EEGDataset(data.train_x, data.train_y, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型（查询编码器和密钥编码器）
    model_q = EEGNet_feature(in_channels=22, n_dim=120).to(device)
    model_k = EEGNet_feature(in_channels=22, n_dim=120).to(device)
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False  # 不更新k

    # 优化器与损失
    optimizer = torch.optim.Adam(model_q.parameters(), lr=args.lr)
    criterion = SimCLR(temperature=0.1)

    # 训练
    for epoch in range(args.epochs):
        model_q.train()
        total_loss = 0.0
        for x1, x2, y in train_loader:
            x1, x2 = x1.unsqueeze(1).float().to(device), \
                x2.unsqueeze(1).float().to(device)

            # 前向传播
            q1 = model_q(x1)
            q2 = model_q(x2)
            with torch.no_grad():
                k1 = model_k(x1)
                k2 = model_k(x2)

            # 构建特征对
            features = torch.cat([q1.unsqueeze(1), q2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels=y.to(device))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新k（EMA）
            for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
                param_k.data = param_k.data * 0.999 + param_q.data * (1 - 0.999)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()