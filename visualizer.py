# visualizer.py
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, title="Training History", save_path="history.png"):
    """
    繪製 訓練集 與 驗證集 的 準確率 和 損失 曲線
    history: 應包含 'loss', 'acc', 'val_loss', 'val_acc' 四個鍵
    """
    epochs = range(1, len(history['loss']) + 1)

    # 設置全局字體，防止中文亂碼（可選）
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用更美觀的圖表樣式
    plt.figure(figsize=(14, 6))

    # --- 1. 繪製 Loss 曲線 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'r-o', markersize=2, alpha=0.7, label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'orange', linestyle='--', label='Val Loss')

    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- 2. 繪製 Accuracy 曲線 ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], 'b-o', markersize=2, alpha=0.7, label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], 'cyan', linestyle='--', label='Val Acc')

    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # 準確率固定在 0-100 方便觀察
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 提高解析度
    print(f"📈 訓練可視化圖表（含驗證集）已保存至: {save_path}")
    plt.close()