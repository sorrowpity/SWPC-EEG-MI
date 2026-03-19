import torch

print(f"PyTorch 版本: {torch.__version__}")
# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

if cuda_available:
    # 打印可用的 GPU 信息
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 警告：PyTorch 未检测到 CUDA。请检查驱动、CUDA Toolkit 和 PyTorch 安装。")