import numpy as np
from nolds import lyap_r  # 需要安装 nolds 库: pip install nolds


def compute_batch_lyapunov(data_batch, fs=250):
    """
    计算 batch 中每个样本的平均最大 Lyapunov 指数
    data_batch: (Batch, Channels, TimePoints)
    """
    batch_size = data_batch.shape[0]
    lyap_exps = []

    # 为了计算效率，我们可以选取关键通道（如 C3, Cz, C4）或者对通道取平均
    # 也可以降采样。这里采用通道平均后的近似
    for i in range(batch_size):
        signal = np.mean(data_batch[i], axis=0)  # 融合多通道动力学
        try:
            # 这里的参数需要根据你的数据长度调整，250点较短，使用近似算法
            l = lyap_r(signal, emb_dim=10, lag=None)
        except:
            l = 0.0  # 异常处理
        lyap_exps.append(l)

    return np.array(lyap_exps, dtype=np.float32)