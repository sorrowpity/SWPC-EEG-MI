#check_data.py
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
from scipy.io import loadmat

# 加載一個原始文件
mat_data = loadmat('./data/train_A5.mat')
X = mat_data['X'] # (Trials, Channels, Time)
y = np.squeeze(mat_data['y'])

# 挑選一個右手的試次
idx = np.where(y == 'right_hand')[0][0]
sample_eeg = X[idx, 6, :] # 取 C3 通道 (通常是索引 6 或 7)，那是運動皮層

# 1. 畫時域波形
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sample_eeg)
plt.axvspan(0, 250, color='green', alpha=0.2, label='Proposed Rest')
plt.axvspan(850, 1100, color='red', alpha=0.2, label='Proposed MI')
plt.title("EEG Time Series (C3 Channel)")
plt.legend()

# 2. 比較頻譜 (真相在這裡)
f_rest, p_rest = welch(sample_eeg[0:250], fs=250)
f_mi, p_mi = welch(sample_eeg[850:1100], fs=250)

plt.subplot(2, 1, 2)
plt.semilogy(f_rest, p_rest, label='Rest (0-1s)')
plt.semilogy(f_mi, p_mi, label='MI (3.4-4.4s)')
plt.xlim(0, 40)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title("Power Spectral Density (ERD Check)")
plt.legend()
plt.tight_layout()
plt.show()