import numpy as np
from scipy.linalg import inv, sqrtm


def compute_EA_matrix(eeg_data):
    """
    計算 EA 對齊矩陣 R = mean(X * X^T)^(-1/2)
    eeg_data: (N_trials, Channels, TimePoints)
    """
    n_trials = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]
    cov = np.zeros((n_channels, n_channels))

    for i in range(n_trials):
        # 計算單個 trial 的協方差
        X = eeg_data[i]
        trial_cov = np.dot(X, X.T) / X.shape[1]
        cov += trial_cov

    mean_cov = cov / n_trials
    # 計算矩陣的負二分之一次方
    R = inv(sqrtm(mean_cov)).real
    return R


def apply_EA(window, R):
    """
    應用對齊矩陣到當前窗口
    window: (Channels, TimePoints)
    """
    return np.dot(R, window)