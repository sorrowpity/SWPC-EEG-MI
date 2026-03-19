import torch
import numpy as np
from ShallowConvNet_Lyapunov import ShallowConvNet
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from EEG_cross_subject_loader_MI import EEG_loader
from scipy.linalg import inv, sqrtm

# 配置参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJ = 1


# =================================================================
# 🔥 补上缺失的标签归一化函数
# =================================================================
def normalize_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    normalized_labels = np.array([label_map[old] for old in labels])
    return normalized_labels


# 1. 核心：维度+索引全标准化
def standardize_eeg_dim(X):
    if len(X.shape) > 3:
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
    elif len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    return X


def standardize_subj_indices(subj_indices, n_samples):
    if np.isscalar(subj_indices):
        subj_indices = np.full(n_samples, subj_indices)
    else:
        subj_indices = np.array(subj_indices).reshape(-1)
        if len(subj_indices) > n_samples:
            subj_indices = subj_indices[:n_samples]
        elif len(subj_indices) < n_samples:
            subj_indices = np.pad(subj_indices, (0, n_samples - len(subj_indices)), 'constant')
    return subj_indices


# 2. EA对齐函数
def compute_EA_matrix(X):
    X = standardize_eeg_dim(X)
    n_samples, n_channels, n_timepoints = X.shape
    cov = np.zeros((n_channels, n_channels))
    for i in range(n_samples):
        trial_cov = np.dot(X[i], X[i].T) / n_timepoints
        cov += trial_cov
    mean_cov = cov / n_samples
    try:
        R = inv(sqrtm(mean_cov)).real
    except:
        R = np.eye(n_channels)
    return R


def apply_EA_to_dataset(X, subj_indices):
    X_ori = X
    X = standardize_eeg_dim(X)
    n_samples = X.shape[0]
    subj_indices = standardize_subj_indices(subj_indices, n_samples)

    X_aligned = np.zeros_like(X, dtype=np.float32)
    unique_subjs = np.unique(subj_indices)
    print(f"--- 测试集EA对齐，共{len(unique_subjs)}个被试，通道数{X.shape[1]} ---")

    for subj in unique_subjs:
        mask = (subj_indices == subj)
        X_subj = X[mask]
        if len(X_subj) == 0:
            continue
        R = compute_EA_matrix(X_subj)
        X_aligned[mask] = np.matmul(R, X_subj)

    if len(X_ori.shape) > 3:
        X_aligned = X_aligned.reshape(X_ori.shape)
    return X_aligned


# 3. 测试函数（最终完美版）
def test_model():
    # ====================== 预筛选模块测试 ======================
    print(">>> 测试预筛选模块（Rest vs MI）")
    loader_rest = EEG_loader_resting(test_subj=TEST_SUBJ)
    test_x = loader_rest.test_x
    test_y = loader_rest.test_y

    test_x = standardize_eeg_dim(test_x)
    test_y = np.array(test_y).reshape(-1)[:test_x.shape[0]]
    n_samples = test_x.shape[0]

    if hasattr(loader_rest, 'test_subj_id'):
        test_subj = loader_rest.test_subj_id
    else:
        test_subj = TEST_SUBJ
    test_subj = standardize_subj_indices(test_subj, n_samples)

    print(f"预筛选测试样本：{n_samples}个 | Rest(0){sum(test_y == 0)}个 | MI(1){sum(test_y == 1)}个")
    test_x_ea = apply_EA_to_dataset(test_x, test_subj)

    n_channels = test_x.shape[1]
    n_timepoints = test_x.shape[2]
    model_1 = ShallowConvNet(
        num_classes=2,
        channels=n_channels,
        time_points=n_timepoints,
        dropout_rate=0.6
    ).to(DEVICE)
    model_1.load_state_dict(torch.load('prescreen_model_best_sl_ssl_cnet.pth', map_location=DEVICE), strict=False)
    model_1.eval()

    test_x_tensor = torch.from_numpy(test_x_ea).float().to(DEVICE)
    test_x_tensor = (test_x_tensor - test_x_tensor.mean(dim=-1, keepdim=True)) / (
                test_x_tensor.std(dim=-1, keepdim=True) + 1e-6)
    test_x_tensor = test_x_tensor.unsqueeze(1)

    with torch.no_grad():
        outputs = model_1(test_x_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    correct = np.sum(preds == test_y)
    prescreen_acc = 100 * correct / n_samples
    print(f"预筛选模块测试精度：{prescreen_acc:.1f}% (正确{correct}/{n_samples})")

    # ====================== 分类模块测试（完全修复） ======================
    print("\n>>> 测试分类模块（Left vs Right）")
    loader_cls = EEG_loader(test_subj=TEST_SUBJ)
    test_x_cls = loader_cls.test_x
    test_y_cls = loader_cls.test_y

    test_x_cls = standardize_eeg_dim(test_x_cls)
    test_y_cls = np.array(test_y_cls).reshape(-1)[:test_x_cls.shape[0]]

    # 🔥 正确标签过滤（保留 1 和 2）
    valid_mask = (test_y_cls == 1) | (test_y_cls == 2)
    test_x_cls = test_x_cls[valid_mask]
    test_y_cls = test_y_cls[valid_mask]

    # 🔥 归一化标签
    test_y_cls = normalize_labels(test_y_cls)
    n_samples_cls = len(test_x_cls)

    if hasattr(loader_cls, 'test_subj_id'):
        test_subj_cls = loader_cls.test_subj_id
    else:
        test_subj_cls = TEST_SUBJ
    test_subj_cls = standardize_subj_indices(test_subj_cls, n_samples_cls)

    print(f"分类测试样本：{n_samples_cls}个 | Left(0){sum(test_y_cls == 0)}个 | Right(1){sum(test_y_cls == 1)}个")
    test_x_ea_cls = apply_EA_to_dataset(test_x_cls, test_subj_cls)

    n_channels_cls = test_x_cls.shape[1]
    n_timepoints_cls = test_x_cls.shape[2]

    # 🔥 模型用 ShallowConvNet
    model_2 = ShallowConvNet(
        num_classes=2,
        channels=n_channels_cls,
        time_points=n_timepoints_cls,
        dropout_rate=0.6
    ).to(DEVICE)
    model_2.load_state_dict(torch.load('classifier_model_best_sl_ssl_cnet.pth', map_location=DEVICE), strict=False)
    model_2.eval()

    test_x_cls_tensor = torch.from_numpy(test_x_ea_cls).float().to(DEVICE)
    test_x_cls_tensor = (test_x_cls_tensor - test_x_cls_tensor.mean(dim=-1, keepdim=True)) / (
                test_x_cls_tensor.std(dim=-1, keepdim=True) + 1e-6)
    test_x_cls_tensor = test_x_cls_tensor.unsqueeze(1)

    with torch.no_grad():
        outputs_cls = model_2(test_x_cls_tensor)
        if isinstance(outputs_cls, tuple):
            outputs_cls = outputs_cls[0]
        preds_cls = torch.argmax(outputs_cls, dim=1).cpu().numpy()
    correct_cls = np.sum(preds_cls == test_y_cls)
    classifier_acc = 100 * correct_cls / n_samples_cls
    print(f"分类模块测试精度：{classifier_acc:.1f}% (正确{correct_cls}/{n_samples_cls})")

    print("\n=== 测试总结 & 优化建议 ===")
    print(f"1. 预筛选模块（Rest vs MI）：测试精度 {prescreen_acc:.1f}%")
    print(f"2. 分类模块（Left vs Right）：测试精度 {classifier_acc:.1f}%")


if __name__ == '__main__':
    test_model()