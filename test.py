# test_data.py
import os
data_dir = './data/'
for subj in range(1, 6):
    train_file = os.path.join(data_dir, f'train_A{subj}.mat')
    test_file = os.path.join(data_dir, f'test_A{subj}.mat')
    print(f"train_A{subj}.mat 存在: {os.path.exists(train_file)}")
    print(f"test_A{subj}.mat 存在: {os.path.exists(test_file)}")