import scipy.io as io
data = io.loadmat('./data/train_A1.mat')
print(data['y'][:10]) # 看看前10個試次是什麼標籤