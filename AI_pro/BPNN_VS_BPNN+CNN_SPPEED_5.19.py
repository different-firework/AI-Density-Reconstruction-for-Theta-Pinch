import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 基础设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 加载数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
_, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)

np.random.seed(42)
idxs = np.random.choice(len(X_test), 10000, replace=False)
X_test = X_test[idxs]
X_test_tensor = torch.tensor(X_test, device=device)

# ===================== BPNN 模型 =====================
class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 21), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# ===================== CNN+BPNN 模型 =====================
class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*9, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 21)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)

# ===================== 加载模型 =====================
model_bpnn = BPNN().to(device)
model_bpnn.load_state_dict(torch.load("BPNN_Final_25dB.pth"))
model_bpnn.eval()

model_cnn = CNN_BPNN_Medium().to(device)
model_cnn.load_state_dict(torch.load("中模型_带25dB噪声.pth"))
model_cnn.eval()

# ===================== 完全复刻你原版测速函数 =====================
def test_group_speed(model, name):
    group_size = 100
    num_groups = len(X_test_tensor) // group_size
    group_times = []

    print(f"\n{'='*60}")
    print(f"    {name} | 每组 {group_size} 个批量推理")
    print(f"{'='*60}")

    with torch.no_grad():
        model(X_test_tensor[:1])  # 预热
        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            batch = X_test_tensor[start:end]

            t0 = time.time()
            model(batch)
            t1 = time.time()

            avg = (t1 - t0) / group_size * 1000
            group_times.append(avg)
            print(f"第 {i+1:2d} 组 | 平均单样本：{avg:.4f} ms")
    return group_times

# ===================== 开始测速 =====================
bpnn_times = test_group_speed(model_bpnn, "BPNN")
cnn_times = test_group_speed(model_cnn, "CNN+BPNN")

# ===================== 统一画图 =====================
plt.figure(figsize=(12,5))
plt.plot(range(1, 101), bpnn_times, 'ro-', markersize=3, label='BPNN', linewidth=1)
plt.plot(range(1, 101), cnn_times, 'bo-', markersize=3, label='CNN+BPNN', linewidth=1)
plt.title("BPNN vs CNN+BPNN 每100样本平均速度", fontsize=14)
plt.xlabel("分组序号", fontsize=12)
plt.ylabel("平均单样本耗时 (ms)", fontsize=12)
plt.grid(linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()