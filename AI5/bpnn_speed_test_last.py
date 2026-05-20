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

# 取 10000 个样本用于分组测速
np.random.seed(42)
idxs = np.random.choice(len(X_test), 10000, replace=False)
X_test = X_test[idxs]
X_test_tensor = torch.tensor(X_test, device=device)

# ===================== AI 模型 =====================
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

model_ai = BPNN().to(device)
model_ai.load_state_dict(torch.load("BPNN_Final_25dB.pth"))
model_ai.eval()

# ===================== 分组测速：每 100 个一组 =====================
group_size = 100
num_groups = len(X_test) // group_size  # 一共 100 组
group_times = []

print("=" * 60)
print(f"    BPNN 分组测速 | 共 {len(X_test)} 样本 | 每组 {group_size} 个")
print("=" * 60)

with torch.no_grad():
    # 先预热一次
    model_ai(X_test_tensor[:1])

    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        batch = X_test_tensor[start:end]

        t0 = time.time()
        model_ai(batch)
        t1 = time.time()

        total = t1 - t0
        avg_per_sample = total / group_size * 1000  # ms
        group_times.append(avg_per_sample)

        print(f"第 {i+1:2d} 组 | 平均单样本耗时：{avg_per_sample:.4f} ms")

# ===================== 画图 =====================
plt.figure(figsize=(12, 5))
plt.plot(range(1, num_groups+1), group_times, marker='o', markersize=3, linestyle='-', color='#4b7fff', linewidth=1)
plt.title("BPNN 每100样本平均单样本反演时间", fontsize=14)
plt.xlabel("分组序号", fontsize=12)
plt.ylabel("平均单样本耗时 (ms)", fontsize=12)
plt.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
#plt.savefig("BPNN分组速度变化曲线.png", dpi=300)
plt.show()