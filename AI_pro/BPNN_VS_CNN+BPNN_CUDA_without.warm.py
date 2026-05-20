import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 设备 & 绘图 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
_, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)
X_test = torch.tensor(X_test, device=device)

# ===================== BPNN =====================
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

# ===================== CNN+BPNN =====================
class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__(),
        self.cnn = nn.Sequential(
            nn.Conv1d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv1d(16,32,3,padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*9,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,21)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)

# ===================== 加载模型 =====================
bpnn = BPNN().to(device)
cnnbp = CNN_BPNN_Medium().to(device)

bpnn.load_state_dict(torch.load("BPNN_Final_25dB.pth"))
cnnbp.load_state_dict(torch.load("中模型_带25dB噪声.pth"))

bpnn.eval()
cnnbp.eval()

# ===================== ✅ NVIDIA 官方测速 =====================
def nvidia_speed(model, total=10000, step=100):
    times = []
    x = X_test[:1]

    # ===================== 官方步骤：先 Warm-up 排除初始化开销 =====================
    with torch.no_grad():
        for _ in range(10):
            model(x)

    # ===================== 正式测速（CUDA Event 高精度） =====================
    for _ in range(total // step):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start.record()
            for __ in range(step):
                model(x)
            end.record()
            torch.cuda.synchronize()

        t = start.elapsed_time(end) / step  # ms
        times.append(t)
    return times

# ===================== 运行 =====================
t1 = nvidia_speed(bpnn)
t2 = nvidia_speed(cnnbp)

# ===================== 画图 =====================
x = np.arange(100,10100,100)
plt.figure(figsize=(10,4))
plt.plot(x, t1, 'ro-', lw=2.5, label='BPNN')
plt.plot(x, t2, 'bo-', lw=2.5, label='CNN+BPNN')
plt.xlabel("样本数量")
plt.ylabel("单样本推理时间 (ms)")
plt.title("BPNN与BPNN+CNN速度对比(单样本循环推理)")
plt.legend()
plt.grid(True)
plt.show()

# ===================== 输出 =====================
print(" BPNN 平均速度：", round(np.mean(t1),4), "ms")
print(" CNN+BPNN 平均速度：", round(np.mean(t2),4), "ms")