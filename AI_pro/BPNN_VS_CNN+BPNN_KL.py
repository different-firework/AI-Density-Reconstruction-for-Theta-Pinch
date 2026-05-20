import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 基础设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 加载数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
y = data["output"].T.astype(np.float32)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = torch.tensor(X_test, device=device)
y_test = torch.tensor(y_test, device=device)

# ===================== 模型 =====================
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
bpnn = BPNN().to(device)
cnnbp = CNN_BPNN_Medium().to(device)

bpnn.load_state_dict(torch.load("BPNN_Final_25dB.pth", map_location=device))
cnnbp.load_state_dict(torch.load("中模型_带25dB噪声.pth", map_location=device))

bpnn.eval()
cnnbp.eval()

# ===================== 批量加噪声 =====================
def add_noise(x, snr_db):
    if snr_db is None:
        return x
    x_pwr = x.pow(2).mean()
    noise_pwr = x_pwr / (10 ** (snr_db / 10))
    return x + torch.randn_like(x) * torch.sqrt(noise_pwr)

# ===================== 批量 KL 散度（超快！） =====================
def batch_kl(model, x, y, snr):
    with torch.no_grad():
        x_in = add_noise(x, snr)
        pred = model(x_in)
        pred = torch.clamp(pred, 1e-8, 1 - 1e-8)
        true = torch.clamp(y, 1e-8, 1 - 1e-8)
        kl = (true * torch.log(true / pred)).sum(dim=1).mean().item()
    return kl

# ===================== 测试 =====================
snr_levels = [None, 40, 35, 30, 25, 20]
labels = ["Clean", "40dB", "35dB", "30dB", "25dB", "20dB"]

kl_bp = [batch_kl(bpnn, X_test, y_test, s) for s in snr_levels]
kl_cn = [batch_kl(cnnbp, X_test, y_test, s) for s in snr_levels]

# ===================== 画图 =====================
plt.figure(figsize=(10,5))
plt.plot(labels, kl_bp, 'ro-', label='BPNN', lw=2, ms=8)
plt.plot(labels, kl_cn, 'bo-', label='CNN+BPNN', lw=2, ms=8)
plt.xlabel("噪音水平")
plt.ylabel("平均KL散度")
plt.title("不同噪音水平下KL散度对比")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("BPNN:", kl_bp)
print("CNN+BPNN:", kl_cn)