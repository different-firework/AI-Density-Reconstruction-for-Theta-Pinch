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

# 取 2000 个样本测速
np.random.seed(42)
idxs = np.random.choice(len(X_test), 1000, replace=False)
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

# ===================== 传统 Abel =====================
X_chord = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669, 0.0831, 0.0993, 0.1155, 0.1318, 0.15])
dr_abel = 0.002
r_abel  = np.arange(0, 0.15 + dr_abel, dr_abel)
r_final = np.linspace(0, 0.15, 21)

def threesimple(X, Y):
    X, Y = np.float64(X), np.float64(Y)
    n = len(X)
    h = np.diff(X)
    lam, mu = np.zeros(n-1), np.zeros(n-1)
    for i in range(1, n-1):
        lam[i-1] = h[i-1] / (h[i-1]+h[i])
        mu[i-1] = 1 - lam[i-1]
    d = np.zeros(n)
    for i in range(1, n-1):
        d[i] = 6 * ((Y[i+1]-Y[i])/h[i] - (Y[i]-Y[i-1])/h[i-1]) / (h[i-1]+h[i])
    A = np.diag(np.ones(n)*2)
    for i in range(1, n-1):
        A[i,i-1] = mu[i-1]
        A[i,i+1] = lam[i-1]
    M = np.linalg.solve(A, d)
    return None, h, None, None, M

def threesimple1(X, Y, x):
    D, h, A, g, M = threesimple(X, Y)
    n, m = len(X), len(x)
    s = np.zeros(m)
    for t in range(m):
        for i in range(n-1):
            if X[i] <= x[t] <= X[i+1]:
                t1 = M[i]*(X[i+1]-x[t])**3/(6*h[i])
                t2 = M[i+1]*(x[t]-X[i])**3/(6*h[i])
                t3 = (Y[i]-M[i]*h[i]**2/6)*(X[i+1]-x[t])/h[i]
                t4 = (Y[i+1]-M[i+1]*h[i]**2/6)*(x[t]-X[i])/h[i]
                s[t] = t1+t2+t3+t4
                break
        else: s[t]=0
    return s

def traditional_abel(neL_9):
    neL_k = np.hstack([neL_9, 0.0])
    s = threesimple1(X_chord, neL_k, r_abel)
    f = np.zeros_like(r_abel)
    for i in range(len(r_abel)):
        d = 0.0
        for j in range(i, len(r_abel)-1):
            den = np.sqrt((r_abel[j]+dr_abel)**2 - r_abel[i]**2 + 1e-12)
            d += -(s[j+1]-s[j])/np.pi / den
        f[i] = d
    return np.interp(r_final, r_abel, f)

# ===================== 测速 =====================
print("="*60)
print("           速度对比测试（1000 样本）")
print("="*60)

# 传统 Abel
t0 = time.time()
for x in X_test:
    traditional_abel(x)
t_abel = time.time() - t0
avg_abel = t_abel / len(X_test) * 1000

# AI 模型
with torch.no_grad():
    model_ai(X_test_tensor[:1])  # 预热
t0 = time.time()
model_ai(X_test_tensor)
t_ai = time.time() - t0
avg_ai = t_ai / len(X_test) * 1000

# 输出
print(f"传统 Abel  总耗时: {t_abel:.3f} s   | 单样本: {avg_abel:.3f} ms")
print(f"BPNN 模型   总耗时: {t_ai:.4f} s   | 单样本: {avg_ai:.3f} ms")
print(f"🚀 AI 速度提升: {avg_abel / avg_ai:.1f} 倍")

# ===================== 画图 =====================
methods = ["传统 Abel", "BPNN 模型"]
times   = [avg_abel, avg_ai]
colors  = ["#ff4b4b", "#4b7fff"]

plt.figure(figsize=(7,5))
bars = plt.bar(methods, times, color=colors, width=0.5)
plt.bar_label(bars, fmt='%.3f ms', fontsize=12)

plt.title("单样本平均推理耗时对比", fontsize=14)
plt.ylabel("平均耗时 (ms)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("速度对比柱状图1000.png", dpi=300)
plt.show()