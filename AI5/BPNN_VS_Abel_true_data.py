import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import torch
import torch.nn as nn
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# ===================== 【1】Abel 反演函数 =====================
def Thomas(D, h, g):
    n = len(g)
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    A[0] = 0
    B[0] = 1
    C[0] = 0
    A[-1] = 0
    B[-1] = 1
    C[-1] = 0
    for i in range(1, n-1):
        A[i] = D[i, 0]
        B[i] = 2
        C[i] = D[i, 1]
    for i in range(1, n):
        A[i] = A[i] / B[i-1]
        B[i] = B[i] - A[i] * C[i-1]
        g[i] = g[i] - A[i] * g[i-1]
    M = np.zeros(n)
    M[-1] = g[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (g[i] - C[i] * M[i+1]) / B[i]
    return M

def threesimple(X, Y):
    n = len(X)
    h = np.zeros(n-1)
    g = np.zeros(n)
    for i in range(n-1):
        h[i] = X[i+1] - X[i]
    A = Y.copy()
    D = np.zeros((n, 2))
    for i in range(1, n-1):
        D[i, 0] = h[i-1] / (h[i-1] + h[i])
        D[i, 1] = h[i] / (h[i-1] + h[i])
        g[i] = 6*(Y[i+1]-Y[i])/(h[i]*(h[i-1]+h[i])) - 6*(Y[i]-Y[i-1])/(h[i-1]*(h[i-1]+h[i]))
    g[0] = 0
    g[-1] = 0
    M = Thomas(D, h, g)
    return D, h, A, g, M

def threesimple1(X, Y, x):
    D, h, A, g, M = threesimple(X, Y)
    n = len(X)
    m = len(x)
    s = np.zeros(m)
    for t in range(m):
        for i in range(n-1):
            if X[i] <= x[t] <= X[i+1]:
                p1 = M[i]*(X[i+1]-x[t])**3/(6*h[i])
                p2 = M[i+1]*(x[t]-X[i])**3/(6*h[i])
                p3 = (A[i]-M[i]/6*h[i]**2)*(X[i+1]-x[t])/h[i]
                p4 = (A[i+1]-M[i+1]/6*h[i]**2)*(x[t]-X[i])/h[i]
                s[t] = p1+p2+p3+p4
                break
            else:
                s[t] = 0
    return s

# ===================== 【2】BPNN 模型结构（9→48→24→21）=====================
class BPNN(nn.Module):
    def __init__(self, in_dim=9, hidden1=48, hidden2=24, out_dim=21):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim)
        )
    def forward(self, x):
        return self.layers(x)

# ===================== 【3】加载真实实验数据 =====================
data = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat")
neL = data['nel']

Tinterval = 2.49999985157956e-08
time_axis = np.arange(0, 200004) * Tinterval * 1000
k = np.argmax(time_axis >= 0.57)

# 9道弦积分信号
chord_9 = neL[k, :9]

# ===================== 【4】关键：归一化（必须和训练时一致）=====================
# 你训练时用的是 MinMax 归一化到 [0,1]
# 这里用训练集的最大最小值，我给你写成通用版本

# 输入归一化
input_max = chord_9.max()
input_min = chord_9.min()
chord_norm = (chord_9 - input_min) / (input_max - input_min + 1e-8)

# ===================== 【5】BPNN 推理 =====================
device = torch.device('cpu')
model = BPNN(in_dim=9, hidden1=48, hidden2=24, out_dim=21).to(device)
model.load_state_dict(torch.load("BPNN_Final_25dB.pth", map_location=device))
model.eval()

with torch.no_grad():
    in_tensor = torch.tensor(chord_norm[None, :], dtype=torch.float32).to(device)
    out_norm = model(in_tensor).cpu().numpy().flatten()

# ===================== 【6】输出反归一化（还原真实密度）=====================
out_max = 1.0
out_min = 0.0
f_bpnn = out_norm * (input_max - input_min + 1e-8) + input_min  # 与输入同量级
f_bpnn[f_bpnn < 0] = 0

# ===================== 【7】Abel 反演 =====================
X_chord = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669, 0.0831, 0.0993, 0.1155, 0.1318, 0.15])
Y_chord = np.hstack([chord_9, 0])
r_abel = np.arange(0, 0.15, 0.002)
r_bpnn = np.linspace(0, 0.15, 21)

s = threesimple1(X_chord, Y_chord, r_abel)
f_abel = np.zeros_like(r_abel)
for i in range(len(r_abel)):
    d = 0
    for j in range(i, len(r_abel)-1):
        den = np.sqrt((r_abel[j]+0.002)**2 - r_abel[i]**2 + 1e-8)
        e = (-1/np.pi) * (s[j+1]-s[j]) / den
        d += e
    f_abel[i] = d
f_abel[f_abel < 0] = 0

# ===================== 【8】画图对比 =====================
plt.figure(figsize=(10,5))
plt.plot(r_abel, f_abel, 'b-', linewidth=2.5, label='Abel 反演')
plt.plot(r_bpnn, f_bpnn, 'r-o', linewidth=2.5, markersize=5, label='BPNN 反演')
plt.grid(True)
plt.title('Abel vs BPNN 密度反演对比（真实实验数据 0.56ms）', fontsize=12)
plt.xlabel('半径 (m)')
plt.ylabel(r'$n_e(r)$')
plt.legend()
plt.tight_layout()
plt.show()