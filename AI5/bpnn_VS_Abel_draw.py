import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 你的模型结构：9-48-24-21，完全匹配
class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 21),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# 你的Abel反演代码（完全保留）
def Thomas(D, h, g):
    n = len(g)
    A, B, C = np.zeros(n), np.zeros(n), np.zeros(n)
    A[0] = 0; B[0] = 1; C[0] = 0
    A[-1] = 0; B[-1] = 1; C[-1] = 0
    for i in range(1, n-1):
        A[i] = D[i, 0]
        B[i] = 2
        C[i] = D[i, 1]
    for i in range(1, n):
        A[i] /= B[i-1]
        B[i] -= A[i] * C[i-1]
        g[i] -= A[i] * g[i-1]
    M = np.zeros(n)
    M[-1] = g[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (g[i] - C[i] * M[i+1]) / B[i]
    return M

def threesimple(X, Y):
    n = len(X)
    h = np.diff(X)
    g = np.zeros(n)
    D = np.zeros((n, 2))
    for i in range(1, n-1):
        D[i, 0] = h[i-1]/(h[i-1]+h[i])
        D[i, 1] = h[i]/(h[i-1]+h[i])
        g[i] = 6 * ((Y[i+1]-Y[i])/h[i] - (Y[i]-Y[i-1])/h[i-1]) / (h[i-1]+h[i])
    g[0] = g[-1] = 0
    return D, h, Y, g, Thomas(D, h, g)

def threesimple1(X, Y, x):
    D, h, A, g, M = threesimple(X, Y)
    s = np.zeros_like(x)
    for t in range(len(x)):
        for i in range(len(X)-1):
            if X[i] <= x[t] <= X[i+1]:
                dx = X[i+1] - X[i]
                s[t] = M[i]*(X[i+1]-x[t])**3/(6*dx) + \
                       M[i+1]*(x[t]-X[i])**3/(6*dx) + \
                       (A[i] - M[i]*dx**2/6)*(X[i+1]-x[t])/dx + \
                       (A[i+1] - M[i+1]*dx**2/6)*(x[t]-X[i])/dx
                break
    return s

def abel_invert(real9):
    X = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669, 0.0831, 0.0993, 0.1155, 0.1318, 0.15])
    Y = np.hstack([real9, 0.0])
    r = np.arange(0, 0.15+0.002, 0.002)
    s = threesimple1(X, Y, r)
    f = np.zeros_like(r)
    for i in range(len(r)):
        d = 0
        for j in range(i, len(r)-1):
            dy = s[j+1] - s[j]
            rad = np.sqrt(r[j+1]**2 - r[i]**2 + 1e-12)
            d += (-1.0 / np.pi) * dy / rad
        f[i] = d
    f[f < 0] = 0
    return r, f

# 1. 加载模型
device = torch.device('cpu')
model = BPNN().to(device)
model.load_state_dict(torch.load('BPNN_Final_25dB.pth', map_location=device))
model.eval()

# 2. 加载真实数据 1228011.mat，取0.56ms
data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
data = scipy.io.loadmat(data_path)
nel = data['nel']
Tinterval = 2.49999985157956e-08
time_axis = np.arange(nel.shape[0]) * Tinterval * 1000
k = np.argmax(time_axis >= 0.58)
real9 = nel[k]

# 3. BPNN反演：按训练集方式归一化 + 输出还原
# 关键：这里用真实数据的最大值来匹配模型的0~1输出
x_norm = real9 / np.max(real9)
with torch.no_grad():
    bpnn_out_norm = model(torch.tensor(x_norm, dtype=torch.float32)).numpy()

# 还原尺度：按Abel曲线的最大值来匹配模型输出
r_bpnn = np.linspace(0, 0.15, 21)
r_abel, f_abel = abel_invert(real9)
bpnn_out = bpnn_out_norm * np.max(f_abel)

# 4. 画图对比
plt.figure(figsize=(10, 6))
plt.plot(r_abel, f_abel, 'r-o', lw=3, label='Abel 反演')
plt.plot(r_bpnn, bpnn_out, 'b-', lw=3, label='BPNN 反演')
plt.grid(True)
plt.legend(fontsize=12)
plt.xlabel('半径 m', fontsize=12)
plt.ylabel('电子密度', fontsize=12)
plt.title('0.58ms 时刻 | Abel vs BPNN 对比', fontsize=14)
plt.tight_layout()
plt.show()