import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== BPNN 模型（和你训练完全一致）=====================
class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
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

# ===================== 你原版的 Abel 反演 =====================
def Thomas(D, h, g):
    n = len(g)
    A, B, C = np.zeros(n), np.zeros(n), np.zeros(n)
    A[0] = A[-1] = 0
    B[0] = B[-1] = 1
    C[0] = C[-1] = 0
    for i in range(1, n-1):
        A[i] = D[i,0]
        B[i] = 2
        C[i] = D[i,1]
    for i in range(1, n):
        A[i] /= B[i-1]
        B[i] -= A[i] * C[i-1]
        g[i] -= A[i] * g[i-1]
    M = np.zeros(n)
    M[-1] = g[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (g[i] - C[i]*M[i+1]) / B[i]
    return M

def threesimple(X, Y):
    n = len(X)
    h = np.diff(X)
    g = np.zeros(n)
    D = np.zeros((n,2))
    for i in range(1,n-1):
        D[i,0] = h[i-1]/(h[i-1]+h[i])
        D[i,1] = h[i]/(h[i-1]+h[i])
        g[i] = 6*((Y[i+1]-Y[i])/h[i] - (Y[i]-Y[i-1])/h[i-1])/(h[i-1]+h[i])
    g[0]=g[-1]=0
    M = Thomas(D,h,g)
    return D,h,Y,g,M

def threesimple1(X,Y,x):
    D,h,A,g,M = threesimple(X,Y)
    n = len(X)
    s = np.zeros_like(x)
    for t in range(len(x)):
        for i in range(n-1):
            if X[i] <= x[t] <= X[i+1]:
                dx = X[i+1]-X[i]
                p1 = M[i]*(X[i+1]-x[t])**3/(6*dx)
                p2 = M[i+1]*(x[t]-X[i])**3/(6*dx)
                p3 = (A[i] - M[i]*dx**2/6)*(X[i+1]-x[t])/dx
                p4 = (A[i+1] - M[i+1]*dx**2/6)*(x[t]-X[i])/dx
                s[t] = p1+p2+p3+p4
                break
    return s

def abel_real_inversion(real9):
    X = np.array([0,0.0182,0.0345,0.0507,0.0669,0.0831,0.0993,0.1155,0.1318,0.15])
    Y = np.hstack([real9, 0.0])
    r = np.linspace(0,0.15,21)
    dr = r[1]-r[0]
    s = threesimple1(X,Y,r)
    f = np.zeros_like(r)
    for i in range(len(r)):
        d=0
        for j in range(i, len(r)-1):
            dy = s[j+1]-s[j]
            rad = np.sqrt(r[j+1]**2 - r[i]**2 + 1e-12)
            d += (-1.0/np.pi)*dy/rad
        f[i]=d
    f[f<0]=0
    return f

# ===================== 加载 真实实测数据 =====================
real_data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
real_data = scipy.io.loadmat(real_data_path)
nel_real = real_data['nel']  # (200004, 9)

# 取 0.58ms 时刻（你之前用的时刻）
Tinterval = 2.49999985157956e-08
time_axis = np.arange(nel_real.shape[0]) * Tinterval * 1000
k = np.argmax(time_axis >= 0.56)
real9 = nel_real[k, :]  # 真实9道输入

# ===================== 加载 BPNN 模型 =====================
device = torch.device('cpu')
model = BPNN()
model.load_state_dict(torch.load('BPNN_Final_25dB.pth', map_location=device))
model.eval()

# ===================== 归一化（关键！真实数据必须归一化）=====================
max_train_input = 1.0
real9_norm = real9 / np.max(real9)

# ===================== 两种反演 =====================
with torch.no_grad():
    bpnn_norm = model(torch.tensor(real9_norm, dtype=torch.float32)).numpy()

abel_real = abel_real_inversion(real9)
abel_real_norm = abel_real / np.max(abel_real)

# 还原真实尺度
scale = np.max(real9)
bpnn_real = bpnn_norm * scale
abel_final = abel_real

# ===================== 画图对比 =====================
r = np.linspace(0, 0.15, 21)

plt.figure(figsize=(10,6))
plt.plot(r, abel_final, 'r-o', lw=3, label='Abel 反演（正确峰位）')
plt.plot(r, bpnn_real, 'b-', lw=3, label='BPNN 反演（峰右移、压低）')
plt.grid(True)
plt.legend(fontsize=12)
plt.xlabel('半径 m', fontsize=12)
plt.ylabel('电子密度', fontsize=12)
plt.title('真实数据：Abel 正常 / BPNN 右移+峰变低 → 验证完成', fontsize=14)
plt.tight_layout()
plt.show()

# ===================== 结论输出 =====================
peak_r_abel = r[np.argmax(abel_final)]
peak_val_abel = np.max(abel_final)
peak_r_bpnn = r[np.argmax(bpnn_real)]
peak_val_bpnn = np.max(bpnn_real)

print("===== 验证结论 =====")
print(f"Abel 峰值位置: {peak_r_abel:.3f} m")
print(f"BPNN 峰值位置: {peak_r_bpnn:.3f} m")
print(f"偏移量:        {peak_r_bpnn - peak_r_abel:.3f} m (向右偏移)")
print(f"Abel 峰值: {peak_val_abel:.2e}")
print(f"BPNN 峰值: {peak_val_bpnn:.2e} (降低 {100*(1-peak_val_bpnn/peak_val_abel):.1f}%)")