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

test_names = ['Clean', '40dB', '35dB', '30dB', '25dB', '20dB']
test_dB    = [100, 40, 35, 30, 25, 20]
r_final = np.linspace(0, 0.15, 21)

# ===================== 加载数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
Y = data["output"].T.astype(np.float32)
_, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 随机选100个
np.random.seed(42)
idx = np.random.choice(len(X_test), 100, replace=False)
X_test = X_test[idx]
y_test = y_test[idx]
X_test_tensor = torch.tensor(X_test, device=device)

# ===================== AI模型 =====================
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

# ===================== 加噪声 =====================
def add_noise(x, dB):
    if dB >= 100: return x.clone()
    sig_pow = torch.mean(x**2)
    noise_pow = sig_pow / (10 ** (dB / 10))
    noise = torch.randn_like(x) * torch.sqrt(noise_pow)
    return x + noise

# ===================== 统一归一化函数 =====================
def normalize_profile(pred, true):
    pred = np.maximum(pred, 1e-8)
    sum_p = np.sum(pred, axis=1, keepdims=True)
    sum_t = np.sum(true, axis=1, keepdims=True)
    return pred * sum_t / (sum_p + 1e-8)

# ===================== KL 计算 =====================
def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / np.sum(p, axis=1, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    return np.mean(np.sum(p * np.log(p / q), axis=1))

# ===================== 测试 =====================
print("="*70)
print("           传统方法 vs 25dB AI（Abel已归一化）")
print("="*70)

kl_trad = []
kl_ai = []

for i, dB in enumerate(test_dB):
    X_noisy = add_noise(X_test_tensor, dB)

    # AI
    with torch.no_grad():
        ai_out = model_ai(X_noisy).cpu().numpy()

    # 传统 Abel
    trad_out_raw = np.array([traditional_abel(x) for x in X_noisy.cpu().numpy()])

    # 先归一化再算KL
    trad_out = normalize_profile(trad_out_raw, y_test)

    # 计算KL
    kt = kl_divergence(y_test, trad_out)
    ka = kl_divergence(y_test, ai_out)

    kl_trad.append(kt)
    kl_ai.append(ka)

    # ✅ 修复这里！
    print(f"{test_names[i]:<6s} | 传统={kt:.4f} | AI={ka:.4f}")

# ===================== 画图 =====================
plt.figure(figsize=(10,5))
plt.plot(test_names, kl_trad, 'o-r', linewidth=3, label='传统Abel（归一化后）')
plt.plot(test_names, kl_ai,    's-b', linewidth=3, label='25dB AI模型')
plt.xlabel("噪声强度")
plt.ylabel("KL 散度")
plt.title("Abel与BPNN鲁棒性对比")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("KL对比_最终正确版.png", dpi=300)
plt.show()