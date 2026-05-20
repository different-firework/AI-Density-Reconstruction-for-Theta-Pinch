import numpy as np
import scipy.io as sio

# ===================== 1. 加载你的原始归一化数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X_data = data["input"].T.astype(np.float32)   # 形状 (N,9) → 每一行就是一组neL

# 随便选第 k 条数据做测试（比如第 0 条，你可以随便换 k）
k = 0
neL_9 = X_data[k]  # 这就是你要的 neL(k,1)~neL(k,9)

# 第10位补 0，和师兄完全一致
neL_k = np.hstack([neL_9, 0.0])
print("neL(k,:) =", neL_k)

# ===================== 2. 师兄的三弯矩插值 + Abel 逆变换 =====================
X = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669, 0.0831, 0.0993, 0.1155, 0.1318, 0.15])
dr = 0.002
r = np.arange(0, 0.15 + dr, dr)

# ---------- threesimple 三弯矩 ----------
def threesimple(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)
    if n < 2:
        return None, None, None, None, np.zeros_like(X)
    h = np.diff(X)
    lam = np.zeros(n-1)
    mu  = np.zeros(n-1)
    for i in range(1, n-1):
        lam[i-1] = h[i-1] / (h[i-1] + h[i])
        mu[i-1]  = 1.0 - lam[i-1]
    d = np.zeros(n)
    for i in range(1, n-1):
        d[i] = 6 * ((Y[i+1]-Y[i])/h[i] - (Y[i]-Y[i-1])/h[i-1]) / (h[i-1]+h[i])
    A = np.diag(np.ones(n)*2)
    for i in range(1, n-1):
        A[i,i-1] = mu[i-1]
        A[i,i+1] = lam[i-1]
    M = np.linalg.solve(A, d)
    return None, h, None, None, M

# ---------- threesimple1 插值 ----------
def threesimple1(X, Y, x):
    D, h, A, g, M = threesimple(X, Y)
    n = len(X)
    m = len(x)
    s = np.zeros(m)
    for t in range(m):
        for i in range(n-1):
            if X[i] <= x[t] <= X[i+1]:
                t1 = M[i] * (X[i+1]-x[t])**3 / (6*h[i])
                t2 = M[i+1]*(x[t]-X[i])**3 / (6*h[i])
                t3 = (Y[i] - M[i]*h[i]**2/6) * (X[i+1]-x[t])/h[i]
                t4 = (Y[i+1]-M[i+1]*h[i]**2/6)*(x[t]-X[i])/h[i]
                s[t] = t1 + t2 + t3 + t4
                break
        else:
            s[t] = 0
    return s

# ---------- 插值得到 s ----------
s = threesimple1(X, neL_k, r)

# ---------- Abel 逆变换 ----------
f_traditional = np.zeros_like(r)
for i in range(len(r)):
    d = 0.0
    for j in range(i, len(r)-1):
        num = -(s[j+1] - s[j]) / np.pi
        den = np.sqrt((r[j]+dr)**2 - r[i]**2 + 1e-12)
        d += num / den
    f_traditional[i] = d

# ===================== 结果 =====================
print("\n传统 Abel 反演密度剖面 f =")
print(f_traditional)