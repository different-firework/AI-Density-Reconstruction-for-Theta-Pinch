import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# ===================== 三次样条核心函数（已修复）=====================
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

    for i in range(1, n - 1):
        A[i] = D[i, 0]  # 修复：只有 0 和 1，没有 2
        B[i] = 2
        C[i] = D[i, 1]  # 修复：这里是 1，不是 2

    for i in range(1, n):
        A[i] = A[i] / B[i - 1]
        B[i] = B[i] - A[i] * C[i - 1]
        g[i] = g[i] - A[i] * g[i - 1]

    M = np.zeros(n)
    M[-1] = g[-1] / B[-1]

    for i in range(n - 2, -1, -1):
        M[i] = (g[i] - C[i] * M[i + 1]) / B[i]

    return M


def threesimple(X, Y):
    n = len(X)
    h = np.zeros(n - 1)
    g = np.zeros(n)

    for i in range(n - 1):
        h[i] = X[i + 1] - X[i]

    A = Y.copy()
    D = np.zeros((n, 2))

    for i in range(1, n - 1):
        D[i, 0] = h[i - 1] / (h[i - 1] + h[i])
        D[i, 1] = h[i] / (h[i - 1] + h[i])
        g[i] = 6 * (Y[i + 1] - Y[i]) / (h[i] * (h[i - 1] + h[i])) - \
               6 * (Y[i] - Y[i - 1]) / (h[i - 1] * (h[i - 1] + h[i]))

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
        for i in range(n - 1):
            if X[i] <= x[t] <= X[i + 1]:
                p1 = M[i] * (X[i + 1] - x[t]) ** 3 / (6 * h[i])
                p2 = M[i + 1] * (x[t] - X[i]) ** 3 / (6 * h[i])
                p3 = (A[i] - M[i] / 6 * h[i] ** 2) * (X[i + 1] - x[t]) / h[i]
                p4 = (A[i + 1] - M[i + 1] / 6 * h[i] ** 2) * (x[t] - X[i]) / h[i]
                s[t] = p1 + p2 + p3 + p4
                break
            else:
                s[t] = 0
    return s


# ===================== 加载数据 =====================
data = scipy.io.loadmat(
    r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat")
neL = data['nel']

# ===================== 时间轴 =====================
Tinterval = 2.49999985157956e-08
time_axis = np.arange(0, 200004) * Tinterval * 1000  # 转 ms
k = np.argmax(time_axis >= 0.58)  # 找到第一个 ≥0.58ms 的点

# ===================== 输入参数 =====================
X = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669, 0.0831, 0.0993, 0.1155, 0.1318, 0.15])
Y = np.array([
    neL[k, 0], neL[k, 1], neL[k, 2], neL[k, 3], neL[k, 4],
    neL[k, 5], neL[k, 6], neL[k, 7], neL[k, 8], 0.0
])

dr = 0.002
r = np.arange(0, 0.15 + dr, dr)

# ===================== 计时开始 =====================
start_time = time.time()

# 三次样条插值
s = threesimple1(X, Y, r)

# Abel 反演主计算
f = np.zeros_like(r)
for i in range(len(r)):
    d = 0.0
    for j in range(i, len(r) - 1):
        numerator = s[j + 1] - s[j]
        denominator = np.sqrt((r[j] + dr) ** 2 - r[i] ** 2)
        e = (-1.0 / np.pi) * numerator / denominator
        d += e
    f[i] = d

# 计时结束
time_cost = time.time() - start_time
print(f'Abel 反演耗时：{time_cost:.6f} 秒')

# 密度非负
f[f < 0] = 0

# ===================== 画图 =====================
plt.figure(figsize=(9, 6))

plt.subplot(2, 1, 1)
plt.plot(X, Y, 'ro-', linewidth=2, markersize=6)
plt.grid(True)
plt.title('0.58 ms 弦积分信号', fontsize=12)
plt.xlabel('位置 m')
plt.ylabel(r'$n_e L$')

plt.subplot(2, 1, 2)
plt.plot(r, f, 'b', linewidth=2.5)
plt.grid(True)
plt.title(f'Abel 反演密度 | 耗时：{time_cost:.6f} s', fontsize=12)
plt.xlabel('半径 m')
plt.ylabel(r'$n_e(r)$')

plt.tight_layout()
plt.show()