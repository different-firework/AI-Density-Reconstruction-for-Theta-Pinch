import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 自动检测CUDA，优先用GPU =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前使用设备：{device}")

# ===================== BPNN 模型 =====================
class BPNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9, 48), torch.nn.ReLU(),
            torch.nn.Linear(48, 24), torch.nn.ReLU(),
            torch.nn.Linear(24, 21), torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# ===================== CNN+BPNN 中模型 =====================
class CNN_BPNN_Medium(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 3, padding=1), torch.nn.ReLU()
        )
        self.bpnn = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*9, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 21)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)

# ===================== 数据集 =====================
mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = mat["input"].T.astype(np.float32)
y = mat["output"].T.astype(np.float32)

# 数据直接搬到GPU上，避免每次推理都拷贝
X_test = torch.tensor(X[int(0.8*len(X)):], device=device)
y_test = torch.tensor(y[int(0.8*len(y)):], device=device)

# ===================== 加载模型并搬到GPU =====================
bpnn = BPNN().to(device)
bpnn.load_state_dict(torch.load("BPNN_Final_25dB.pth", map_location=device))
bpnn.eval()

cnn_bpnn = CNN_BPNN_Medium().to(device)
cnn_bpnn.load_state_dict(torch.load("中模型_带25dB噪声.pth", map_location=device))
cnn_bpnn.eval()

# ===================== 精度测试（GPU版，不影响结果） =====================
def compute_metrics(model):
    mse_sum = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0
    n = len(X_test)
    with torch.no_grad():
        for i in range(n):
            pred = model(X_test[i:i+1])[0].cpu().numpy()
            true = y_test[i].cpu().numpy()
            mse_sum += np.mean((pred - true)**2)
            mae_sum += np.mean(np.abs(pred - true))
            rmse_sum += np.sqrt(np.mean((pred - true)**2))
    return mse_sum/n, mae_sum/n, rmse_sum/n

mse_bp, mae_bp, rmse_bp = compute_metrics(bpnn)
mse_cnn, mae_cnn, rmse_cnn = compute_metrics(cnn_bpnn)

# ===================== 速度测试：CUDA全速版（解决慢的问题） =====================
def speed_step_gpu(model):
    step = 100
    total = 1000
    speeds = []
    # 1. 先做warm-up，消除GPU初始化开销
    warmup_input = X_test[0:1]
    with torch.no_grad():
        for _ in range(20):
            model(warmup_input)
    # 2. 正式测速，用CUDA事件精准计时，避免Python循环误差
    for _ in range(total // step):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            for __ in range(step):
                model(X_test[0:1])
            end.record()
            torch.cuda.synchronize()  # 等待GPU全部完成，计时才准
        avg_time = start.elapsed_time(end) / step  # 直接得到单样本ms
        speeds.append(avg_time)
    return speeds

speed_bp = speed_step_gpu(bpnn)
speed_cnn = speed_step_gpu(cnn_bpnn)

# ===================== 1. 单样本反演图 =====================
def plot1():
    idx = np.random.randint(len(X_test))
    with torch.no_grad():
        yt = y_test[idx].cpu().numpy()
        y1 = bpnn(X_test[idx:idx+1])[0].cpu().numpy()
        y2 = cnn_bpnn(X_test[idx:idx+1])[0].cpu().numpy()
    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(10,4))
    plt.plot(r, yt, 'k-o', lw=3, label='真实')
    plt.plot(r, y1, 'r--', lw=2, label='BPNN')
    plt.plot(r, y2, 'b-', lw=2, label='CNN+BPNN')
    plt.xlabel('径向位置 (m)')
    plt.ylabel('归一化电子密度')
    plt.title('测试集样本反演对比')
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 2. 误差柱状图（MAE） =====================
def plot2_error_bar():
    names = ['BPNN', 'CNN+BPNN']
    maes = [mae_bp, mae_cnn]
    plt.figure(figsize=(6,4))
    plt.bar(names, maes, color=['red','royalblue'], alpha=0.8)
    plt.ylabel('平均绝对误差 MAE')
    plt.title('测试集整体误差对比（越低越好）')
    for i, v in enumerate(maes):
        plt.text(i, v+0.0002, f'{v:.4f}', ha='center', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# ===================== 3. 速度折线图（每100个平均） =====================
def plot3_speed():
    x = np.arange(100, 1100, 100)
    plt.figure(figsize=(10,4))
    plt.plot(x, speed_bp, 'ro-', lw=2, label='BPNN')
    plt.plot(x, speed_cnn, 'bo-', lw=2, label='CNN+BPNN')
    plt.xlabel('样本数量')
    plt.ylabel('单样本推理时间 (ms)')
    plt.title('每100个数据平均反演速度对比')
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 输出结果 =====================
print("="*60)
print("BPNN 25dB | MSE:%.6f MAE:%.6f RMSE:%.6f" % (mse_bp, mae_bp, rmse_bp))
print("CNN+BPNN | MSE:%.6f MAE:%.6f RMSE:%.6f" % (mse_cnn, mae_cnn, rmse_cnn))
print("="*60)
print(f"BPNN 平均单样本时间：{np.mean(speed_bp):.4f} ms")
print(f"CNN+BPNN 平均单样本时间：{np.mean(speed_cnn):.4f} ms")
print("="*60)

plot1()
plot2_error_bar()
plot3_speed()