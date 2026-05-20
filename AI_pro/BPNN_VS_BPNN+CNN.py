import numpy as np
import time
import scipy.io

# -------------------- 用 CPU 推理，避开环境bug --------------------
import torch
torch.manual_seed(42)


# ===================== BPNN 模型（和你完全一致） =====================
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

X_test = torch.tensor(X[int(0.8*len(X)):])
y_test = torch.tensor(y[int(0.8*len(y)):])

# ===================== 加载模型 =====================
bpnn = BPNN()
bpnn.load_state_dict(torch.load("BPNN_Final_25dB.pth", map_location="cpu"))
bpnn.eval()

cnn_bpnn = CNN_BPNN_Medium()
cnn_bpnn.load_state_dict(torch.load("中模型_带25dB噪声.pth", map_location="cpu"))
cnn_bpnn.eval()

# ===================== 精度测试 =====================
def test_acc(model, name):
    mse = 0
    mae = 0
    rmse = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            pred = model(X_test[i:i+1])[0].numpy()
            true = y_test[i].numpy()
            mse += np.mean((pred-true)**2)
            mae += np.mean(np.abs(pred-true))
            rmse += np.sqrt(np.mean((pred-true)**2))
    n = len(X_test)
    print(f"【{name}】")
    print(f"MSE: {mse/n:.6f}")
    print(f"MAE: {mae/n:.6f}")
    print(f"RMSE: {rmse/n:.6f}")

# ===================== 速度测试 =====================
def test_speed(model, name):
    t0 = time.time()
    with torch.no_grad():
        for _ in range(1000):
            model(X_test[0:1])
    t = time.time()-t0
    print(f"【{name}】1000样本耗时: {t:.3f}s  单样本: {t/1000:.6f}s")

# ===================== 运行 =====================
print("="*50)
test_acc(bpnn, "BPNN 25dB")
test_acc(cnn_bpnn, "CNN+BPNN 中模型 25dB")
print("="*50)
test_speed(bpnn, "BPNN")
test_speed(cnn_bpnn, "CNN+BPNN")
print("="*50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")