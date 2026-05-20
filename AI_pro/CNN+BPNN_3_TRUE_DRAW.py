import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io
import warnings

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 3个模型结构（和训练完全一致） =====================
class CNN_BPNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv1d(8, 16, 3, padding=1), nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(), nn.Linear(16 * 9, 64), nn.ReLU(), nn.Linear(64, 21))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))


class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1), nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(), nn.Linear(32 * 9, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, 21))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))


class CNN_BPNN_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(), nn.Linear(64 * 9, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                  nn.Linear(128, 21))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))


# ===================== 加载全部模型 =====================
def load_all_models():
    models = {}
    models["小模型"] = CNN_BPNN_Small().to(device)
    models["中模型"] = CNN_BPNN_Medium().to(device)
    models["大模型"] = CNN_BPNN_Large().to(device)

    models["小模型"].load_state_dict(torch.load("小模型_sigmoid.pth", map_location=device))
    models["中模型"].load_state_dict(torch.load("中模型_sigmoid.pth", map_location=device))
    models["大模型"].load_state_dict(torch.load("大模型_sigmoid.pth", map_location=device))

    for m in models.values():
        m.eval()
    return models


# ===================== 核心：预测 0.56ms =====================
def predict_056ms_all_models():
    # 加载真实实验数据
    real_data_path = r"\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
    real_mat = scipy.io.loadmat(real_data_path)
    nel_data = real_mat["nel"]

    # 自动找到 0.56ms
    dt = 2.49999985157956e-08
    time_ms = np.arange(len(nel_data)) * dt * 1000
    target_idx = np.argmin(np.abs(time_ms - 0.56))

    # 取样本 + 归一化
    sample = nel_data[target_idx]
    sample = sample / np.max(sample)
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

    # 加载3个模型并预测
    models = load_all_models()
    results = {}
    with torch.no_grad():
        results["小模型"] = models["小模型"](x).cpu().numpy()[0]
        results["中模型"] = models["中模型"](x).cpu().numpy()[0]
        results["大模型"] = models["大模型"](x).cpu().numpy()[0]

    # 画图（三模型对比）
    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(10, 5))

    plt.plot(r, results["小模型"], 'm--', linewidth=2, label="小模型")
    plt.plot(r, results["中模型"], 'b-', linewidth=3, label="中模型(最优)")
    plt.plot(r, results["大模型"], 'g--', linewidth=2, label="大模型")

    plt.xlabel("径向位置 r (m)")
    plt.ylabel("归一化电子密度")
    plt.title("真实实验数据 0.56ms 时刻 | 三模型反演结果对比")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 输出峰值（论文重要指标）
    print("✅ 0.56ms 真实数据反演峰值：")
    print(f"小模型：{np.max(results['小模型']):.4f}")
    print(f"中模型：{np.max(results['中模型']):.4f}")
    print(f"大模型：{np.max(results['大模型']):.4f}")


# ===================== 运行 =====================
if __name__ == "__main__":
    predict_056ms_all_models()