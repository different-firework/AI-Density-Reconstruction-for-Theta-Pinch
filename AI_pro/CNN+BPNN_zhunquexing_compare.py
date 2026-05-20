import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import scipy.io
import warnings

warnings.filterwarnings("ignore")

# ===================== 基础配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 3个模型结构 =====================
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


# ===================== 数据集 =====================
class DensityDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
        self.X = torch.tensor(mat["input"].T, dtype=torch.float32)
        self.Y = torch.tensor(mat["output"].T, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# ===================== 加载3个模型 =====================
def load_all_models():
    models = {
        "小模型": CNN_BPNN_Small().to(device),
        "中模型": CNN_BPNN_Medium().to(device),
        "大模型": CNN_BPNN_Large().to(device)
    }
    models["小模型"].load_state_dict(torch.load("小模型_sigmoid.pth", map_location=device))
    models["中模型"].load_state_dict(torch.load("中模型_sigmoid.pth", map_location=device))
    models["大模型"].load_state_dict(torch.load("大模型_sigmoid.pth", map_location=device))

    for m in models.values():
        m.eval()
    return models


# ===================== 测试集准确性测试 =====================
def test_accuracy():
    # 加载数据和模型
    dataset = DensityDataset()
    test_size = int(0.2 * len(dataset))
    _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
    models = load_all_models()

    # 随机取20个测试样本
    np.random.seed(42)
    test_indices = np.random.choice(len(test_dataset), 20, replace=False)
    r = np.linspace(0, 0.15, 21)

    # 量化指标统计
    metrics = {name: {"MAE": [], "RMSE": [], "峰值误差": []} for name in models.keys()}

    print("=" * 60)
    print("开始测试集准确性评估")
    print("=" * 60)

    # 逐个样本预测
    for idx in test_indices:
        x, y_true = test_dataset[idx]
        x_tensor = x.unsqueeze(0).to(device)

        with torch.no_grad():
            for name, model in models.items():
                y_pred = model(x_tensor).cpu().numpy()[0]
                # 计算量化指标
                mae = np.mean(np.abs(y_pred - y_true.numpy()))
                rmse = np.sqrt(np.mean((y_pred - y_true.numpy()) ** 2))
                peak_err = np.abs(np.max(y_pred) - np.max(y_true.numpy()))
                # 存入统计
                metrics[name]["MAE"].append(mae)
                metrics[name]["RMSE"].append(rmse)
                metrics[name]["峰值误差"].append(peak_err)

    # 输出平均量化结果
    print("\n📊 20个测试样本平均量化指标（越小越好）：")
    for name in models.keys():
        avg_mae = np.mean(metrics[name]["MAE"])
        avg_rmse = np.mean(metrics[name]["RMSE"])
        avg_peak = np.mean(metrics[name]["峰值误差"])
        print(f"{name}: MAE={avg_mae:.6f} | RMSE={avg_rmse:.6f} | 峰值误差={avg_peak:.6f}")

    # 画1个样本的对比图
    plot_idx = test_indices[0]
    x_plot, y_plot_true = test_dataset[plot_idx]
    x_plot_tensor = x_plot.unsqueeze(0).to(device)

    plt.figure(figsize=(10, 6))
    plt.plot(r, y_plot_true.numpy(), 'r-o', lw=3, label="真实剖面")
    with torch.no_grad():
        plt.plot(r, models["小模型"](x_plot_tensor).cpu().numpy()[0], 'b--', lw=2, label="小模型预测")
        plt.plot(r, models["中模型"](x_plot_tensor).cpu().numpy()[0], 'g--', lw=2, label="中模型预测")
        plt.plot(r, models["大模型"](x_plot_tensor).cpu().numpy()[0], 'm--', lw=2, label="大模型预测")
    plt.xlabel("径向位置 r (m)")
    plt.ylabel("归一化电子密度")
    plt.title(f"测试集样本 #{plot_idx} 反演结果对比")
    plt.legend()
    plt.grid(True)
    plt.show()

    return metrics


# ===================== 运行 =====================
if __name__ == "__main__":
    test_accuracy()