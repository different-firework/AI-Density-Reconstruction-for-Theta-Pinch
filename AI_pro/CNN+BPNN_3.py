import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io
import warnings

warnings.filterwarnings("ignore")

# ===================== 基础配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 数据集 =====================
class DensityDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
        self.X = torch.tensor(mat["input"].T, dtype=torch.float32)
        self.Y = torch.tensor(mat["output"].T, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# ===================== 3种模型 全部末尾加Sigmoid =====================
# 小模型 1→8→16
class CNN_BPNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 9, 64), nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)  # 值域约束


# 中模型 1→16→32
class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)


# 大模型 1→32→64
class CNN_BPNN_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 21)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)


# ===================== 统一训练函数 固定50轮 =====================
def train(model, name, epochs=50, lr=4e-4):
    dataset = DensityDataset()
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    print(f"\n===== 开始训练 {name} 共{epochs}轮 =====")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss += loss_fn(model(x), y).item() * len(x)
        test_loss /= len(test_ds)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | Train {train_loss:.6f} | Test {test_loss:.6f}")

    torch.save(model.state_dict(), f"{name}_sigmoid.pth")
    print(f"✅ {name} 训练完成已保存")
    return train_losses, test_losses


# ===================== 批量运行 =====================
if __name__ == "__main__":
    tr1, te1 = train(CNN_BPNN_Small(), "小模型")
    tr2, te2 = train(CNN_BPNN_Medium(), "中模型")
    tr3, te3 = train(CNN_BPNN_Large(), "大模型")

    # 绘制测试集loss对比
    plt.figure(figsize=(12, 6))
    plt.plot(te1, label="小模型", linewidth=2)
    plt.plot(te2, label="中模型", linewidth=2)
    plt.plot(te3, label="大模型", linewidth=2)
    plt.xlabel("迭代轮数")
    plt.ylabel("测试集MSE损失")
    plt.title("三种CNN-BPNN结构测试损失对比(输出加Sigmoid)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 输出最终结果
    print("\n========== 50轮训练最终测试集损失 ==========")
    print(f"小模型：{te1[-1]:.6f}")
    print(f"中模型：{te2[-1]:.6f}")
    print(f"大模型：{te3[-1]:.6f}")