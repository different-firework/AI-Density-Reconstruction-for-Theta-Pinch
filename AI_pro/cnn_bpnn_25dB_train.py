import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
warnings.filterwarnings("ignore")

# ===================== 配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 数据集 =====================
class DensityDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
        self.X = torch.tensor(mat["input"].T, dtype=torch.float32)
        self.Y = torch.tensor(mat["output"].T, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ===================== 中模型（最优） =====================
class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.bpnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*9, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bpnn(x)
        return torch.sigmoid(x)  # 输出 0~1

# ===================== 25dB 高斯白噪声函数 =====================
def add_white_noise(x, snr_db=25):
    # x: 输入信号 [batch, 9]
    with torch.no_grad():
        batch_size, dim = x.shape
        x_power = torch.mean(x ** 2, dim=1, keepdim=True)
        noise_power = x_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

# ===================== 训练（带噪声 + 分训练测试集） =====================
def train_with_noise():
    dataset = DensityDataset()
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = CNN_BPNN_Medium().to(device)
    opt = optim.Adam(model.parameters(), lr=4e-4)
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []
    epochs = 50

    print("===== 开始训练：中模型 + 25dB 噪声 + 8:2划分 =====")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # ===================== 训练时加噪声！=====================
            x_noisy = add_white_noise(x, snr_db=25)
            pred = model(x_noisy)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_ds)

        # 测试集（不加噪声）
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

    torch.save(model.state_dict(), "中模型_带25dB噪声.pth")
    print("✅ 模型训练完成！已保存：中模型_带25dB噪声.pth")

    # 画图
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="训练 Loss")
    plt.plot(test_losses, label="测试 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("中模型 + 25dB噪声训练曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# ===================== 运行 =====================
if __name__ == "__main__":
    train_with_noise()