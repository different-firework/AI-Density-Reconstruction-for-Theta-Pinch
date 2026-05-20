import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from sklearn.model_selection import train_test_split

# ===================== 基础设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

batch_size = 256
epochs = 50
lr = 0.001

# 只训练这 5 个噪声，去掉 0dB 和 Clean
dB_list = [40, 35, 30, 25, 20]

# 综合最优结构
h1, h2 = 48, 24

# ===================== 数据划分 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
y = data["output"].T.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, device=device)
y_train_tensor = torch.tensor(y_train, device=device)

# ===================== BPNN 模型 =====================
class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 21), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# ===================== 加噪声函数 =====================
def add_gaussian_noise_by_dB(signal, dB):
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (dB / 10.0))
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise

# ===================== 训练 5 个噪声模型 =====================
print("===== 训练 48-24 结构 | 噪声：40/35/30/25/20dB =====\n")

for dB in dB_list:
    print(f"\n======================================")
    print(f"             {dB} dB 噪声模型")
    print(f"======================================")

    # 训练集加噪声
    X_noisy = add_gaussian_noise_by_dB(X_train_tensor.clone(), dB)
    dataset = TensorDataset(X_noisy, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BPNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(X_train)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), f"BPNN_Final_{dB}dB.pth")
    print(f"✅ 保存：BPNN_Final_{dB}dB.pth")

print("\n🎉 5 个噪声模型训练完成！")