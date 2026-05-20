import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===================== 基础设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


torch.manual_seed(42)
np.random.seed(42)

batch_size = 256
epochs = 50
lr = 0.001

# 只训练 25dB
target_dB = 25

# 网络结构
h1, h2 = 48, 24

# ===================== 数据划分 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
y = data["output"].T.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转换为 tensor
X_train_tensor = torch.tensor(X_train, device=device)
y_train_tensor = torch.tensor(y_train, device=device)
X_test_tensor = torch.tensor(X_test, device=device)
y_test_tensor = torch.tensor(y_test, device=device)

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

# ===================== 只训练 25dB (25dB.2) =====================
print("===== 训练 48-24 结构 | 噪声：25dB | 保存名：25dB.2 =====\n")
print(f"             25 dB 噪声模型 (25dB.2)")

# 给训练集加噪声
X_noisy_train = add_gaussian_noise_by_dB(X_train_tensor.clone(), target_dB)
train_dataset = TensorDataset(X_noisy_train, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 给测试集加噪声（必须加！和训练一致的噪声水平）
X_noisy_test = add_gaussian_noise_by_dB(X_test_tensor.clone(), target_dB)

# 模型初始化
model = BPNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 记录 loss
train_loss_history = []
test_loss_history = []

# ===================== 训练 + 测试 每轮 =====================
model.train()
for epoch in range(1, epochs+1):
    # -------- 训练阶段 --------
    total_train_loss = 0.0
    for bx, by in train_loader:
        pred = model(bx)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(bx)
    avg_train_loss = total_train_loss / len(X_train)

    # -------- 测试阶段（不更新梯度） --------
    model.eval()
    with torch.no_grad():
        pred_test = model(X_noisy_test)
        avg_test_loss = criterion(pred_test, y_test_tensor).item()
    model.train()

    # 保存loss
    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)

    # 打印
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

# 保存模型
torch.save(model.state_dict(), "BPNN_Final_25dB.2.pth")
print(f"\n✅ 保存：BPNN_Final_25dB.2.pth")

# ===================== 绘制双loss曲线 =====================
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_loss_history, 'b-', linewidth=2, label='训练集 Loss')
plt.plot(range(1, epochs+1), test_loss_history, 'r-', linewidth=2, label='测试集 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('25dB模型 训练/测试 Loss 曲线')
plt.legend()
plt.grid(True)
plt.show()

print("\n🎉 25dB.2 模型训练完成！训练+测试loss曲线已显示！")