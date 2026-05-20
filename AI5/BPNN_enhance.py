import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 开启 CUDA 加速 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

# ===================== BPNN 模型（带Dropout + 迁移到CUDA）=====================
class BPNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(BPNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 48),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(24, 21),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# ===================== 数据集 =====================
class EnhancedDataset(Dataset):
    def __init__(self, sim_input, sim_output, real_data_folder, Tinterval=2.49999985157956e-08, noise_db=25):
        self.sim_input = sim_input.T
        self.sim_output = sim_output.T
        self.noise_db = noise_db

        self.real_profiles = []
        real_files = ["1228008.mat", "1228009.mat", "1228011.mat"]
        for file in real_files:
            path = os.path.join(real_data_folder, file)
            data = scipy.io.loadmat(path)
            nel = data['nel']
            time_axis = np.arange(nel.shape[0]) * Tinterval * 1000
            valid_idx = np.where((time_axis >= 0.5) & (time_axis <= 1.0))[0]
            self.real_profiles.append(nel[valid_idx, :])
        self.real_profiles = np.vstack(self.real_profiles)

    def add_noise(self, signal, snr_db):
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise

    def add_baseline_drift(self, signal):
        drift = np.linspace(-0.05, 0.05, 9) * np.random.uniform(0.5, 1.5)
        return signal + drift

    def normalize_per_sample(self, x):
        max_val = np.max(x)
        if max_val > 1e-8:
            return x / max_val
        return x

    def __len__(self):
        return len(self.sim_input)

    def __getitem__(self, idx):
        x = self.sim_input[idx].copy()
        y = self.sim_output[idx].copy()

        if random.random() < 0.5 and len(self.real_profiles) > 0:
            real_idx = random.randint(0, len(self.real_profiles)-1)
            real_x = self.real_profiles[real_idx].copy()
            scale = np.random.uniform(0.5, 1.5)
            x = real_x * scale

        x = self.add_noise(x, self.noise_db)
        x = self.add_baseline_drift(x)

        # 去掉警告的写法
        for i in range(9):
            if random.random() < 0.3:
                x[i] += np.random.normal(0, 0.02)  # 这里修复了警告

        x = self.normalize_per_sample(x)
        y = self.normalize_per_sample(y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ===================== 加载数据 =====================
sim_data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat"
real_data_folder = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分"

sim_data = scipy.io.loadmat(sim_data_path)
sim_input = sim_data['input']
sim_output = sim_data['output']

dataset = EnhancedDataset(sim_input, sim_output, real_data_folder)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

# ===================== 模型放入 CUDA =====================
model = BPNN(dropout_rate=0.1).to(device)  # 这里用CUDA
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

patience = 10
best_loss = float('inf')
counter = 0
model_save_path = "BPNN_Enhanced_CUDA.pth"

# ===================== 训练 =====================
print("开始 CUDA 加速训练...")
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        # 数据也放入 CUDA
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ 最佳模型已保存 | Loss: {best_loss:.6f}")
    else:
        counter += 1
        if counter >= patience:
            print(f"⏹️ 早停触发")
            break

print("🎉 训练完成！模型：BPNN_Enhanced_CUDA.pth")

plt.figure(figsize=(8,4))
plt.plot(train_losses, linewidth=2)
plt.title("训练损失曲线")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()