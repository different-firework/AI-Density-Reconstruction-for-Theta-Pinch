import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io
import warnings
warnings.filterwarnings("ignore")

# ===================== 固定配置 =====================
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda else "cpu")

# ===================== 模型结构 =====================
class CNN_BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 9, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 21)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===================== 测试真实数据：固定 0.56ms =====================
def test_real_data_at_056ms():
    # 1. 加载模型
    model = CNN_BPNN().to(device)
    model.load_state_dict(torch.load("CNN_BPNN_FINAL.pth", map_location=device))
    model.eval()
    print("✅ 模型加载成功！")

    # 2. 加载真实数据（可以换成 08 / 09 / 11）
    real_data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"

    real_mat = scipy.io.loadmat(real_data_path)
    nel_data = real_mat["nel"]

    # ===================== 关键：自动定位 0.56ms =====================
    dt = 2.49999985157956e-08  # 时间步长
    time_ms = np.arange(len(nel_data)) * dt * 1000
    target_idx = np.argmin(np.abs(time_ms - 0.56))  # 👈 就这一行！

    # 3. 取 0.56ms 那一行
    single_sample = nel_data[target_idx]

    # 4. 归一化 + 预测
    single_sample = single_sample / np.max(single_sample)
    x_tensor = torch.tensor(single_sample, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_profile = model(x_tensor).cpu().numpy()[0]

    # 5. 画图
    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(10,5))
    plt.plot(r, pred_profile, 'b-o', lw=3, label="CNN+BPNN 反演剖面")
    plt.xlabel("径向位置 r (m)")
    plt.ylabel("归一化电子密度")
    plt.title(f"真实实验数据 0.56ms 时刻反演结果")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 运行 =====================
if __name__ == "__main__":
    test_real_data_at_056ms()