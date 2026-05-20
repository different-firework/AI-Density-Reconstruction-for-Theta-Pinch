import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 基本设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 加载数据 =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
y = data["output"].T.astype(np.float32)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# 取一部分测试集算MIV
X_sample = X_test[:500]  # 取500个样本足够稳定

# ===================== 加载训练好的最优模型 =====================
class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 21), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = BPNN().to(device)
model.load_state_dict(torch.load("BPNN_Final_25dB.pth"))
model.eval()

# ===================== MIV 核心算法 =====================
def calculate_miv(model, X, n_channels=9):
    miv = np.zeros(n_channels)
    X_tensor = torch.tensor(X, device=device)

    for ch in range(n_channels):
        # 构造 +10% 和 -10% 扰动
        X_plus = X_tensor.clone()
        X_plus[:, ch] *= 1.1

        X_minus = X_tensor.clone()
        X_minus[:, ch] *= 0.9

        # 模型预测
        with torch.no_grad():
            y_plus = model(X_plus).cpu().numpy()
            y_minus = model(X_minus).cpu().numpy()

        # 影响 = 输出差异的平均值
        diff = y_plus - y_minus
        miv[ch] = np.mean(np.abs(diff))  # 取绝对值平均影响

    # 归一化到 0~100 方便看
    miv = miv / np.max(miv) * 100
    return miv

# ===================== 计算并输出 =====================
miv_result = calculate_miv(model, X_sample)
channel_names = [f"通道{i+1}" for i in range(9)]

print("="*60)
print("          9个输入通道 MIV 敏感性排序（越大越重要）")
print("="*60)
for i in np.argsort(-miv_result):
    print(f"{channel_names[i]} : {miv_result[i]:>6.2f} 分")

# ===================== 画柱状图 =====================
plt.figure(figsize=(10,5))
bars = plt.bar(channel_names, miv_result, color='#5b9dff')
plt.bar_label(bars, fmt='%.1f', fontsize=10)
plt.title('BPNN 输入通道敏感性分析（MIV方法）', fontsize=14)
plt.ylabel('MIV 相对影响值', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("MIV通道敏感性.png", dpi=300)
plt.show()