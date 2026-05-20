import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 固定设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# 最终最优结构
H1, H2 = 48, 24

# 所有模型（训练好的）
model_files = [
    ("Clean", "BPNN_Clean_48-24.pth"),
    ("40dB", "BPNN_Final_40dB.pth"),
    ("35dB", "BPNN_Final_35dB.pth"),
    ("30dB", "BPNN_Final_30dB.pth"),
    ("25dB", "BPNN_Final_25dB.pth"),
    ("20dB", "BPNN_Final_20dB.pth"),
]

# 鲁棒性测试噪声顺序（和你论文图完全一致）
test_noise_names = ['Clean', '40dB', '35dB', '30dB', '25dB', '20dB']
test_dB_values = [100, 40, 35, 30, 25, 20]  # 100 = 无噪声

# ===================== 数据集划分（仅用测试集！） =====================
data = sio.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
X = data["input"].T.astype(np.float32)
y = data["output"].T.astype(np.float32)

# 8:2 划分，只使用测试集测试
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, device=device)
y_test_np = y_test

# ===================== BPNN 模型 =====================


class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU(),
            nn.Linear(H2, 21), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# ===================== 加噪声 =====================
def add_noise(signal, dB):
    if dB >= 100:
        return signal.clone()
    sig_pow = torch.mean(signal ** 2)
    noise_pow = sig_pow / (10 ** (dB / 10.0))
    noise = torch.randn_like(signal) * torch.sqrt(noise_pow)
    return signal + noise


# ===================== KL 散度 =====================
def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.mean(np.sum(p * np.log(p / q), axis=1))


# ===================== 鲁棒性测试主程序 =====================
print("=" * 70)
print("                模型鲁棒性测试（KL 越小越好）")
print("=" * 70)

kl_matrix = []
model_names = [m[0] for m in model_files]

# 逐个加载模型测试
for model_name, model_path in model_files:
    model = BPNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    kl_row = []
    print(f"\n【模型】{model_name}")

    with torch.no_grad():
        for dB in test_dB_values:
            x_noisy = add_noise(X_test_tensor, dB)
            y_pred = model(x_noisy).cpu().numpy()
            kl = kl_divergence(y_test_np, y_pred)
            kl_row.append(kl)
            print(f"  测试噪声 {dB if dB < 100 else 'Clean'}dB → KL = {kl:.4f}")

    kl_matrix.append(kl_row)

# ===================== 自动找最鲁棒的模型 =====================
mean_kl_all = [np.mean(row) for row in kl_matrix]
best_idx = np.argmin(mean_kl_all)
best_model = model_names[best_idx]

print("\n" + "=" * 70)
print(f"🏆 综合鲁棒性最优模型：【{best_model}】")
print(f"📉 平均 KL：{mean_kl_all[best_idx]:.4f}")
print("=" * 70)

# ===================== 画图（论文级别） =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(11, 6))
colors = ['k', 'r', 'g', 'b', 'orange', 'purple']
linestyles = ['-', '-', '-', '-', '-', '-']
markers = ['o', 's', '^', 'D', '*', 'v']

for i, name in enumerate(model_names):
    plt.plot(
        test_noise_names, kl_matrix[i],
        label=f" {name}",
        color=colors[i], marker=markers[i],
        linewidth=2.5, markersize=7
    )

plt.xlabel("测试噪声强度", fontsize=14)
plt.ylabel("KL 散度", fontsize=14)
plt.title("不同噪声下模型的鲁棒性测试", fontsize=16)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Robustness_Test_Result.png", dpi=300)
plt.show()