import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import time

# ===================== 基础配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]


# =============================================================================
# 🔥 纯 Transformer 模型（9 → 21）
# =============================================================================
class PureTransformer(nn.Module):
    def __init__(self, input_dim=9, output_dim=21, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # 标准 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=32,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # 输入 x: [batch, 9]
        x = self.input_proj(x).unsqueeze(1)  # [batch, 1, d_model]
        x = self.transformer(x)  # 纯Transformer计算
        x = x.squeeze(1)
        return self.fc(x)


# =============================================================================
# 训练
# =============================================================================
def train():
    print("加载训练数据...")
    mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
    X = torch.tensor(mat["input"].T, dtype=torch.float32).to(device)
    Y = torch.tensor(mat["output"].T, dtype=torch.float32).to(device)

    model = PureTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    print("开始训练 Transformer...")
    model.train()
    for epoch in range(200):
        pred = model(X)
        loss = loss_fn(pred, Y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

    torch.save(model.state_dict(), "PureTransformer.pth")
    print("✅ 纯Transformer训练完成！")


# =============================================================================
# 推理（单时刻 0.56ms）
# =============================================================================
def infer():
    model = PureTransformer().to(device)
    model.load_state_dict(torch.load("PureTransformer.pth", map_location=device))
    model.eval()

    path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
    nel = scipy.io.loadmat(path)["nel"]

    dt = 2.49999985157956e-08
    time_ms = np.arange(nel.shape[0]) * dt * 1000
    idx = np.argmax(time_ms >= 0.58)
    x_raw = nel[idx]
    x_norm = x_raw / np.max(x_raw)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)

    # 推理计时
    start = time.time()
    with torch.no_grad():
        profile = model(x_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
    cost = time.time() - start

    # 绘图
    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(9, 4))
    plt.plot(r, profile, 'purple', linewidth=3, label="Pure Transformer")
    plt.grid(True)
    plt.xlabel("半径 r (m)")
    plt.ylabel("归一化密度")
    plt.title(f"纯Transformer反演 | 耗时：{cost:.6f} s")
    plt.legend()
    plt.show()
    print(f"推理耗时：{cost:.6f} s")


# =============================================================================
# 运行
# =============================================================================
if __name__ == "__main__":
    #train()
    infer()