import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset

# ===================== 设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]


# =============================================================================
# Transformer + Attention 模型
# =============================================================================
class TransformerAttentionModel(nn.Module):
    def __init__(self, input_dim=9, output_dim=21, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            activation="relu", batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)


# =============================================================================
# ✅ 修复版训练：用 DataLoader 小批量，不爆显存
# =============================================================================
def train_model():
    print("正在加载训练数据...")
    mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")

    X = mat["input"].T  # [200000,9]
    Y = mat["output"].T  # [200000,21]

    # 小批量加载，GPU 绝不会卡
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    model = TransformerAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("开始训练 (3050Ti 约10秒内完成)...")
    model.train()
    for epoch in range(200):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1:2d} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "Transformer_Attention.pth")
    print("✅ 训练完成！模型已保存")


# =============================================================================
# 反演
# =============================================================================
def infer_single():
    model = TransformerAttentionModel().to(device)
    model.load_state_dict(torch.load("Transformer_Attention.pth", map_location=device))
    model.eval()

    data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
    mat = scipy.io.loadmat(data_path)
    nel = mat["nel"]

    Tinterval = 2.49999985157956e-08
    time_axis = np.arange(nel.shape[0]) * Tinterval * 1000
    idx = np.argmax(time_axis >= 0.56)
    x_raw = nel[idx]

    x_norm = x_raw / np.max(x_raw)

    start = time.time()
    with torch.no_grad():
        profile = model(torch.tensor(x_norm, dtype=torch.float32).to(device)).cpu().numpy()
    infer_time = time.time() - start

    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(9, 4))
    plt.plot(r, profile, 'b-', linewidth=3)
    plt.grid(True)
    plt.xlabel("半径 m")
    plt.ylabel("归一化密度")
    plt.title(f"Transformer+Attention 反演 | 耗时：{infer_time:.6f} s")
    plt.show()
    print(f"推理耗时：{infer_time:.6f} s")


# =============================================================================
# 运行
# =============================================================================
if __name__ == "__main__":
    #train_model()
    infer_single()