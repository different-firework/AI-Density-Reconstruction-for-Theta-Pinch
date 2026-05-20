import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# 字体+负号修复
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 1. 定义CNN模型（和训练时完全一致） =====================
class DensityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 9, 64)
        self.fc2 = nn.Linear(64, 21)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ===================== 2. 定义Transformer模型（和训练时完全一致） =====================
class PureTransformer(nn.Module):
    def __init__(self, input_dim=9, output_dim=21, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# ===================== 3. 加载已训练好的权重 =====================
def load_models():
    cnn_model = DensityCNN().to(device)
    cnn_model.load_state_dict(torch.load("CNN_Final.pth", map_location=device))
    cnn_model.eval()

    trans_model = PureTransformer().to(device)
    trans_model.load_state_dict(torch.load("PureTransformer.pth", map_location=device))
    trans_model.eval()
    return cnn_model, trans_model

# ===================== 4. 读取训练数据集，抽取2组测试样本 =====================
def get_test_samples():
    mat_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat"
    data = scipy.io.loadmat(mat_path)
    all_x = data["input"].T   # (N,9)
    all_y = data["output"].T  # (N,21)

    # 任选两组不同样本做测试
    idx1 = 5005
    idx2 = 15000
    x1, y1 = all_x[idx1], all_y[idx1]
    x2, y2 = all_x[idx2], all_y[idx2]
    return (x1,y1), (x2,y2)

# ===================== 5. 单样本推理 =====================
def predict_one(model_cnn, model_trans, x_np):
    x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_cnn = model_cnn(x_tensor).squeeze(0).cpu().numpy()
        pred_trans = model_trans(x_tensor).squeeze(0).cpu().numpy()
    return pred_cnn, pred_trans

# ===================== 6. 绘图对比 =====================
def plot_compare(sample_id, true_y, cnn_y, trans_y):
    r = np.linspace(0, 0.15, 21)
    plt.figure(figsize=(10,5))
    plt.plot(r, true_y, 'r-o', linewidth=2, label='真实剖面')
    plt.plot(r, cnn_y, 'g--', linewidth=2, label='CNN预测')
    plt.plot(r, trans_y, 'b-.', linewidth=2, label='Transformer预测')
    plt.xlabel("径向位置 r (m)")
    plt.ylabel("归一化电子密度")
    plt.title(f"测试样本{sample_id} 反演结果对比")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 主运行 =====================
if __name__ == "__main__":
    cnn_net, trans_net = load_models()
    sample1, sample2 = get_test_samples()

    # 第一组样本
    x1, y1_true = sample1
    cnn1, trans1 = predict_one(cnn_net, trans_net, x1)
    plot_compare(1, y1_true, cnn1, trans1)

    # 第二组样本
    x2, y2_true = sample2
    cnn2, trans2 = predict_one(cnn_net, trans_net, x2)
    plot_compare(2, y2_true, cnn2, trans2)