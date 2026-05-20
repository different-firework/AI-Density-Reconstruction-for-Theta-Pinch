import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import scipy.io
import time
import warnings

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 模型（必须和训练时一样） =====================
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


# ===================== 数据集 =====================
class DensityDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
        self.X = torch.tensor(mat["input"].T, dtype=torch.float32)
        self.Y = torch.tensor(mat["output"].T, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# ===================== 测速主函数 =====================
def speed_test():
    # 1. 加载模型
    model = CNN_BPNN().to(device)
    model.load_state_dict(torch.load("CNN_BPNN_FINAL.pth", map_location=device))
    model.eval()

    # 2. 划分测试集（20%）
    dataset = DensityDataset()
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    # 3. 随机取 1000 个测试样本
    np.random.seed(0)
    indices = np.random.choice(len(test_dataset), 1000, replace=False)

    # 准备输入
    samples = []
    for idx in indices:
        x, _ = test_dataset[idx]
        samples.append(x.to(device))

    print("=" * 60)
    print("开始测试：1000 个测试样本，每 100 个计算平均推理时间")
    print("=" * 60)

    total_time = 0
    group_times = []

    # 每 100 个一组
    for i in range(0, 1000, 100):
        group = samples[i:i + 100]
        t0 = time.time()

        with torch.no_grad():
            for x in group:
                model(x.unsqueeze(0))

        t1 = time.time() - t0
        avg = t1 / 100
        group_times.append(avg)
        total_time += t1

        print(f"第 {i // 100 + 1} 组 (100个) | 总耗时 {t1:.4f}s | 平均每个 {avg:.6f}s")

    print("=" * 60)
    print(f"✅ 1000个样本总推理时间：{total_time:.4f}s")
    print(f"✅ 单个样本平均推理时间：{total_time / 1000:.6f}s")
    print("=" * 60)


# ===================== 运行 =====================
if __name__ == "__main__":
    speed_test()