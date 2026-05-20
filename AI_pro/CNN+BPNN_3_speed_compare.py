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

# ===================== 3个模型结构（必须和训练一致） =====================
class CNN_BPNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1,8,3,padding=1),nn.ReLU(),nn.Conv1d(8,16,3,padding=1),nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(),nn.Linear(16*9,64),nn.ReLU(),nn.Linear(64,21))
    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))

class CNN_BPNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,32,3,padding=1),nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(),nn.Linear(32*9,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,21))
    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))

class CNN_BPNN_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU())
        self.bpnn = nn.Sequential(nn.Flatten(),nn.Linear(64*9,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,21))
    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.bpnn(self.cnn(x)))

# ===================== 数据集 =====================
class DensityDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat(r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat")
        self.X = torch.tensor(mat["input"].T, dtype=torch.float32)
        self.Y = torch.tensor(mat["output"].T, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ===================== 测速函数 =====================
def speed_one_model(model_path, model_class, model_name):
    # 加载模型
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 取测试集
    dataset = DensityDataset()
    test_size = int(0.2 * len(dataset))
    _, test_dataset = random_split(dataset, [len(dataset)-test_size, test_size])

    # 随机取1000个样本
    np.random.seed(0)
    indices = np.random.choice(len(test_dataset), 1000, replace=False)
    samples = [test_dataset[i][0].to(device) for i in indices]

    print(f"\n========================================")
    print(f"正在测试：{model_name}")
    print(f"========================================")

    total_time = 0
    for i in range(0, 1000, 100):
        group = samples[i:i+100]
        t0 = time.time()
        with torch.no_grad():
            for x in group:
                model(x.unsqueeze(0))
        t1 = time.time() - t0
        avg = t1 / 100
        total_time += t1
        print(f"第{i//100+1}组 | 总耗时 {t1:.4f}s | 单个 {avg:.6f}s")

    avg_all = total_time / 1000
    print(f"\n✅ {model_name} 汇总：")
    print(f"  1000个总时间：{total_time:.4f}s")
    print(f"  单样本平均：{avg_all:.6f}s")
    return avg_all

# ===================== 一次性测3个模型 =====================
if __name__ == "__main__":
    print("🚀 开始测试 3 个模型反演速度！")

    t1 = speed_one_model("小模型_sigmoid.pth", CNN_BPNN_Small, "小模型")
    t2 = speed_one_model("中模型_sigmoid.pth", CNN_BPNN_Medium, "中模型")
    t3 = speed_one_model("大模型_sigmoid.pth", CNN_BPNN_Large, "大模型")

    print("\n" + "="*60)
    print("📊 三个模型最终速度对比（单位：秒/样本）")
    print(f"小模型：{t1:.6f}")
    print(f"中模型：{t2:.6f}")
    print(f"大模型：{t3:.6f}")
    print("="*60)