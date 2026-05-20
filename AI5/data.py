import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 加载训练数据 =====================
data_path = r"C:\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\样本训练数据.mat"
data = scipy.io.loadmat(data_path)

input_data  = data['input']   # shape: [9, 200000]
output_data = data['output']  # shape: [21, 200000]

print("输入数据维度(9道弦积分):", input_data.shape)
print("输出数据维度(21道密度):", output_data.shape)

# ===================== 坐标定义（和你论文一致）=====================
# 9道干涉仪位置
X9 = np.array([0, 0.0182, 0.0345, 0.0507, 0.0669,
               0.0831, 0.0993, 0.1155, 0.1318])
# 21道径向位置
R21 = np.linspace(0, 0.15, 21)

# ===================== 随机抽取样本数量 =====================
sample_num = 4  # 想多看几个改这里就行
total_samples = input_data.shape[1]
rand_idx = random.sample(range(total_samples), sample_num)

# ===================== 逐样本画图 =====================
plt.figure(figsize=(12, 3*sample_num))

for n, idx in enumerate(rand_idx):
    # 取当前样本
    in_9  = input_data[:, idx]
    out_21 = output_data[:, idx]

    # 子图：上半部分画9道输入
    plt.subplot(sample_num, 2, 2*n + 1)
    plt.plot(X9, in_9, 'ro-', linewidth=2, markersize=5)
    plt.grid(True)
    plt.title(f'样本{idx}：9道归一化弦积分输入', fontsize=11)
    plt.xlabel('径向位置 m')
    plt.ylabel('归一化 $n_e L$')
    plt.ylim(-0.05, 1.05)

    # 子图：下半部分画21道真实密度输出
    plt.subplot(sample_num, 2, 2*n + 2)
    plt.plot(R21, out_21, 'b-', linewidth=2.5)
    plt.grid(True)
    plt.title(f'样本{idx}：21道归一化真实密度输出', fontsize=11)
    plt.xlabel('径向位置 m')
    plt.ylabel('归一化 $n_e(r)$')
    plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# 打印其中一个样本的数值，方便你看数据
print(f"\n任选一个样本id={rand_idx[0]} 的9道输入值：")
print(np.round(input_data[:, rand_idx[0]], 4))
print(f"\n对应21道真实输出值：")
print(np.round(output_data[:, rand_idx[0]], 4))