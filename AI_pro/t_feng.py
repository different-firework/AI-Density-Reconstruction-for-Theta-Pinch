import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io
import warnings
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 径向参数：9个通道对应实际半径范围(单位cm)
r_min = 0.0    # 最内侧通道半径
r_max = 15.0   # 最外侧通道半径
channel_num = 9
# 通道序号映射实际半径
chan2r = lambda idx: r_min + idx * (r_max - r_min) / (channel_num - 1)

# 高斯拟合函数
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# ===================== BPNN模型 =====================
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

# ===================== 主函数 =====================
def plot_peak_radius_time():
    real_data_path = r"\Users\86139\Desktop\2026-03\密度剖面反演BPNN\密度剖面反演BPNN\9道干涉仪密度积分\1228011.mat"
    real_mat = scipy.io.loadmat(real_data_path)
    nel_data = real_mat["nel"]

    # 时间轴
    dt = 2.49999985157956e-08
    time_ms = np.arange(len(nel_data)) * dt * 1000

    # 选取时间区间
    t_start, t_end = 0.54, 0.58
    mask = (time_ms >= t_start) & (time_ms <= t_end)
    time_selected = time_ms[mask]
    data_selected = nel_data[mask]

    channels = np.arange(9)
    # 存储两种方法得到的实际半径(cm)
    peak_r_max = []
    peak_r_fit = []

    for frame in data_selected:
        sig = frame.copy()
        sig = sig / (np.max(sig) + 1e-8)

        # 1.最大值法 → 映射为实际半径
        max_chan = np.argmax(sig)
        r1 = chan2r(max_chan)
        peak_r_max.append(r1)

        # 2.高斯拟合法 → 得到浮点通道位置再转半径
        try:
            popt, _ = curve_fit(gaussian, channels, sig, p0=[1, max_chan, 1], maxfev=1000)
            _, mu_chan, _ = popt
            r2 = chan2r(mu_chan)
        except:
            r2 = r1
        peak_r_fit.append(r2)

    # 绘图：Y轴为径向半径cm
    plt.figure(figsize=(10, 5))
    plt.plot(time_selected, peak_r_max, 'ro-', markersize=3, linewidth=1.5, label="最大值法")
    plt.plot(time_selected, peak_r_fit, 'b-', linewidth=2.5, label="高斯拟合法")
    plt.xlabel("时间 (ms)")
    plt.ylabel("径向位置 (cm)")
    plt.title("0.54~0.58ms 等离子体密度峰值径向位置时序变化")
    plt.xlim(t_start, t_end)
    plt.ylim(r_min-0.5, r_max+0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

    # BPNN反演 仅计算不绘图
    model = BPNN().to(device)
    model.load_state_dict(torch.load("BPNN_Final_25dB.pth", map_location=device))
    model.eval()
    tar_idx = np.argmin(np.abs(time_ms - 0.56))
    samp = nel_data[tar_idx] / np.max(nel_data[tar_idx])
    inp = torch.tensor(samp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        res = model(inp).cpu().numpy()[0]
    print(f"0.56ms时刻25dB模型反演密度峰值：{np.max(res):.4f}")

if __name__ == "__main__":
    plot_peak_radius_time()