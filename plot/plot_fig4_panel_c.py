import numpy as np
import matplotlib.pyplot as plt
import os

# ================== 路径配置 ==================
base_dir = r"D:/桌面/continue_res"
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # 保留文本为 <text> 元素
mpl.rcParams['font.sans-serif'] = ['Arial']  # 确保 Illustrator 可识别字体

save_name = "1218_mnist"
bdn_file = os.path.join(base_dir, "mnist_bdn_res.txt")
snn_file = os.path.join(base_dir, "mnist_snn_res.txt")

# save_name = "1218_cifar"
# bdn_file = os.path.join(base_dir, "cifar10_bdn_res.txt")
# snn_file = os.path.join(base_dir, "cifar10_snn_res.txt")

# save_name = "1218_DVS"
# bdn_file = os.path.join(base_dir, "gesture_bdn_res.txt")
# snn_file = os.path.join(base_dir, "gesture_snn_res.txt")

# save_name = "1218_mat"
# bdn_file = os.path.join(base_dir, "mathgreek_bdn_res.txt")
# snn_file = os.path.join(base_dir, "mathgreek_snn_res.txt")

# save_name = "1218_alphabet"
# bdn_file = os.path.join(base_dir, "alphabet_bdn_res.txt")
# snn_file = os.path.join(base_dir, "alphabet_snn_res.txt")
# ================== 读取数据函数 ==================
def load_mean_std(file_path):
    data = np.loadtxt(file_path, comments="#")
    mean = data[:, 0] * 100   # mean 转为百分比
    std  = data[:, 1] * 100   # std 也转为百分比
    return mean, std

bdn_mean_all, bdn_std_all = load_mean_std(bdn_file)
snn_mean_all, snn_std_all = load_mean_std(snn_file)

# ===== 跨 task 的 mean =====
bdn_final_mean = np.mean(bdn_mean_all)
snn_final_mean = np.mean(snn_mean_all)

# ===== 使用“数据里的 std”，再取平均 =====
bdn_final_std  = np.mean(bdn_std_all)
snn_final_std  = np.mean(snn_std_all)
print(f"bdn_final_mean:{bdn_final_std}")
print(f"snn_final_mean:{snn_final_std}")

print(f"bdn_final_std:{bdn_final_std}")
print(f"snn_final_std:{snn_final_std}")


# ================== 绘图 ==================
plt.figure(figsize=(4, 4))

methods = ["BDN", "SNN"]
final_means = [bdn_final_mean, snn_final_mean]
final_stds  = [bdn_final_std,  snn_final_std]

colors = ["blue", "red"]
# x = np.arange(len(methods))
x = np.array([0, 0.55])   # 原来是 [0, 1]

plt.bar(
    x,
    final_means,
    yerr=final_stds,
    capsize=6,
    color=colors,
    alpha=0.8,
    width=0.4
)

plt.xticks(x, methods, fontsize=12)
# plt.ylabel("Accuracy (%)", fontsize=12)

plt.ylim(0, 105)
plt.tight_layout()


# ================== 保存 ==================

# plt.savefig(
#     rf"D:\桌面\atari_result_v2\con_res\{save_name}_bar.svg",
#     format="svg"
# )
#
# plt.savefig(
#     rf"D:\桌面\atari_result_v2\con_res\{save_name}_bar.png",
#     format="png",
#     dpi=600,
#     bbox_inches="tight",
#     pad_inches=0.05,
#     transparent=False
# )

plt.show()
plt.close()

