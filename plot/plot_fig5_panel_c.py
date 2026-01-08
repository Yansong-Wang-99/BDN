import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # 保留文本为 <text> 元素
mpl.rcParams['font.sans-serif'] = ['Arial']  # 确保 Illustrator 可识别字体


# # ========== 配置部分 ==========

name = 'D:/xxxx/xxxxx'
env_name = "Ant-v4"
save_name = 'xxxxxxxx'

df = pd.read_excel(f'{name}.xlsx')

noise = df["noise_level"].values

plt.figure(figsize=(5,4), dpi=600)

plot_settings = [
    (["snn_mean", "snn_std"], "SNN", "red"),
    # (["ann_mean", "ann_std"], "ANN", "green"),
    (["bdn_mean", "bdn_std"], "BDN", "blue"),
]

plt.figure(figsize=(5,4), dpi=600)

for (mean_col, std_col), label, color in plot_settings:
    mean = df[mean_col].values
    std = df[std_col].values

    plt.plot(noise, mean, color=color, label=label, linewidth=2)
    plt.fill_between(noise, mean - std, mean + std, color=color, alpha=0.2)

plt.xlabel("Noise level", fontsize=14)
plt.ylabel("Mean reward", fontsize=14)
plt.title(f'{env_name}', fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12, frameon=False)

plt.tight_layout()
# plt.savefig(rf"D:\桌面\result_v1\{save_name}.pdf", format="pdf")
plt.savefig(rf"D:\桌面\atari_result_v2\rob_svg\{save_name}.svg", format="svg")

# 高质量PNG导出
plt.savefig(
    rf"D:\桌面\atari_result_v2\rob_png\{save_name}.png",
    format="png",
    dpi=600,            # 关键参数：分辨率（推荐400~600）
    bbox_inches="tight",# 去掉多余白边
    pad_inches=0.05,    # 边距微调
    transparent=False   # 如果想要透明背景可改为 True
)
plt.show()
