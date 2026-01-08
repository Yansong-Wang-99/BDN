import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['svg.fonttype'] = 'none'  # 保留文本为 <text> 元素
mpl.rcParams['font.sans-serif'] = ['Arial']  # 确保 Illustrator 可识别字体


""""
pacman_result_v3 MsPacman-v4
ant_result_v2 Ant-v4
halfcheetah_v2 HalfCheetah-v4
break_v2 Breakout-v4
"""

name = 'D:/xxxx/xxxxx'
env_name = "Ant-v4"
save_name = 'xxxxxxxx'

df = pd.read_excel(f'{name}.xlsx')

steps = df["Step"].values


# 在这里定义你要画的对比实验 (列名列表, 曲线标签, 颜色)
plot_settings = [
    # # pac performance
    # (["snn_da_seed1","snn_da_seed3","snn_da_seed25","snn_da_seed26","snn_da_seed27",], "SNN", "red"),
    # (["ann_v1_seed1","ann_v1_seed2","ann_v1_seed3","ann_v1_seed26","ann_v1_seed27",], "ANN", "green"),
    # (["bdn_v504_3","bdn_v504_26",
    #   "bdn_vbilinear_v504_02_0005_nb_clip1_seed25" ,"bdn_v504_101"], "BDN", "blue"),

    # # # # breakout performance
    # (["snn_seed1","snn_seed2","snn_seed3","snn_seed4","snn_seed5",], "SNN", "red"),
    # (["ann_v1_seed1","ann_v1_seed2","ann_v1_seed4","ann_v1_seed5",], "ANN", "green"),
    # (["bdn_v504_1", "bdn_v504_2", "bdn_v504_100","bdn_v504_102",
    #   "bdn_v504_104",], "BDN", "blue"),

    #halfchee performance
    # (["pop_snn_can24_seed1", "pop_snn_can24_seed2", "pop_snn_can24_seed3", "pop_snn_can24_seed4"], "SNN", "red"),
    # (["ann_v1_seed1","ann_v1_seed2","ann_v1_seed3","ann_v1_seed4","ann_v1_seed5"], "ANN", "green"),
    # (["bdn_v504_1","bdn_v504_2",  "bdn_v504_6","bdn_bilinear_v504_y_b_3"], "BDN", "blue"),

    # # # # ant performance
    (["pop_snn_seed2", "pop_snn_seed3","pop_snn_seed4", "pop_snn_seed6"], "SNN", "red"),
    (["ann_v1_seed1","ann_v1_seed2","ann_v1_seed3","ann_v1_seed5"], "ANN", "green"),
    (["bdn_v504_2","bdn_v504_3","bdn_v504_5",  ], "BDN", "blue"),

]

# ========== 画图部分 ==========
# ---- 将 steps 转换为 0~100 ----
steps_norm = (steps / steps.max()) * 100

plt.figure(figsize=(5,4), dpi=600)

for cols, label, color in plot_settings:
    mean = df[cols].mean(axis=1)
    std = df[cols].std(axis=1)

    plt.plot(steps_norm, mean, color=color, label=label, linewidth=2)
    plt.fill_between(steps_norm, mean-std, mean+std, color=color, alpha=0.2)

# # ====== xlabel 自动设置 ======
if env_name in ["Ant-v4", "HalfCheetah-v4"]:
    plt.xlabel("Training steps (×10k)", fontsize=14)
elif env_name in ["MsPacman-v4", "Breakout-v4"]:
    plt.xlabel("Training steps (×100k)", fontsize=14)
else:
    plt.xlabel("Steps", fontsize=14)
plt.ylabel("Mean reward", fontsize=14)
plt.title(f'{env_name}', fontsize=14)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linestyle="--", alpha=0.6)

# ---- 禁止使用科学计数法 ----
plt.ticklabel_format(style='plain', axis='x')

plt.tight_layout()
# plt.savefig(rf"D:\桌面\result_v1\{save_name}.pdf", format="pdf")
# plt.savefig(rf"D:\桌面\atari_result_v2\svg\{save_name}.svg", format="svg")
#
# # 高质量PNG导出
# plt.savefig(
#     rf"D:\桌面\atari_result_v2\png\{save_name}.png",
#     format="png",
#     dpi=600,            # 关键参数：分辨率（推荐400~600）
#     bbox_inches="tight",# 去掉多余白边
#     pad_inches=0.05,    # 边距微调
#     transparent=False   # 如果想要透明背景可改为 True
# )
plt.show()



