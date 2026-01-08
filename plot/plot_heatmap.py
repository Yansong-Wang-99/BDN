import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# df = pd.read_excel(f'{name}.xlsx')
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # 保留文本为 <text> 元素
mpl.rcParams['font.sans-serif'] = ['Arial']  # 确保 Illustrator 可识别字体


name = 'D:/xxxx/xxxxx'
env_name = "Ant-v4"
save_name = 'xxxxxxxx'

df = pd.read_excel(f'{name}.xlsx')

# # ==== 2. 定义算法 ====

algorithms = ['bdn','ann', 'snn', 'bdn_noburst', 'bdn_nobilinear',
              'bdn_nodecay', 'bdn_nobap', 'bdn_noapical']
# 获取 mean 和 std 数据
mean_data = df[[f'{alg}_mean' for alg in algorithms]]
std_data = df[[f'{alg}_std' for alg in algorithms]]

# ==== 3. 构造两行文字的矩阵 ====
annot_text = pd.DataFrame('', index=mean_data.index, columns=mean_data.columns)

for i in range(len(df)):
    for j, alg in enumerate(algorithms):
        m = mean_data.iloc[i, j]
        s = std_data.iloc[i, j]
        # 均值在上，标准差在下
        annot_text.iloc[i, j] = f"{m:.2f}\n±{s:.2f}"

plt.figure(figsize=(9, 6))
ax = sns.heatmap(
    mean_data,
    cmap='YlOrRd',
    yticklabels=df['noise_level'],
    xticklabels=algorithms,
    cbar=False,          # 关闭自动 colorbar
    square=True
)

# 手动添加 colorbar
cbar = plt.colorbar(ax.collections[0], shrink=0.7)
cbar.set_label('Mean value', fontsize=18)
cbar.ax.tick_params(labelsize=12)

# 替换 X 轴名称
xtick_labels = [
    "BDN-SNN",
    "Relu-ANN",
    "LIF-SNN",
    "Burst\nablated",
    "Bilinear\nablated",
    "Decay\nablated",
    "Bap\nablated",
    "Apical\nablated"
]
ax.set_xticklabels(xtick_labels, rotation=0, fontsize=14)

# ==== 5. 在每个格子内添加两行文字 ====
for i in range(mean_data.shape[0]):      # 遍历行
    for j in range(mean_data.shape[1]):  # 遍历列
        mean_val = mean_data.iloc[i, j]
        std_val = std_data.iloc[i, j]
        # 奖励均值（上面一行，正常粗细）
        ax.text(j + 0.5, i + 0.5, f"{mean_val:.2f}",
                ha='center', va='center',
                fontsize=11, color='black')
        # 标准差（下面一行，靠近奖励一点）
        ax.text(j + 0.5, i + 0.7, f"±{std_val:.2f}",
                ha='center', va='center',
                fontsize=8, color='black')

# ==== 6. 美化 ====
plt.title(f'{env_name}', fontsize=18, pad=15)
plt.ylabel('Noise Level')
plt.tick_params(axis='both', labelsize=10)
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
