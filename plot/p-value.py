import pandas as pd
from scipy.stats import ttest_rel


name = 'D:/桌面/continue_res/p-value'

# 读取你的文件
df = pd.read_excel(f'{name}.xlsx')

# 基准列
base_col = "bdn_mean"

# 你手动指定要比较的列
# compare_cols = ["snn_mean", "ann_mean",
#                 "bdn_noburst_mean", "bdn_nobilinear_mean",
#                 "bdn_nodecay_mean", "bdn_nobap_mean",
#                 "bdn_noapical_mean"
#                 ]
# compare_cols = ["ant_snn", "ant_ann",
#                 ]
compare_cols = ["snn_mean",
                ]
# 基准列数据
base_values = df[base_col].astype(float)

# 保存结果的列表
results = []

# 逐列执行 paired t-test
for col in compare_cols:
    target_values = df[col].astype(float)

    t, p = ttest_rel(base_values, target_values)

    print(f"Paired t-test: {base_col} vs {col}")
    print(f"    p = {p:.4g}")
    print("-" * 40)
    results.append({"comparison": f"{base_col} vs {col}",  "p_value": p})
