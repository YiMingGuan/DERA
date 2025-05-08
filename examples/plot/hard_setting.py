import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 示例数据
methods = ['baseline', 'JAPE', 'BootEA', 'GCN-Align', 'MuGNN',
           'MultiKE', 'RDGCN', 'AttrGNN', 'FGWEA', 'ETS']
regular_setting_performance_hits_1_zh_en = [
    0.603,
    0.412,
    0.630,
    0.413,
    0.494,
    0.437,
    0.708,
    0.796,
    0.976,
    0.968,
]

regular_setting_performance_hits_10_zh_en = [
    0.710,
    0.745,
    0.848,
    0.744,
    0.844,
    0.516,
    0.846,
    0.929,
    0.994,
    0.994,
]

regular_setting_performance_mrr_zh_en = [
    0.642,
    0.490,
    0.703,
    0.549,
    0.611,
    0.466,
    0.749,
    0.845,
    0.983,
    0.979,
]

hard_setting_performance_hits1_zh_en = [
    0.384,
    0.350,
    0.513,
    0.366,
    0.406,
    0.279,
    0.604,
    0.662,
    0.756,
    0.967
]

hard_setting_performance_hits10_zh_en = [
    0.551,
    0.566,
    0.746,
    0.647,
    0.746,
    0.352,
    0.766,
    0.818,
    0.868,
    0.993
]

hard_setting_performance_mrr_zh_en = [
    0.444,
    0.451,
    0.593,
    0.464,
    0.521,
    0.306,
    0.662,
    0.719,
    0.796,
    0.977
]

# 设置
bar_width = 0.35  # 柱状图的宽度
index = np.arange(len(methods))  # 方法的索引

colors_regular = 'cornflowerblue'
colors_hard = 'salmon'

# 填充样式
patterns = ['/', '.', '|', '-', '+', 'x', '\\', 'O', 'o', '*']

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(index - bar_width/2, regular_setting_performance_mrr_zh_en, bar_width, label='Regular Setting',
               color=colors_regular, edgecolor='black')
bars2 = ax.bar(index + bar_width/2, hard_setting_performance_mrr_zh_en, bar_width, label='Hard Setting',
               color=colors_hard , edgecolor='black')

# 将不同的填充样式应用于每个方法
for bars, pattern in zip([bars1, bars2], patterns*5):
    for bar in bars:
        bar.set_hatch(pattern)

# 添加一些文本标签
# ax.set_xlabel('Methods')
ax.set_ylabel('Hits@1')
# ax.set_title('Performance of different methods under Regular and Hard settings')
ax.set_xticks(index)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()

# 显示图表
# plt.tight_layout()
plt.savefig('regular_hard_zh_en_mrr_small.pdf', format='pdf')