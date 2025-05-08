import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 示例数据创建
P1 = [
    [0.851, 0.779, 0.755],
    [0.914, 0.842, 0.813],
    [0.981,0.952,0.919],
    [0.894,0.846,0.865],
    [0.943,0.866,0.852],
    [0.981,0.957,0.951],
]
aa = np.random.uniform(70, 100, 18)

P1 = [
    0.851, 0.779, 0.755, 0.894,0.846,0.865, 0.914, 0.842, 0.813, 0.943,0.866,0.852, 0.981,0.952,0.919, 0.981,0.957,0.951
]
P2 = [
    0.933,0.909,0.889,0.976,0.971,0.942,0.900,0.914,0.885,0.957,0.952,0.938,0.957,0.947,0.900,0.995,0.981,0.971
]
P1 = np.array(P1) * 100
P2 = np.array(P2) * 100
data = {
    'Model': ['Qwen-14B-Chat', 'Qwen-14B-Chat', 'Qwen-14B-Chat', 'Qwen-32B-Chat-AWQ', 'Qwen-32B-Chat-AWQ', 'Qwen-32B-Chat-AWQ'] * 3,
    'Order': ['Ordered', 'Random', 'Reversed', 'Ordered', 'Random', 'Reversed'] * 3,
    'Performance': P2,
    'Dataset': ['ZH-EN']*6 + ['JA-EN']*6 + ['FR-EN']*6
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置图表大小和布局
fig, axes = plt.subplots(2, 3, figsize=(4.9, 4.6), sharey=True, sharex=True)

# 设置整体图表的标题和标签
# fig.suptitle('Performance Comparison by Model and Dataset', fontsize=16)
fig.text(0.04, 0.5, 'Hits@1', va='center', rotation='vertical', fontsize=14)
# fig.text(0.5, 0.01, 'Dataset', ha='center', fontsize=12)
# fig.text(0.5, 0.98, 'Order of Options: Ordered, Random, Reversed', ha='center', fontsize=12)

# 绘制每个数据集和模型的组合图
orders = ['Ordered', 'Random', 'Reversed']
# 柱子颜色 浅色
colors = ['skyblue', 'lightcoral', 'lightgreen']
# 柱子形状
hatch = ['/', '..', '\\']
# 透明度
alpha = 0.7

num_orders = len(orders)
group_width = 0.5  # 组宽度
bar_width = group_width / num_orders  # 柱子宽度
# 自适应位置，窄距离
positions = np.arange(len(orders)) * bar_width
positions[0] -= bar_width / 3
positions[2] += bar_width / 3

print(positions)
for i, dataset in enumerate(['ZH-EN', 'JA-EN', 'FR-EN']):
    for j, model in enumerate(['Qwen-14B-Chat', 'Qwen-32B-Chat-AWQ']):
        ax = axes[j, i]
        # 过滤特定数据集和模型的数据
        subset_df = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        for index, order in enumerate(orders):
            order_data = subset_df[subset_df['Order'] == order]['Performance']
            print(order_data)
            # 计算柱子的位置
            ax.bar(positions[index], order_data, bar_width, label=order, color=colors[index], alpha=alpha, hatch=hatch[index])


        # ax.set_title(f'{dataset}')
        ax.set_xticks(positions)
        ax.set_xticklabels(["", "", ""])
        ax.set_ylim(70, 100)  # 确保y轴一致
        if i == 0:
            ax.set_ylabel(f'{model}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if j == 1:
            ax.set_xlabel(f'{dataset}', fontsize=11)

# 显示行和列的共用标签
# for ax, dataset in zip(axes[1], ['A', 'B', 'C']):
    # ax.set_title(dataset, loc='center', fontsize=14)

# 添加图例解释
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1.01), ncol=3, frameon=False, fontsize='medium')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # 调整布局留出空间
plt.savefig('position_bias2.pdf', format='pdf')
# plt.show()
