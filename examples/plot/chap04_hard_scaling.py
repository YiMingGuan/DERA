import matplotlib.pyplot as plt

# 模型规模
model_sizes = [7, 14, 32]  # 单位为Billion

# 不同数据集在不同模型规模上的Hits@1指标
hits_at_one_zh_hard = [0.537, 0.837, 0.926]
hits_at_one_ja_hard = [0.497,0.795,0.883]
hits_at_one_fr_hard = [0.597,0.85,0.919]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(4, 5))

# 绘制三条折线
ax.plot(model_sizes, hits_at_one_zh_hard, marker='o', linestyle='-', label='ZH-EN')
ax.plot(model_sizes, hits_at_one_ja_hard, marker='s', linestyle='--', label='JA-EN')
ax.plot(model_sizes, hits_at_one_fr_hard, marker='^', linestyle='-.', label='FR-EN')

# 设置图例
ax.legend(loc='lower right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# 设置横坐标和纵坐标的标签
ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
ax.set_ylabel('Hits@1', fontsize=12)

# # 自动调整横纵坐标轴的范围
ax.set_xscale('log')  # 设置横坐标为对数尺度以反映参数规模的变化
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.autoscale()
ax.set_ylim(0.47, 1.0)
#save
handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='best', bbox_to_anchor=(0.51, 1.0), ncol=1, fontsize='large')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # 调整布局留出空间
plt.savefig('chap04_hard_scaling.pdf', format='pdf')