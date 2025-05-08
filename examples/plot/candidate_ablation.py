import matplotlib.pyplot as plt

# 模型规模
candiate_size = [10, 20, 30, 40, 50]

# 不同数据集在不同模型规模上的Hits@1指标
hits1_zh_32b = [0.894, 0.851,0.717,0.511,0.390]
hits1_ja_32b = [0.866, 0.839,0.784,0.533,0.381]
hits1_fr_32b = [0.957,0.924,0.848,0.590,0.479]

hits1_zh_14b = [0.779,0.743,0.629,0.467,0.371]
hits1_ja_14b = [0.842,0.819,0.714,0.390,0.371]
hits1_fr_14b = [0.952,0.952,0.819,0.552,0.466]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(5, 4))

plot_size = '14b'

if plot_size == '32b':
    plot_number = [
        hits1_zh_32b, hits1_ja_32b, hits1_fr_32b
    ]
elif plot_size == '14b':
    plot_number = [
        hits1_zh_14b, hits1_ja_14b, hits1_fr_14b
    ]
else:
    raise ValueError('Please select a valid model size to plot.')

# 绘制三条折线
ax.plot(candiate_size, plot_number[0], marker='o', linestyle='-', label='ZH-EN')
ax.plot(candiate_size, plot_number[1], marker='s', linestyle='--', label='JA-EN')
ax.plot(candiate_size, plot_number[2], marker='^', linestyle='-.', label='FR-EN')

# 设置图例
ax.legend(loc='lower left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# 设置横坐标和纵坐标的标签
# ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
ax.set_ylabel('Hits@1', fontsize=12)

# # 自动调整横纵坐标轴的范围
# ax.set_xscale('log')  # 设置横坐标为对数尺度以反映参数规模的变化
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlim(10, 50)
# ax.autoscale()
plt.xticks(candiate_size)
# ax.set_ylim(0.47, 1.0)
#save
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='best', bbox_to_anchor=(0.51, 1.0), ncol=1, fontsize='large')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # 调整布局留出空间
plt.savefig(f'candidate_ablation_{plot_size}.pdf', format='pdf')