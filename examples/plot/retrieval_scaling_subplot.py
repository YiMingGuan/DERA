import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# 创建数据
model_params = [128, 425, 1250]

performance_dataset1_hits1_hits10_mrr = [
    [0.951, 0.955, 0.961],
    [0.991, 0.992, 0.994],
    [0.967, 0.970, 0.974]
]
performance_dataset2_hits1_hits10_mrr = [
    [0.938, 0.950, 0.958],
    [0.984, 0.989, 0.992],
    [0.956, 0.965, 0.971]
]

performance_all = []
for i in range(3):
    performance_all.extend(performance_dataset1_hits1_hits10_mrr[i])
    performance_all.extend(performance_dataset2_hits1_hits10_mrr[i])



model_params_billion = [p / 1000 for p in model_params]


# 创建一个1行3列的子图布局，所有子图共享X轴
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharex=True, sharey=False)
xlabel_str = [
    'Hits@1',
    'Hits@10',
    'MRR'
]
# 分别在每个子图上绘制数据
for i, ax in enumerate(axes):
    ax.plot(model_params_billion, performance_dataset1_hits1_hits10_mrr[i], '-', label='DBP15K-ZH-EN', marker='^', color='red', markerfacecolor='red', markersize=12)
    ax.plot(model_params_billion, performance_dataset2_hits1_hits10_mrr[i], '-', label='DBP15K-JA-EN', marker='o', color='blue', markerfacecolor='blue', fillstyle='none', markersize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # 调整x轴的刻度显示，显示参数量（以Billion为单位）
    ax.set_xticks(model_params_billion)
    ax.set_xticklabels([f'{p}' for p in model_params_billion])
    ax.set_ylabel(xlabel_str[i], fontsize=14)
    min_perf, max_perf = min(performance_dataset1_hits1_hits10_mrr[i]), max(performance_dataset2_hits1_hits10_mrr[i])
    padding = (max_perf - min_perf) * 0.1
    ax.set_ylim([min(performance_dataset1_hits1_hits10_mrr[i] + performance_dataset2_hits1_hits10_mrr[i]) - padding, max(performance_dataset1_hits1_hits10_mrr[i] + performance_dataset2_hits1_hits10_mrr[i]) + padding])
    # ax.set_ylim([min_perf - padding, max_perf + padding])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


# 设置共享的X轴标签
fig.text(0.5, 0.03, 'Model scale (# parameters in billions)', ha='center', va='center', fontsize=14)

# 创建一个共享的图例
# 由于所有子图绘制的数据相同，我们可以从任何一个子图创建图例
# 并将其放置在图表外面，这里选择放在图表的下方
lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1), fontsize=14)

# 调整子图间距
plt.subplots_adjust(wspace=0.35, hspace=0.6)

# plt.show()
plt.savefig('retrieval_scaling_subplot.pdf', format='pdf')
