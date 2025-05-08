import matplotlib.pyplot as plt

model_params = [128, 425, 1250]  # 以百万为单位，更新为刚计算的参数量
performance_dataset1_hits1 = [0.951, 0.955, 0.961]
performance_dataset2_hits1 = [0.938, 0.950, 0.958]

performance_dataset1_mrr = [0.967, 0.970, 0.974]
performance_dataset2_mrr = [0.956, 0.965, 0.971]

performance_dataset1 = performance_dataset1_mrr
performance_dataset2 = performance_dataset2_mrr

# 更新参数量单位为Billion
model_params_billion = [p / 1000 for p in model_params]  # 从Million转换为Billion

# 创建图形和轴对象，调整尺寸
fig, ax = plt.subplots(figsize=(4, 5))

# 绘制折线图，明确指定折线连接处的圆圈大小
ax.plot(model_params_billion, performance_dataset1, '-', label='DBP15K-ZH-EN', marker='^', markersize=10, color='red', markerfacecolor='red')
ax.plot(model_params_billion, performance_dataset2, '-', label='DBP15K-JA-EN', marker='o', markersize=10, color='blue', markerfacecolor='blue', fillstyle='none')

# 设置图表标题和轴标签
xlabel_str = 'Hits@1'
# ax.set_title('Model Parameters (Billion) vs. Performance', fontsize=16)
ax.set_xlabel('Model scale (# parameters in billions)', fontsize=12)
ax.set_ylabel(xlabel_str, fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
# 调整x轴的刻度显示，显示参数量（以Billion为单位）
ax.set_xticks(model_params_billion)
ax.set_xticklabels([f'{p}' for p in model_params_billion])

min_perf, max_perf = min(performance_dataset1), max(performance_dataset2)

padding = (max_perf - min_perf) * 0.1

# 自适应纵坐标范围，略微扩大范围以展示趋势
ax.set_ylim([min(performance_dataset1 + performance_dataset2) - padding, max(performance_dataset1 + performance_dataset2) + padding])
# ax.legend(loc='upper center', frameon=False)

# 展示图表
plt.tight_layout()
# plt.show()
plt.savefig('retrieval_scaling_mrr.pdf', format='pdf')
