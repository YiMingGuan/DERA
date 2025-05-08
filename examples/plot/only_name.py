import matplotlib.pyplot as plt

# 示例数据
datasets = ['ZH-EN', 'JA-EN', 'FR-EN']
methods = ['GMNN', 'SelfKG', 'TEA-NSP', 'TEA-MLM', 'ETS', 'baseline']

# 假设的Hits@1性能数据，每个方法在每个数据集上的表现
performance = {
    'GMNN': [0.679, 0.740, 0.894],
    'SelfKG': [0.745, 0.816, 0.957],
    'TEA-NSP': [0.815, 0.890, 0.968],
    'TEA-MLM': [0.831, 0.883, 0.968],
    'ETS': [0.846, 0.866, 0.980],
    'baseline': [0.604, 0.745, 0.874]
}


# 设置不同的线条样式和标记以区分不同的方法
line_styles = {
    'GMNN': {'color': 'b', 'marker': 'o', 'linestyle': '-'},
    'SelfKG': {'color': 'g', 'marker': 's', 'linestyle': '--'},
    'TEA-NSP': {'color': 'r', 'marker': '^', 'linestyle': '-.'},
    'TEA-MLM': {'color': 'c', 'marker': 'D', 'linestyle': ':'},
    'ETS': {'color': 'm', 'marker': 'h', 'linestyle': '-'},
    'baseline': {'color': 'k', 'marker': 'p', 'linestyle': '--'}
}

# 创建折线图
plt.figure(figsize=(6, 6))

# 为每个方法绘制一条线，应用不同的样式
for method, scores in performance.items():
    plt.plot(datasets, scores, **line_styles[method], label=method)

# 添加标题和轴标签
# plt.title('Methods Performance Comparison on Different Datasets')
plt.xlabel('Dataset')
plt.ylabel('Hits@1')

# 将图例放在图表的上方中央
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# 调整网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 保存为SVG文件
plt.savefig('only_name_methods_performance_comparison_updated_(6,6).pdf', format='pdf')

# 显示图表
# plt.show()
