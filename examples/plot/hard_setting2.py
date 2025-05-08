import numpy as np
import matplotlib.pyplot as plt

# 示例数据
methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5',
           'Method 6', 'Method 7', 'Method 8', 'Method 9', 'Method 10']
regular_setting_performance = np.random.rand(10) * 100
hard_setting_performance = np.random.rand(10) * 100
performance_decrease = regular_setting_performance - hard_setting_performance

# 设置
bar_width = 0.35  # 柱状图的宽度
index = np.arange(len(methods))  # 方法的索引

# 颜色
colors_regular = 'cornflowerblue'
colors_hard = 'salmon'

# 绘制柱状图
fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(index - bar_width/2, regular_setting_performance, bar_width, label='Regular Setting',
               color=colors_regular, edgecolor='black')
bars2 = ax.bar(index + bar_width/2, hard_setting_performance, bar_width, label='Hard Setting',
               color=colors_hard, edgecolor='black')

# 添加性能下降的注释
for i in range(len(methods)):
    height = max(regular_setting_performance[i], hard_setting_performance[i])
    diff = performance_decrease[i]
    ax.annotate(f'{diff:.2f}',
                xy=(i, height),
                xytext=(i, height + 5),
                textcoords="offset points",
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))

# 添加一些文本标签
ax.set_xlabel('Methods')
ax.set_ylabel('Performance')
ax.set_title('Performance of different methods under Regular and Hard settings')
ax.set_xticks(index)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()

# 显示图表
# plt.tight_layout()
plt.savefig('regular_hard2.pdf', format='pdf')
