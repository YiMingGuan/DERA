import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['ZH-EN', 'JA-EN', 'FR-EN']

# 基准LLM1的Hits@1数据
llm1_original_hits_at_1 = [0.42, 0.4448, 0.432]
llm1_enhanced_hits_at_1 = [0.749, 0.785, 0.805]

# 基准LLM2的Hits@1数据
llm2_original_hits_at_1 = [0.42, 0.4448, 0.432]
llm2_enhanced_hits_at_1 = [0.769, 0.792, 0.812]

# Hits@10数据（性能上限）
hits_at_10 = [0.7895, 0.815, 0.812]

x = np.arange(len(datasets))  # 标签位置
width = 0.25  # 柱状图的宽度

fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharey=True)

# 第一个LLM的柱状图
axs[0].bar(x - width/2, llm1_original_hits_at_1, width, label='Original Hits@1', color='red', alpha=0.7, hatch='//')
axs[0].bar(x + width/2, llm1_enhanced_hits_at_1, width, label='Enhanced Hits@1', color='orange', alpha=0.7, hatch='\\')

# 第二个LLM的柱状图
axs[1].bar(x - width/2, llm2_original_hits_at_1, width, color='red', alpha=0.7, hatch='//')
axs[1].bar(x + width/2, llm2_enhanced_hits_at_1, width, color='orange', alpha=0.7, hatch='\\')
axs[0].plot([], [], color='grey', linestyle='dashed', label='Hits@10')
# axs[1].plot([], [], color='grey', linestyle='dashed', label='Hits@10')
# 绘制Hits@10的性能上限线
for ax in axs:
    for i in range(len(datasets)):
        ax.hlines(y=hits_at_10[i], xmin=x[i]-width, xmax=x[i]+width, colors='grey', linestyles='dashed')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 添加公共元素
# fig.suptitle('Hits@1 Comparison with Performance Upper Bound for Two LLMs')
axs[0].set_ylabel('Hits@1')
axs[0].set_xlabel('Qwen-14B-Chat')
axs[1].set_xlabel('Qwen-32B-Chat-AWQ')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize='large')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局，预留出空间放置图例
plt.savefig('gcnalign-rerank.pdf', format='pdf')
# plt.show()
