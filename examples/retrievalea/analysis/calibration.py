import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# 假设这些是你的模型预测的正确对齐概率和实际的对齐情况（1为正确对齐，0为错误对齐）
pred_probs = np.random.rand(1000)  # 随机生成的预测概率
true_labels = np.random.randint(0, 2, 1000)  # 随机生成的实际标签

# 计算可靠性曲线
prob_true, prob_pred = calibration_curve(true_labels, pred_probs, n_bins=10)

# 绘制可靠性图
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Reliability Diagram')
plt.savefig('reliability_diagram.svg', format='svg')