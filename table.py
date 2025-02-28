import numpy as np
import matplotlib.pyplot as plt

# 数据
x = np.array([0, 10, 20, 30, 40, 50, 60])  # 恶意参与者百分比
vanilla_fl = np.array([1.00, 0.97, 0.99, 0.95, 0.93, 0.90, 0.89])  # Vanilla FL 准确率
vanilla_fl_error = np.array([0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05])  # Vanilla FL 误差

biscotti = np.array([1.00, 0.99, 0.98, 0.98, 0.96, 0.87, 0.85])  # Biscotti 准确率
biscotti_error = np.array([0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06])  # Biscotti 误差

biscotti_wo_dp = np.array([1.00, 0.99, 0.99, 0.98, 0.97, 0.91, 0.83])  # Biscotti (w/o DP) 准确率
biscotti_wo_dp_error = np.array([0.01, 0.03, 0.04, 0.05, 0.05, 0.06, 0.07])  # Biscotti (w/o DP) 误差

block_dfl = np.array([1.00, 1.00, 1.00, 0.99, 0.99, 0.94, 0.92])  # BlockDFL 准确率
block_dfl_error = np.array([0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05])  # BlockDFL 误差

# 创建图表
plt.figure(figsize=(8, 6))

# Vanilla FL
plt.errorbar(x, vanilla_fl, yerr=vanilla_fl_error, fmt='-o', label='Vanilla FL', color='orange', capsize=3)

# Biscotti
plt.errorbar(x, biscotti, yerr=biscotti_error, fmt='-s', label='Biscotti', color='blue', capsize=3)

# Biscotti (w/o DP)
plt.errorbar(x, biscotti_wo_dp, yerr=biscotti_wo_dp_error, fmt='-^', label='Biscotti (w/o DP)', color='skyblue', linestyle='dotted', capsize=3)

# BlockDFL
plt.errorbar(x, block_dfl, yerr=block_dfl_error, fmt='-d', label='BlockDFL', color='red', capsize=3)

# 图例和标签
plt.xlabel('Percentage of Malicious Participants', fontsize=12)
plt.ylabel('Average Test Accuracy', fontsize=12)
plt.title('MNIST (non-IID)', fontsize=14)
plt.legend(fontsize=10, loc='lower left')

# 坐标轴范围
plt.ylim(0.86, 1.01)
plt.xlim(-5, 65)

# 网格线
plt.grid(alpha=0.3)

# 保存或展示
plt.tight_layout()
plt.show()
