import numpy as np
import matplotlib.pyplot as plt

# 数据
x = np.array([0, 10, 20, 30, 40, 50, 60])  # 恶意参与者百分比

# 数据集1: MNIST (IID)
vanilla_fl_mnist_iid = np.array([0, 20, 40, 60, 80, 100, 100])
biscotti_mnist_iid = np.array([0, 15, 35, 55, 75, 95, 100])
biscotti_wo_dp_mnist_iid = np.array([0, 10, 30, 50, 70, 90, 100])
block_dfl_mnist_iid = np.array([0, 5, 15, 30, 50, 60, 70])

# 数据集2: MNIST (non-IID)
vanilla_fl_mnist_non_iid = np.array([0, 25, 50, 70, 90, 100, 100])
biscotti_mnist_non_iid = np.array([0, 20, 45, 65, 85, 95, 100])
biscotti_wo_dp_mnist_non_iid = np.array([0, 15, 40, 60, 80, 90, 100])
block_dfl_mnist_non_iid = np.array([0, 10, 25, 45, 65, 75, 85])

# 数据集3: CIFAR-10 (IID)
vanilla_fl_cifar_iid = np.array([0, 30, 60, 80, 90, 100, 100])
biscotti_cifar_iid = np.array([0, 25, 55, 75, 85, 95, 100])
biscotti_wo_dp_cifar_iid = np.array([0, 20, 50, 70, 80, 90, 100])
block_dfl_cifar_iid = np.array([0, 10, 20, 40, 60, 70, 80])

# 数据集4: CIFAR-10 (non-IID)
vanilla_fl_cifar_non_iid = np.array([0, 35, 65, 85, 95, 100, 100])
biscotti_cifar_non_iid = np.array([0, 30, 60, 80, 90, 100, 100])
biscotti_wo_dp_cifar_non_iid = np.array([0, 25, 55, 75, 85, 95, 100])
block_dfl_cifar_non_iid = np.array([0, 15, 30, 50, 70, 80, 90])

# 图表布局
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

datasets = [
    ("MNIST (IID)", vanilla_fl_mnist_iid, biscotti_mnist_iid, biscotti_wo_dp_mnist_iid, block_dfl_mnist_iid),
    ("MNIST (non-IID)", vanilla_fl_mnist_non_iid, biscotti_mnist_non_iid, biscotti_wo_dp_mnist_non_iid,
     block_dfl_mnist_non_iid),
    ("CIFAR-10 (IID)", vanilla_fl_cifar_iid, biscotti_cifar_iid, biscotti_wo_dp_cifar_iid, block_dfl_cifar_iid),
    ("CIFAR-10 (non-IID)", vanilla_fl_cifar_non_iid, biscotti_cifar_non_iid, biscotti_wo_dp_cifar_non_iid,
     block_dfl_cifar_non_iid)
]

# 循环绘制每个子图
for i, (title, vanilla, biscotti, biscotti_wo_dp, block_dfl) in enumerate(datasets):
    axs[i].plot(x, vanilla, 'o-', label="Vanilla FL", color='orange', linestyle='dotted')
    axs[i].plot(x, biscotti, '^-', label="Biscotti", color='blue', linestyle='dashed')
    axs[i].plot(x, biscotti_wo_dp, 's-', label="Biscotti (w/o DP)", color='skyblue', linestyle='dashdot')
    axs[i].plot(x, block_dfl, 'd-', label="BlockDFL", color='red', linestyle='solid')

    axs[i].set_title(f"({chr(97 + i)}) {title}")
    axs[i].set_xlabel("Percentage of Malicious Participants")
    axs[i].set_ylim(0, 105)
    axs[i].grid(alpha=0.3)
    if i == 0:
        axs[i].set_ylabel("Ratio of Successful Attacks")
    if i == 3:
        axs[i].legend(fontsize=8, loc="lower right")

# 调整布局和展示
plt.tight_layout()
plt.show()
