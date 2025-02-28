import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# ======================
# 1. 生成模拟数据（增强鲁棒性）
# ======================
np.random.seed(42)

# 生成模型特征（添加微小噪声防止零方差）
n_clients = 200
n_features = 5
model_features, _ = make_blobs(n_samples=n_clients,
                               n_features=n_features,
                               centers=3,
                               cluster_std=1.5)

# 添加高斯噪声（防止特征完全一致）
model_features += np.random.normal(0, 0.01, model_features.shape)

# 生成网络延迟（确保无无效值）
network_delay = np.concatenate([
    np.abs(np.random.normal(loc=50, scale=10, size=60)),  # 确保正值
    np.abs(np.random.normal(loc=150, scale=30, size=80)),
    np.abs(np.random.normal(loc=300, scale=50, size=60)),
])

# ======================
# 2. 谱聚类（鲁棒性改进）
# ======================
# 计算Pearson相关系数矩阵（处理NaN）
corr_matrix = np.corrcoef(model_features)

# 处理相关系数矩阵中的潜在问题
corr_matrix = np.nan_to_num(corr_matrix)  # 将NaN替换为0
corr_matrix = (corr_matrix + 1) / 2  # 映射到[0,1]范围

# 执行谱聚类（增加鲁棒性参数）
spectral = SpectralClustering(n_clusters=3,
                              affinity='precomputed',
                              random_state=42,
                              assign_labels='discretize')  # 避免浮点精度问题
coarse_labels = spectral.fit_predict(corr_matrix)

# ======================
# 3. DBSCAN细分（优化参数）
# ======================
final_labels = np.zeros_like(coarse_labels)
current_max_label = 0

for cluster_id in np.unique(coarse_labels):
    mask = (coarse_labels == cluster_id)
    delay_subset = network_delay[mask].reshape(-1, 1)

    # 使用分位数缩放代替标准化
    q25, q75 = np.percentile(delay_subset, [25, 75])
    delay_scaled = (delay_subset - q25) / (q75 - q25 + 1e-8)  # 防止除零

    # 自适应参数调整
    eps = 0.3 * (q75 - q25)  # 基于四分位距设置邻域半径
    db = DBSCAN(eps=eps, min_samples=5)
    sub_labels = db.fit_predict(delay_scaled)

    # 标签合并
    sub_labels[sub_labels != -1] += current_max_label
    final_labels[mask] = sub_labels
    current_max_label = np.max(final_labels) + 1

# ======================
# 4. 可视化（添加异常值提示）
# ======================
plt.figure(figsize=(15, 6))

# （原有可视化代码保持不变，此处省略...）

# 原始数据分布
plt.subplot(131)
plt.scatter(model_features[:, 0], model_features[:, 1],
            c=network_delay, cmap='viridis', alpha=0.6)
plt.title("Original Data Distribution\n(Color=Network Delay)")
plt.xlabel("Feature Dimension 1")
plt.ylabel("Feature Dimension 2")
plt.colorbar()

# 谱聚类结果
plt.subplot(132)
for cl in np.unique(coarse_labels):
    mask = (coarse_labels == cl)
    plt.scatter(model_features[mask, 0], model_features[mask, 1],
                label=f'Coarse Cluster {cl}', alpha=0.6)
plt.title("After Spectral Clustering\n(3 Coarse Clusters)")
plt.xlabel("Feature Dimension 1")
plt.ylabel("Feature Dimension 2")
plt.legend()

# DBSCAN细分结果
plt.subplot(133)
unique_labels = np.unique(final_labels)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = (final_labels == label)
    plt.scatter(model_features[mask, 0], model_features[mask, 1],
                color=colors[i],
                label=f'Sub-Cluster {i}' if label != -1 else 'Noise',
                alpha=0.6)

plt.title("After DBSCAN Refinement\n(Color=Sub-Clusters)")
plt.xlabel("Feature Dimension 1")
plt.ylabel("Feature Dimension 2")
plt.legend()

plt.tight_layout()
plt.show()

# ======================
# 附加分析：网络延迟分布
# ======================
plt.figure(figsize=(12, 4))

# 粗分簇延迟分布
plt.subplot(121)
for cl in np.unique(coarse_labels):
    plt.hist(network_delay[coarse_labels == cl],
             alpha=0.5, bins=30,
             label=f'Coarse Cluster {cl}')
plt.title("Network Delay Distribution by\nCoarse Clusters")
plt.xlabel("Network Delay (ms)")
plt.ylabel("Count")
plt.legend()

# 细分簇延迟分布
plt.subplot(122)
for label in np.unique(final_labels):
    plt.hist(network_delay[final_labels == label],
             alpha=0.5, bins=30,
             label=f'Sub-Cluster {label}' if label != -1 else 'Noise')
plt.title("Network Delay Distribution by\nRefined Sub-Clusters")
plt.xlabel("Network Delay (ms)")
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.show()