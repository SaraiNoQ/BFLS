import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import argparse


def get_config():
    """获取配置参数"""
    parser = argparse.ArgumentParser(description='聚类分析程序')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (yaml格式)')
    args = parser.parse_args()

    # 读取yaml配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误：配置文件读取失败 - {e}")
            exit(1)

    # 验证配置参数
    if 'n_clients' not in config:
        print("错误：配置文件中缺少 'n_clients' 参数")
        exit(1)

    return config


def main(config):
    n_clients = config['n_clients']
    print(f"config setting: n_clients = {n_clients}")

    # ======================
    # 1. 生成模拟数据（增强鲁棒性）
    # ======================
    np.random.seed(config['clustering']['random_seed'])

    # 生成模型特征（添加微小噪声防止零方差）
    n_features = 5

    # 根据客户端数量调整每个簇的大小
    cluster_sizes = [
        int(n_clients * 0.3),  # 30% 的客户端
        int(n_clients * 0.4),  # 40% 的客户端
        n_clients - int(n_clients * 0.3) - int(n_clients * 0.4)  # 剩余客户端
    ]

    model_features, _ = make_blobs(n_samples=n_clients,
                                   n_features=n_features,
                                   centers=3,
                                   cluster_std=1.5)

    # 添加高斯噪声（防止特征完全一致）
    model_features += np.random.normal(0, 0.01, model_features.shape)

    # 生成网络延迟（确保无无效值）
    network_delay = np.concatenate([
        np.abs(np.random.normal(loc=50, scale=10, size=cluster_sizes[0])),  # 确保正值
        np.abs(np.random.normal(loc=150, scale=30, size=cluster_sizes[1])),
        np.abs(np.random.normal(loc=300, scale=50, size=cluster_sizes[2])),
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

        # 使用StandardScaler替代分位数缩放
        scaler = StandardScaler()
        delay_scaled = scaler.fit_transform(delay_subset)

        # 使用更合理的eps参数设置方法
        distances = np.sort(delay_scaled, axis=0)
        knee_point = np.diff(distances, axis=0)
        eps = float(np.percentile(knee_point, 90))  # 使用90分位点作为eps

        # 增加特征维度：添加原始延迟值的差分作为第二维度
        delay_diff = np.diff(delay_subset, axis=0)
        delay_diff = np.vstack([delay_diff, delay_diff[-1]])  # 补充最后一个差分值
        delay_diff_scaled = scaler.fit_transform(delay_diff)

        # 组合两个特征维度
        combined_features = np.hstack([delay_scaled, delay_diff_scaled])

        # 调整DBSCAN参数
        min_samples = max(3, int(len(delay_subset) * 0.05))  # 动态设置min_samples
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        sub_labels = db.fit_predict(combined_features)

        # 标签合并
        valid_clusters = (sub_labels != -1)
        if np.any(valid_clusters):  # 只有当存在有效聚类时才更新标签
            sub_labels[valid_clusters] += current_max_label
            final_labels[mask] = sub_labels
            current_max_label = np.max(final_labels) + 1
        else:
            final_labels[mask] = current_max_label
            current_max_label += 1

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
                    label=f'Coarse cluster {cl}', alpha=0.6)
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
    # 附加分析：网络延迟分布可视化优化
    # ======================
    plt.figure(figsize=(15, 10))

    # 1. 箱线图比较
    plt.subplot(221)
    box_data = [network_delay[coarse_labels == cl] for cl in np.unique(coarse_labels)]
    plt.boxplot(box_data, labels=[f'Coarse Cluster {cl}' for cl in np.unique(coarse_labels)])
    plt.title("Network delay distribution of each coarse cluster (box plot)")
    plt.ylabel("Network delay (ms)")
    plt.grid(True, alpha=0.3)

    # 2. 核密度估计
    plt.subplot(222)
    for cl in np.unique(coarse_labels):
        mask = coarse_labels == cl
        plt.hist(network_delay[mask], bins=30, density=True, alpha=0.3,
                 label=f'Coarse cluster {cl}')
        # 添加核密度估计曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(network_delay[mask])
        x_range = np.linspace(network_delay.min(), network_delay.max(), 200)
        plt.plot(x_range, kde(x_range), label=f'KDE cluster{cl}')
    plt.title("Network delay distribution (kernel density estimation)")
    plt.xlabel("Network delay (ms)")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 细分簇的箱线图
    plt.subplot(223)
    # 获取所有标签，包括噪声点（-1）
    all_labels = np.unique(final_labels)
    box_data_refined = [network_delay[final_labels == label] for label in all_labels]
    labels = [f'Noise' if label == -1 else f'sub {label}' for label in all_labels]

    # 绘制箱线图
    bp = plt.boxplot(box_data_refined, labels=labels)

    # 为噪声点的箱线图添加特殊颜色
    if -1 in all_labels:
        noise_idx = np.where(all_labels == -1)[0][0]
        plt.setp(bp['boxes'][noise_idx], color='red', alpha=0.5)
        plt.setp(bp['medians'][noise_idx], color='red')

    plt.title("Network delay distribution of each sub cluster (box plot)")
    plt.ylabel("Network delay (ms)")
    plt.xticks(rotation=45)  # 旋转标签以防重叠
    plt.grid(True, alpha=0.3)

    # 4. 散点图展示
    plt.subplot(224)
    scatter = plt.scatter(range(len(network_delay)), network_delay,
                          c=final_labels, cmap='tab20',
                          alpha=0.6)
    plt.colorbar(scatter, label='sub cluster label')
    plt.title("Network delay scatter distribution")
    plt.xlabel("Sample Index")
    plt.ylabel("Network delay (ms)")
    plt.grid(True, alpha=0.3)

    # 调整子图布局，确保标签不重叠
    plt.tight_layout(pad=3.0)
    plt.show()

    # 添加统计信息输出
    print("\n=== 聚类统计信息 ===")
    print("\n粗分簇统计：")
    for cl in np.unique(coarse_labels):
        delay_stats = network_delay[coarse_labels == cl]
        print(f"\n簇 {cl}:")
        print(f"样本数量: {len(delay_stats)}")
        print(f"平均延迟: {delay_stats.mean():.2f} ms")
        print(f"标准差: {delay_stats.std():.2f} ms")
        print(f"延迟范围: [{delay_stats.min():.2f}, {delay_stats.max():.2f}] ms")

    print("\n细分簇统计：")
    for label in np.unique(final_labels[final_labels != -1]):
        delay_stats = network_delay[final_labels == label]
        print(f"\n簇 {label}:")
        print(f"样本数量: {len(delay_stats)}")
        print(f"平均延迟: {delay_stats.mean():.2f} ms")
        print(f"标准差: {delay_stats.std():.2f} ms")
        print(f"延迟范围: [{delay_stats.min():.2f}, {delay_stats.max():.2f}] ms")

    print("testing: ", len(final_labels), np.unique(final_labels))

if __name__ == "__main__":
    config = get_config()
    main(config)