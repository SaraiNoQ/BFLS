import numpy as np

# 定义矩阵 P
P = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
], dtype=float)
A = np.array([
    [1,-1,2],
    [2,0,1],
    [3,3,5],
], dtype=float)

# 计算逆矩阵
P_inverse = np.linalg.inv(P)

# 打印结果
print("矩阵 P 的逆矩阵为：")
print(P_inverse)

# 验证结果：P * P_inverse 是否等于单位矩阵
I = np.dot(A, P)
print("\n验证结果（A * P）：")
print(I)