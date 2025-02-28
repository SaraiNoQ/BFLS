import torch
import time

# 增加矩阵尺寸以增加计算复杂度
matrix_size = 10000  # 大尺寸矩阵
a = torch.randn(matrix_size, matrix_size)
b = torch.randn(matrix_size, matrix_size)

# 在 CPU 上计算
start_time = time.time()
result_cpu = torch.matmul(a, b)  # CPU 上的矩阵乘法
cpu_time = time.time() - start_time
print(f"CPU 运算时间: {cpu_time:.4f} 秒")

# 在 GPU 上计算
if torch.cuda.is_available():
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    # 确保 GPU 上没有未完成的操作，防止上次操作影响计时
    torch.cuda.synchronize()

    start_time = time.time()
    result_gpu = torch.matmul(a_gpu, b_gpu)  # GPU 上的矩阵乘法
    torch.cuda.synchronize()  # 确保所有 GPU 操作完成后再计时
    gpu_time = time.time() - start_time
    print(f"GPU 运算时间: {gpu_time:.4f} 秒")

    # 比较时间差
    print(f"加速倍数: {cpu_time / gpu_time:.2f} 倍")
else:
    print("GPU 不可用")