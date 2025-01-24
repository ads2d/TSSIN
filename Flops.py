from model.tsrn import TSRN_TL  # 导入模型类
from fvcore.nn import FlopCountAnalysis
import torch
import torch.nn as nn

# 判断是否有可用的 GPU，如果有则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确保 GPU 可用

# 实例化模型并将其转移到相同设备
model = TSRN_TL().to(device)

# 假设你已经定义了输入的张量尺寸
input_tensor = torch.randn(4, 4, 16, 64).to(device)  # 将输入数据转移到相同设备

# 确保模型和输入都在相同的设备上
print(f"Model device: {next(model.parameters()).device}")
print(f"Input tensor device: {input_tensor.device}")

# 计算 FLOPs
flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")
