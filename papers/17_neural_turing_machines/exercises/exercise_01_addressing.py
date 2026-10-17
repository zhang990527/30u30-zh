"""
练习 1：寻址机制——内容寻址
=====================================================

实现 Graves 等人（2014）定义的 NTM 寻址流程第一步，
对应第 3.3.1 节公式 5。

目标是产生权重向量 w，每个元素表示
读写头应关注某一存储位置的程度，
依据是该位置与检索键 k 的相似度。

公式（公式 5、6）：
w_c(i) = exp(beta * K[k, M(i)]) / sum exp(beta * K[k, M(j)])
其中 K[., .] 为余弦相似度。
"""

import torch
import torch.nn.functional as F

def content_addressing(k, beta, memory):
    """
    参数：
        k：检索键向量 [batch_size, M]
        beta：键强度 [batch_size, 1]
        memory：存储矩阵 [batch_size, N, M]
        
    返回：
        w_c：基于内容的权重 [batch_size, N]
    """
    # TODO：实现基于内容的寻址
    # 1. 计算余弦相似度 K[k, M]
    # 2. 乘以 beta 因子
    # 3. 沿存储位置应用 softmax（dim=1）
    pass

if __name__ == "__main__":
    # 测试你的实现
    batch, N, M = 1, 10, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    beta = torch.tensor([[10.0]]) # 较高的键强度，使关注更集中
    
    memory = torch.randn(batch, N, M)
    memory[0, 0] = k[0] # 在槽位 0 精确匹配
    
    w = content_addressing(k, beta, memory)
    
    if w is not None:
        print(f"槽位 0 的权重：{w[0, 0].item():.4f}")
        assert w[0, 0] > 0.9, '权重应集中在匹配的槽位'
