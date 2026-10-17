"""
解答 1：寻址机制——内容寻址
=====================================================

实现 Graves 等人（2014）第 3.3.1 节的内容寻址逻辑。
该机制使控制器可以根据与当前检索键的相似度
检索信息。
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
    # 1. 余弦相似度 K[k, M]（公式 6）
    # 将 k 与存储向量都归一化为单位长度
    norm_k = k.norm(p=2, dim=1, keepdim=True) + 1e-8
    norm_mem = memory.norm(p=2, dim=2) + 1e-8
    
    # 使用批量矩阵乘法，得到所有存储槽位的点积
    # [batch, N, M] @ [batch, M, 1] -> [batch, N, 1]
    dot_product = torch.matmul(memory, k.unsqueeze(2)).squeeze(2)
    sim = dot_product / (norm_k * norm_mem)
    
    # 2. 键强度与 softmax（公式 5）
    # Beta 控制关注精度；beta 较高时，关注更集中。
    w_c = F.softmax(beta * sim, dim=1)
    
    return w_c

if __name__ == "__main__":
    # 回归检查
    batch, N, M = 1, 10, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    beta = torch.tensor([[10.0]])
    
    memory = torch.randn(batch, N, M)
    memory[0, 0] = k[0]
    
    w = content_addressing(k, beta, memory)
    print(f"槽位 0 的内容权重：{w[0, 0].item():.4f}")
    assert w[0, 0] > 0.9
