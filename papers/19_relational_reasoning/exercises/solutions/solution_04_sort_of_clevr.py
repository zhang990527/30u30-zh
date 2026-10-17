"""
解答 4：Sort-of-CLEVR 对象对生成
按 30u30 第 19 天的格式组织
"""

import torch

def create_contextual_pairs(objects: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """
    解答：向局部对象对注入全局问题向量。
    """
    batch_size, n_objects, object_dim = objects.shape
    
    # 1. 扩展对象以创建 (o_i, o_j) 网格
    # 形状：(B, N, N, D)
    o_i = objects.unsqueeze(2).expand(-1, -1, n_objects, -1)
    o_j = objects.unsqueeze(1).expand(-1, n_objects, -1, -1)
    
    # 2. 扩展问题以匹配网格
    # 形状：(B, N, N, Q)
    q_expanded = question.unsqueeze(1).unsqueeze(2).expand(-1, n_objects, n_objects, -1)
    
    # 3. 拼接全部内容
    # 输出：(B, N, N, 2D + Q)
    return torch.cat([o_i, o_j, q_expanded], dim=-1)
