import torch

def masked_aggregation(g_out: torch.Tensor) -> torch.Tensor:
    """
    解答 6：带掩码的聚合，忽略自身关系。
    """
    batch, n, n_inner, d = g_out.shape
    
    # 1. 创建单位矩阵掩码
    mask = torch.eye(n, device=g_out.device).bool()
    # 反转掩码：对角线为 0，其余为 1
    mask = ~mask
    
    # 2. 扩展掩码以匹配 g_out 形状：(N, N) -> (batch, N, N, D)
    mask = mask.view(1, n, n, 1).expand(batch, -1, -1, d)
    
    # 3. 应用掩码并求和
    masked_g = g_out * mask.float()
    return masked_g.sum(dim=(1, 2))
