import torch

def generate_pairs_via_broadcasting(objects: torch.Tensor) -> torch.Tensor:
    """
    解答 1：通过广播实现成对对象拼接。
    """
    batch_size, num_objects, object_dim = objects.size()
    
    # 扩展对象以构造笛卡尔积
    o_i = objects.unsqueeze(2).repeat(1, 1, num_objects, 1)
    o_j = objects.unsqueeze(1).repeat(1, num_objects, 1, 1)
    
    # 拼接对象对
    pairs = torch.cat([o_i, o_j], dim=-1)
    
    # 将 N x N 网格展平为 N^2 序列
    return pairs.view(batch_size, num_objects * num_objects, -1)
