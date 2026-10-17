import torch
from implementation import RelationNetwork

def verify_permutation_invariance(model: RelationNetwork, objects: torch.Tensor) -> bool:
    """
    解答 3：验证排列不变性。
    """
    model.eval()
    
    # 原始顺序的前向传播
    with torch.no_grad():
        out_orig = model(objects)
        
    # 沿 N 维打乱对象
    indices = torch.randperm(objects.size(1))
    shuffled_objects = objects[:, indices, :]
    
    # 打乱后的前向传播
    with torch.no_grad():
        out_shuffled = model(shuffled_objects)
        
    diff = torch.abs(out_orig - out_shuffled).max().item()
    return diff < 1e-6
