import torch

def prove_counting_bias(num_objects: int):
    """
    解答 7：为什么 sum() 比 mean() 更适合保留计数信息。
    """
    # 创建 N 个对象，每个对象的“属性”强度为 1。
    objects = torch.ones(num_objects)
    
    sum_res = objects.sum().item()
    mean_res = objects.mean().item()
    
    # 求和得到计数 N，均值得到平均强度 1。
    # 若需对对象计数，只有 sum_res 会随 N 改变。
    return sum_res == num_objects and mean_res == 1.0
