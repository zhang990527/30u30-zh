import torch
import numpy as np

def generate_nth_farthest_batch(batch_size, seq_len, vector_dim, n_index=0):
    """
    为本章距离任务生成一个批次。
    
    任务：给定向量序列，输出与序列中指定参考向量
    欧氏距离最远的向量；参考位置由 n_index 指定。
    
    输入：
        batch_size：样本数。
        seq_len：序列长度。
        vector_dim：每个向量的维度。
        n_index：参考向量索引，默认为 0。
        
    返回：
        inputs：(batch, seq_len, vector_dim)，[-1, 1] 内随机向量
        targets：(batch, vector_dim)，距离第 n 个参考向量最远的向量
    """
    # 1. 生成范围为 [-1, 1] 的随机输入
    inputs = torch.rand(batch_size, seq_len, vector_dim) * 2 - 1
    targets = []
    
    for i in range(batch_size):
        seq = inputs[i]
        
        # TODO：识别 n_index 处的参考向量
        ref_vec = # 在此填写代码
        
        # TODO：计算 ref_vec 到序列中其他向量的欧氏距离
        # dists 的形状应为 (seq_len,)
        dists = # 在此填写代码
        
        # TODO：找出最大距离对应索引
        farthest_idx = # 在此填写代码
        
        # TODO：获取目标向量
        target_vec = # 在此填写代码
        
        targets.append(target_vec)
        
    targets = torch.stack(targets)
    return inputs, targets

def test_generator():
    print('正在测试数据生成器……')
    B, L, D = 2, 5, 2
    inputs, targets = generate_nth_farthest_batch(B, L, D)
    
    assert inputs.shape == (B, L, D)
    assert targets.shape == (B, D)
    
    # 对第一个样本手工检查正确性
    seq = inputs[0]
    ref = seq[0]
    dists = torch.norm(seq - ref, dim=1)
    max_idx = torch.argmax(dists)
    expected = seq[max_idx]
    
    # 使用 allclose，考虑浮点精度
    assert torch.allclose(targets[0], expected), '目标验证失败！'
    print('测试通过！')

if __name__ == "__main__":
    test_generator()
