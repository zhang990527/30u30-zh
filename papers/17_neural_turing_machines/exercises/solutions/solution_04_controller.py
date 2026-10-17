"""
解答 4：准备控制器输入
========================================

练习 4 的完整参考解答。
"""

import torch

def prepare_controller_input(x, read_vectors):
    """
    参数：
        x：当前输入 [batch_size, input_size]
        read_vectors：张量列表，每项为 [batch_size, memory_dim]
        
    返回：
        full_input：控制器的拼接输入向量
    """
    # 沿特征维（dim=1）直接拼接
    return torch.cat([x] + read_vectors, dim=1)

if __name__ == "__main__":
    # 测试
    batch = 2
    x = torch.randn(batch, 5)
    r1 = torch.randn(batch, 8)
    r2 = torch.randn(batch, 8)
    
    full_inp = prepare_controller_input(x, [r1, r2])
    print('完整输入形状：', full_inp.shape)
    assert full_inp.shape == (batch, 21)
    print('成功')
