"""
解答 3：存储更新（擦除与添加）
=========================================

练习 3 的完整参考解答。
"""

import torch

def memory_update(w, erase, add, prev_memory):
    """
    参数：
        w：写入权重 [batch_size, N]
        erase：擦除向量 [batch_size, M]
        add：添加向量 [batch_size, M]
        prev_memory：存储矩阵 [batch_size, N, M]
        
    返回：
        new_memory：更新后的存储矩阵 [batch_size, N, M]
    """
    # 1. 计算擦除项
    # w 为 [batch, N]，erase 为 [batch, M]；外积形状为 [batch, N, M]
    erase_term = torch.matmul(w.unsqueeze(2), erase.unsqueeze(1))
    
    # 2. 对 prev_memory 执行擦除
    memory_after_erase = prev_memory * (1 - erase_term)
    
    # 3. 计算添加项
    add_term = torch.matmul(w.unsqueeze(2), add.unsqueeze(1))
    
    # 4. 将添加项写入存储
    new_memory = memory_after_erase + add_term
    
    return new_memory

if __name__ == "__main__":
    # 测试
    batch, N, M = 1, 5, 4
    prev_memory = torch.ones(batch, N, M)
    w = torch.zeros(batch, N)
    w[0, 2] = 1.0 
    
    erase = torch.ones(batch, M) 
    add = torch.tensor([[1.2, 3.4, 5.6, 7.8]])
    
    new_memory = memory_update(w, erase, add, prev_memory)
    print('槽位 2 的新存储：', new_memory[0, 2])
    assert torch.allclose(new_memory[0, 2], add[0])
    print('成功')
