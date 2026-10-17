"""
练习 3：存储更新（擦除与添加）
=========================================

NTM 写头的更新由顺序执行的擦除和添加操作定义
（Graves 等，2014，第 3.3 节，公式 3、4）。

公式：
擦除（公式 3）：M_t_erase(i) = M_{t-1}(i) * [1 - w_t(i) * e_t]
添加（公式 4）：M_t(i) = M_t_erase(i) + w_t(i) * a_t
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
    # TODO：实现公式 3 与公式 4
    # 1. 计算擦除项（w 与 erase 的外积）
    # 2. 计算添加项（w 与 add 的外积）
    pass

if __name__ == "__main__":
    # 测试你的实现
    batch, N, M = 1, 5, 4
    prev_memory = torch.ones(batch, N, M)
    w = torch.zeros(batch, N)
    w[0, 2] = 1.0 # 目标为槽位 2
    
    erase = torch.ones(batch, M) # 完全擦除
    add = torch.tensor([[1.2, 3.4, 5.6, 7.8]]) # 选择性添加
    
    new_memory = memory_update(w, erase, add, prev_memory)
    
    if new_memory is not None:
        print(f"槽位 2 内容：{new_memory[0, 2].tolist()}")
        assert torch.allclose(new_memory[0, 2], add[0])
        assert torch.allclose(new_memory[0, 0], prev_memory[0, 0])
