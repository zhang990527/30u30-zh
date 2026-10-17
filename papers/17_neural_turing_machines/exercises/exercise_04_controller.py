"""
练习 4：准备控制器输入
========================================

将处理与存储解耦，需要 NTM 控制器观察
上一时间步的存储状态（Graves 等，2014，第 2 节）。

任务：
实现输入向量 x 与读取向量 r_{t-1} 的拼接。
拼接结果构成 RNN／控制器的输入。
"""

import torch

def prepare_controller_input(x, read_vectors):
    """
    参数：
        x：当前输入 [batch_size, input_size]
        read_vectors：读取向量列表，每项形状为 [batch_size, M]
        
    返回：
        full_input：拼接向量 [batch_size, input_size + num_heads * M]
    """
    # TODO：沿特征维拼接 x 与所有读取向量
    pass

if __name__ == "__main__":
    # 测试
    batch = 2
    x = torch.randn(batch, 5)
    r1 = torch.randn(batch, 8)
    r2 = torch.randn(batch, 8)
    
    full_inp = prepare_controller_input(x, [r1, r2])
    if full_inp is not None:
        print(f"结果维度：{full_inp.shape[1]}")
        assert full_inp.shape[1] == 21
