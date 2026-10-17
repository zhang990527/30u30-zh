"""
练习 1：实现指针注意力机制
按 30u30 第 18 天的格式组织

任务：完成 PointerAttention 模块，计算加性注意力分数。
依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    """
    指针注意力机制。
    
    TODO：补全下方 __init__ 与 forward 方法。
    """
    def __init__(self, hidden_size):
        super().__init__()
        # TODO：初始化三个线性层（W1、W2、v）
        # 提示：W1、W2 映射 hidden_size -> hidden_size，v 映射 hidden_size -> 1。
        # 按常见实现设为 bias=False。
        self.W1 = None # TODO：待实现
        self.W2 = None # TODO：待实现
        self.v = None  # TODO：待实现

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        注意力分数的前向计算。
        
        参数：
            decoder_hidden: (batch, hidden_size)
            encoder_outputs: (batch, seq_len, hidden_size)
            mask: (batch, seq_len)
            
        返回：
            log_probs: (batch, seq_len)
        """
        # TODO：实现指针注意力公式（公式 3）
        # 1. 分别用 W2、W1 投影 decoder_hidden 与 encoder_outputs
        # 2. 将二者求和，再应用 tanh
        # 3. 用 v 投影结果，再压缩最后一维
        # 4. 若给出 mask，用 masked_fill 与 float('-inf') 屏蔽相应位置
        # 5. 沿最后一维返回 log_softmax

        pass # TODO：待实现

if __name__ == "__main__":
    print('练习 1：指针注意力')
    print("=" * 50)
    print('\n任务：完成 exercise_01_pointer_attention.py 中的 TODO')
    
    # 测试你的实现
    hidden_size = 64
    batch_size = 2
    seq_len = 5
    
    try:
        attn = PointerAttention(hidden_size)
        d = torch.randn(batch_size, hidden_size)
        e = torch.randn(batch_size, seq_len, hidden_size)
        out = attn(d, e)
        
        print(f"\n[OK] 输出形状：{out.shape}")
        print('现在检查你的逻辑是否符合论文公式 3！')
    except Exception as e:
        print(f"\n[FAIL] 发生错误：{e}")
