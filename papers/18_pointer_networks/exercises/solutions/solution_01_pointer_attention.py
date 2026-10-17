"""
解答 1：实现指针注意力机制
按 30u30 第 18 天的格式组织

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # 1. 状态投影
        decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
        encoder_proj = self.W1(encoder_outputs)
        
        # 2. 加性注意力（公式 3）
        combined = torch.tanh(encoder_proj + decoder_proj)
        
        # 3. 投影为分数
        scores = self.v(combined).squeeze(-1)
        
        # 4. 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        return F.log_softmax(scores, dim=-1)
