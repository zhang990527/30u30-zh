"""
解答 1：基本指针注意力机制
==============================================

指针网络注意力机制的完整实现。
原仓库将其对应到 Vinyals 等（2015）的公式 1；该编号尚待一手核验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointerAttention(nn.Module):
    """
    指针注意力：给定查询（解码器状态）与键（编码器输出），
    输出输入索引上的分布。
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        """
        计算指针注意力。
        
        参数：
            query: [batch, hidden_dim]
            keys: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]
        """
        batch_size, seq_len, hidden_dim = keys.size()
        
        # 第 1 步：投影键
        keys_proj = self.W_key(keys)
        
        # 第 2 步：投影查询
        query_proj = self.W_query(query).unsqueeze(1)
        
        # 第 3 步：能量计算（原仓库标为公式 1）
        # 使用原仓库所述论文定义中的 tanh 激活
        combined = torch.tanh(keys_proj + query_proj)
        
        # 第 4 步：映射为分数
        scores = self.v(combined).squeeze(-1)
        
        # 第 5 步：屏蔽（无放回采样）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 第 6 步：softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 第 7 步：选择
        pointer = torch.argmax(attention_weights, dim=-1)
        
        return pointer, attention_weights


def test_pointer_attention():
    print('验证指针注意力参考解答')
    print("-" * 30)
    
    batch_size = 4
    seq_len = 5
    hidden_dim = 32
    
    attention = SimplePointerAttention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    
    pointer, weights = attention(query, keys)
    assert pointer.shape == (batch_size,)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
    print('前向传播验证通过')
    
    mask = torch.ones(batch_size, seq_len)
    mask[:, -1] = 0
    _, weights_masked = attention(query, keys, mask=mask)
    assert weights_masked[0, -1] < 1e-7
    print('屏蔽验证通过')
    
    print('全部验证测试已完成')

if __name__ == "__main__":
    test_pointer_attention()
