"""
练习 1：基本指针注意力机制
==============================================

学习如何计算注意力分数并“指向”输入元素。

概念：给定解码器状态与编码器输出，计算
         当前最相关的输入元素。

【仓库补充：类比】像从菜单中点菜——每次根据当前偏好，
         即解码器状态，将注意力集中在某一项上。

你的任务：实现指针选择的核心注意力机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointerAttention(nn.Module):
    """
    基本指针注意力：给定查询（解码器状态），在键（编码器输出）上
    计算注意力，以选择某个元素。
    
    公式：scores = v^T * tanh(W_key * keys + W_query * query)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # TODO: 初始化三个线性层
        # 1. 将键（编码器输出）投影到 hidden_dim
        # 2. 将查询（解码器状态）投影到 hidden_dim
        # 3. 将组合后的表示投影为标量分数
        
        self.W_key = None  # TODO: 替换 None
        self.W_query = None  # TODO: 替换 None
        self.v = None  # TODO: 替换 None
    
    def forward(self, query, keys, mask=None):
        """
        计算指针注意力。
        
        参数：
            query: [batch, hidden_dim]——当前解码器状态
            keys: [batch, seq_len, hidden_dim]——编码器输出
            mask: [batch, seq_len]——0 表示忽略，1 表示有效
            
        返回：
            pointer: [batch]——所选元素的索引
            attention_weights: [batch, seq_len]——注意力分布
        """
        batch_size, seq_len, hidden_dim = keys.size()
        
        # TODO: 第 1 步——投影键
        # 形状：[batch, seq_len, hidden_dim]
        keys_proj = None 
        
        # TODO: 第 2 步——投影查询并扩展维度，使之与键匹配
        # 形状：[batch, 1, hidden_dim]
        query_proj = None 
        
        # TODO: 第 3 步——组合后应用 tanh 激活
        # 形状：[batch, seq_len, hidden_dim]
        combined = None 
        
        # TODO: 第 4 步——投影为标量分数
        # 形状：[batch, seq_len]
        scores = None 
        
        # TODO: 第 5 步——若提供掩码，则应用屏蔽
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # TODO: 第 6 步——用 softmax 计算注意力权重
        attention_weights = None 
        
        # TODO: 第 7 步——选择指针（argmax）
        pointer = None 
        
        return pointer, attention_weights


def test_pointer_attention():
    """测试指针注意力实现。"""
    print('测试指针注意力')
    print("-" * 30)
    
    batch_size = 4
    seq_len = 5
    hidden_dim = 32
    
    attention = SimplePointerAttention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 测试 1：基本前向传播
    print('测试 1：基本前向传播')
    pointer, weights = attention(query, keys)
    if pointer is not None:
        print(f"   指针形状：{pointer.shape}")
        print(f"   权重形状：{weights.shape}")
        print(f"   权重之和：{weights[0].sum().item():.4f}")
    
    # 测试 2：屏蔽
    print('测试 2：屏蔽')
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0
    pointer_masked, weights_masked = attention(query, keys, mask=mask)
    if weights_masked is not None:
        print(f"   最后 2 个权重：{weights_masked[0, -2:].tolist()}（应为 0.0）")
    
    # 测试 3：数值上的注意力集中程度
    print('测试 3：注意力应集中')
    keys_biased = torch.randn(batch_size, seq_len, hidden_dim)
    keys_biased[:, 0] = query
    pointer_biased, weights_biased = attention(query, keys_biased)
    if weights_biased is not None:
        print(f"   第一个权重：{weights_biased[0, 0].item():.4f}（应较高）")

    print("-" * 30)


if __name__ == "__main__":
    test_pointer_attention()
    
    print('练习 1 小结')
    print("-" * 30)
    print('\n涵盖的关键概念：\n1. 注意力分数：衡量查询与键的相关性。\n2. Softmax：将分数转换为概率分布。\n3. Argmax：选择最相关的索引。\n4. 屏蔽：确保忽略无效位置。\n    ')
