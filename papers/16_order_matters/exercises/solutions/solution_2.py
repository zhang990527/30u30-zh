"""
解答 2：顺序不变集合编码器
========================================

使用自注意力的集合编码器实现；原仓库将逐元素的置换等变性称为置换不变性，见 README。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    基于自注意力的集合编码器。
    原仓库称省略位置编码可确保输出对输入置换不变；
    此处应区分逐元素输出的置换等变性与聚合表示的置换不变性。
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        参数：
            x: [batch, set_size, input_dim]
        """
        # 第 1 步：嵌入
        x = self.embedding(x)
        
        # 第 2 步：自注意力（逐元素输出为置换等变）
        # 注意：此处不添加位置嵌入。
        attn_out, _ = self.self_attention(x, x, x)
        
        # 第 3 步：归一化与残差连接
        x = self.norm1(x + attn_out)
        
        # 第 4 步：前馈网络
        ff_out = self.ff(x)
        
        # 第 5 步：归一化与残差连接
        x = self.norm2(x + ff_out)
        
        return x


def test_order_invariance():
    print('验证顺序不变编码器')
    print("-" * 30)
    
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    encoder = SetEncoder(input_dim, hidden_dim)
    encoder.eval()
    
    points = torch.rand(1, set_size, input_dim)
    indices = torch.randperm(set_size)
    points_permuted = points[:, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    
    assert torch.allclose(agg1, agg2, atol=1e-5)
    print('已通过平均池化验证聚合表示的置换不变性。')
    print('全部验证测试已完成')

if __name__ == "__main__":
    test_order_invariance()
