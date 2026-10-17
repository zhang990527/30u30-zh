"""
练习 2：顺序不变集合编码器
========================================

学习如何编码输入顺序任意、语义却
保持不变的集合。

概念：顺序不变编码器应满足 f(permute(X)) = f(X)，
对输入的任何置换都成立；逐元素输出则应满足置换等变关系，见 README。

要求：用不带位置向量的自注意力构建集合编码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    使用自注意力的集合编码器（不变性与等变性的区别见 README）。
    原仓库称不使用位置编码时，自注意力天然具有置换不变性；
    此处逐元素输出实际为置换等变，聚合表示才可能置换不变。
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # TODO: 用嵌入层将输入投影到 hidden_dim
        self.embedding = None 
        
        # TODO: 多头自注意力
        # 关键：不要添加位置编码。
        self.self_attention = None 
        
        # TODO: 前馈网络（hidden_dim -> hidden_dim*4 -> hidden_dim）
        self.ff = None 
        
        # TODO: 层归一化
        self.norm1 = None
        self.norm2 = None
        
    def forward(self, x):
        """
        对集合进行编码（逐元素输出应随输入置换而等变）。
        
        参数：
            x: [batch, set_size, input_dim]
            
        返回：
            encoded: [batch, set_size, hidden_dim]
        """
        # TODO: 第 1 步——嵌入输入
        x = None 
        
        # TODO: 第 2 步——自注意力
        attn_out, _ = None, None 
        
        # TODO: 第 3 步——残差连接与层归一化
        x = None 
        
        # TODO: 第 4 步——前馈网络
        ff_out = None 
        
        # TODO: 第 5 步——残差连接与层归一化
        x = None 
        
        return x


def test_order_invariance():
    """检查编码器的聚合表示是否满足置换不变性。"""
    print('测试顺序不变性')
    print("-" * 30)
    
    batch_size = 1
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=4)
    encoder.eval()
    
    # 创建原始集合
    points = torch.rand(batch_size, set_size, input_dim)
    
    # 创建置换后的集合
    indices = torch.randperm(set_size)
    points_permuted = points[:, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    # 检查输出向量的集合是否相同（忽略顺序）
    e1_sorted = torch.sort(encoded1[0].sum(dim=1))[0]
    e2_sorted = torch.sort(encoded2[0].sum(dim=1))[0]
    
    diff = (e1_sorted - e2_sorted).abs().mean()
    print(f"置换后的平均差异：{diff.item():.6f}")
    
    # 聚合表示（平均池化）应相同
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    diff_agg = (agg1 - agg2).abs().mean()
    print(f"平均池化表示的差异：{diff_agg.item():.6f}")

    print("-" * 30)

if __name__ == "__main__":
    test_order_invariance()
    
    print('练习 2 小结')
    print("-" * 30)
    print("""
Key concepts covered:
1. Permutation invariance: The encoder output depends only on the content of the set.
2. Self-attention: Mechanisms that allow elements to interact without assuming sequence order.
3. Layer Normalization & Residuals: Standard components for training stability in attention models.
    """)
