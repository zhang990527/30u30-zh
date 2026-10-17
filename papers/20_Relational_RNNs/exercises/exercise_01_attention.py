import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算缩放点积注意力。
    
    参数：
        q（torch.Tensor）：查询 (batch, heads, slots, head_dim)
        k（torch.Tensor）：键 (batch, heads, slots, head_dim)
        v（torch.Tensor）：值 (batch, heads, slots, head_dim)
        mask（torch.Tensor）：可选掩码 (batch, 1, 1, slots)
        
    返回：
        output (torch.Tensor): (batch, heads, slots, head_dim)
        attention_weights (torch.Tensor): (batch, heads, slots, slots)
    """
    d_k = q.size(-1)
    
    # TODO：计算注意力分数
    # 提示：matmul(q, k.transpose(-2, -1)) / sqrt(d_k)
    scores = # 在此填写代码
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # TODO：应用 softmax
    attn_weights = # 在此填写代码
    
    # TODO：对值加权求和
    output = # 在此填写代码
    
    return output, attn_weights

def test_attention():
    print('正在测试缩放点积注意力……')
    batch, heads, slots, dim = 2, 4, 8, 16
    q = torch.randn(batch, heads, slots, dim)
    k = torch.randn(batch, heads, slots, dim)
    v = torch.randn(batch, heads, slots, dim)
    
    out, weights = scaled_dot_product_attention(q, k, v)
    
    assert out.shape == (batch, heads, slots, dim), f"预期 {(batch, heads, slots, dim)}，实际得到 {out.shape}"
    assert weights.shape == (batch, heads, slots, slots), f"预期 {(batch, heads, slots, slots)}，实际得到 {weights.shape}"
    
    print('测试通过！')

if __name__ == "__main__":
    test_attention()
