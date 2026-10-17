import torch

def masked_aggregation(g_out: torch.Tensor) -> torch.Tensor:
    """
    TODO：实现带掩码的聚合，忽略自身关系。
    
    某些关系任务只关注不同对象之间的关系
    （i != j）。本函数应在求和之前
    屏蔽关系矩阵的对角线元素。
    
    输入：
        g_out：形状为 (batch, N, N, hidden_dim) 的张量，表示 g(o_i, o_j)
        
    输出：
        aggregated：形状为 (batch, hidden_dim) 的张量 = sum_{i != j} g(o_i, o_j)
        
    约束：
        - 使用 torch.eye 创建对角线掩码。
        - 不使用 for 循环。
    """
    # 在此填写代码
    pass

def test_masking():
    batch, N, dim = 2, 3, 4
    # 创建对角线元素不同的张量
    # g_out[b, i, j]
    g_out = torch.ones(batch, N, N, dim)
    
    # 将对角线设为 100，便于发现是否误计入
    for i in range(N):
        g_out[:, i, i, :] = 100.0
        
    # 排除对角线后，按对象对计算预期总和
    # 总对象对 = N*N = 9，对角线 = 3，非对角线 = 6。
    # 沿 N,N 求和后，每个特征应为 6 * 1.0 = 6.0
    
    res = masked_aggregation(g_out)
    
    expected_value = (N * N - N) * 1.0
    
    if torch.allclose(res, torch.tensor(float(expected_value))):
        print('练习 6（B 部分）：掩码 [PASS]')
    else:
        print(f"练习 6（B 部分）：[FAIL] 预期 {expected_value}，实际得到 {res[0,0].item()}")
        print('是否已减去或屏蔽对角线（数值为 100）？')

if __name__ == "__main__":
    test_masking()
