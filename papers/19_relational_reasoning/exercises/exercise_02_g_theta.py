import torch
import torch.nn as nn

class RelationFunction(nn.Module):
    """
    TODO：实现 g_theta MLP。
    
    如第 2 节所述，g_theta 是多层感知机，处理
    每个对象对，以及可选的问题嵌入。
    
    架构：
    - 输入：(2 * object_dim) + question_dim
    - 隐藏层：4 层，每层 256 个单元，使用 ReLU
    - 输出：256 个单元
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 在此填写代码
        pass

    def forward(self, x):
        # 在此填写代码
        pass

def test_relation_function():
    input_dim = 16
    hidden_dim = 256
    model = RelationFunction(input_dim, hidden_dim)
    
    # 测试随机输入
    x = torch.randn(10, input_dim)
    try:
        out = model(x)
        assert out.shape == (10, hidden_dim), f"预期 {(10, hidden_dim)}，实际得到 {out.shape}"
        print('练习 2（A 部分）：关系函数 [PASS]')
    except Exception as e:
        print(f"练习 2（A 部分）：[FAIL] {e}")

if __name__ == "__main__":
    test_relation_function()
