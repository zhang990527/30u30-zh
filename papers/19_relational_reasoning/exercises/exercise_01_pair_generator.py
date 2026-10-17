import torch
import torch.nn as nn

def generate_pairs_via_broadcasting(objects: torch.Tensor) -> torch.Tensor:
    """
    TODO：通过广播实现成对对象拼接。
    如第 2.1 节所述，RN 对全部对象对计算关系。
    
    输入：(batch, N, D)
    输出：(batch, N^2, 2*D)
    
    约束：不使用 Python for 循环，使用 unsqueeze、expand/repeat 与 cat。
    """
    # 在此填写代码
    pass

def test_pair_generator():
    batch, N, D = 2, 5, 8
    x = torch.randn(batch, N, D)
    
    pairs = generate_pairs_via_broadcasting(x)
    
    # 验证
    assert pairs.shape == (batch, N*N, 2*D), f"预期 {(batch, N*N, 2*D)}，实际得到 {pairs.shape}"
    
    # 检查第一个对象对是否确为 (o0, o0)
    first_pair = pairs[0, 0]
    expected_first = torch.cat([x[0, 0], x[0, 0]], dim=0)
    assert torch.allclose(first_pair, expected_first), '第一个对象对错误。'
    
    # 检查第二个对象对是否为 (o0, o1)
    second_pair = pairs[0, 1]
    expected_second = torch.cat([x[0, 0], x[0, 1]], dim=0)
    assert torch.allclose(second_pair, expected_second), '第二个对象对错误。'
    
    print('练习 1：[PASS]')

if __name__ == "__main__":
    test_pair_generator()
