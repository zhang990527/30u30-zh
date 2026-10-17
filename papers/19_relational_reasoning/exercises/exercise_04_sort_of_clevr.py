import torch

def create_contextual_pairs(objects: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """
    TODO：实现带问题注入的配对逻辑。
    
    在 Sort-of-CLEVR 中（原仓库引用第 4.1 节），g_theta(o_i, o_j, q)
    取决于问题 q，需要将 q 广播到每个对象对。
    
    输入：
        objects: (batch_size, N, object_dim)
        question: (batch_size, question_dim)
        
    输出：
        pairs: (batch_size, N, N, 2*object_dim + question_dim)
        
    提示：
        1. 创建形状为 (B, N, N, 2*O) 的对象对 (o_i, o_j)
        2. 将问题 q 扩展为 (B, N, N, Q)
        3. 沿最后一维拼接。
    """
    # 在此填写代码
    pass

def test_contextual_pairs():
    # 准备
    batch, N, obj_dim, q_dim = 2, 3, 4, 5
    objects = torch.randn(batch, N, obj_dim)
    question = torch.randn(batch, q_dim)
    
    try:
        pairs = create_contextual_pairs(objects, question)
        
        # 检查形状
        expected_shape = (batch, N, N, 2 * obj_dim + q_dim)
        assert pairs.shape == expected_shape, f"预期 {expected_shape}，实际得到 {pairs.shape}"
        
        # 检查带问题的第一个对象对 (0,0) 的内容
        # pair[0,0,0] 应为 cat(avg_obj[0,0], avg_obj[0,0], q[0])
        # 此处只检查问题部分是否正确
        q_part = pairs[:, :, :, -q_dim:]
        
        # 检查问题向量是否正确广播
        # q_part[b, i, j] 应等于 question[b]
        diff = (q_part - question.view(batch, 1, 1, q_dim)).abs().max()
        assert diff < 1e-5, '问题向量未正确广播。'
        
        print('练习 4：带上下文的对象对 [PASS]')
        
    except Exception as e:
        print(f"练习 4：[FAIL] {e}")

if __name__ == "__main__":
    test_contextual_pairs()
