"""
练习 1 解答：从零构建 LSTM
================================================

原仓库将所有 TODO 填写后的解答。

将你的实现与此对比，检查理解。
"""

import numpy as np


class LSTMFromScratch:
    """带详细注释的完整 LSTM 实现。"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 解答 1：遗忘门权重
        # 遗忘门决定从细胞状态中移除什么
        # 用小随机值初始化（原文称 Xavier 初始化；固定 0.01 缩放并非标准 Xavier 公式）
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        # 偏置初始化为 1，使 LSTM 初始倾向于保留信息，这是重要的设置
        self.bf = np.ones(hidden_size)
        
        # 解答 2：输入门权重
        # 输入门决定添加哪些新信息
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bi = np.zeros(hidden_size)
        
        # 解答 3：候选细胞状态权重
        # 生成待加入细胞状态的新候选值
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bc = np.zeros(hidden_size)
        
        # 解答 4：输出门权重
        # 输出门决定暴露细胞状态中的哪些信息
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bo = np.zeros(hidden_size)
        
        # 解答 5：输出层权重
        # 将隐藏状态映射到输出词表
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros(output_size)
        
    def sigmoid(self, x):
        """sigmoid 激活：映射到 [0, 1] 范围，用于门控。"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        通过 LSTM 前向传播。
        
        逐步实现 LSTM 的核心方程。
        """
        # 存储反向传播所需数据
        self.inputs_cache = []
        self.h_states = [h_prev]
        self.C_states = [C_prev]
        self.f_gates = []
        self.i_gates = []
        self.C_tilde_cache = []
        self.o_gates = []
        self.y_probs = []
        
        loss = 0.0
        
        for t, (input_idx, target_idx) in enumerate(zip(inputs, targets)):
            # 对输入进行独热编码
            x = np.zeros(self.input_size)
            x[input_idx] = 1.0
            self.inputs_cache.append(x)
            
            # 解答 6：拼接 h_{t-1} 与 x_t
            # 作为所有门与候选变换的输入
            concat = np.concatenate([self.h_states[t], x])
            
            # 解答 7：遗忘门
            # 决定从细胞状态中丢弃什么
            # 输出位于 0（全部遗忘）与 1（全部保留）之间
            f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            self.f_gates.append(f_t)
            
            # 解答 8：输入门
            # 决定向细胞状态写入哪些新值
            # 输出位于 0（忽略）与 1（接受）之间
            i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            self.i_gates.append(i_t)
            
            # 解答 9：候选细胞状态
            # 生成新的候选值
            # 使用 tanh 而非 sigmoid，取值介于 -1 与 1 之间
            C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            self.C_tilde_cache.append(C_tilde)
            
            # 解答 10：更新细胞状态
            # 关键方程：细胞状态的加性更新
            # 注意：这里将保留项与写入项相加
            # C_t = （旧状态的保留部分）+（新候选值的写入部分）
            C_t = f_t * self.C_states[t] + i_t * C_tilde
            self.C_states.append(C_t)
            
            # 解答 11：输出门
            # 决定输出细胞状态中的哪些部分
            o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            self.o_gates.append(o_t)
            
            # 解答 12：隐藏状态
            # LSTM 单元的实际输出
            # h_t = （要暴露的比例）*（经 tanh 压缩的细胞状态）
            h_t = o_t * np.tanh(C_t)
            self.h_states.append(h_t)
            
            # 解答 13：计算输出未归一化分数（logits）
            y = np.dot(self.Wy, h_t) + self.by
            
            # 解答 14：使用 softmax 得到概率
            # 减去最大值，以提高数值稳定性
            exp_y = np.exp(y - np.max(y))
            probs = exp_y / np.sum(exp_y)
            self.y_probs.append(probs)
            
            # 解答 15：交叉熵损失
            # -log(正确类别的概率)
            loss += -np.log(probs[target_idx] + 1e-8)  # 加入 epsilon，避免 log(0)
            
        return loss
    
    def backward(self):
        """
        反向传播（BPTT）。
        
        梯度在这里沿时间反向传递。
        注意细胞状态直接路径的梯度如何在门值接近 1 时保留。
        """
        # 将所有梯度初始化为零
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros(self.hidden_size)
        dC_next = np.zeros(self.hidden_size)
        
        # 按时间逆序处理
        T = len(self.inputs_cache)
        
        for t in reversed(range(T)):
            # 解答 16：softmax 与交叉熵结合后的梯度
            # 便捷性质：损失对 logits 的梯度 = probs - 正确类别的独热向量
            dy = self.y_probs[t].copy()
            dy[targets[t]] -= 1
            
            # 解答 17：输出权重梯度
            # dL/dW = dL/dy * dy/dW = dy * h_t^T（外积）
            dWy += np.outer(dy, self.h_states[t+1])
            dby += dy
            
            # 解答 18：隐藏状态的梯度
            # 有两个来源：
            # 1. 当前时间步的输出层
            # 2. 下一时间步（BPTT）
            dh = np.dot(self.Wy.T, dy) + dh_next
            
            # 解答 19：输出门的梯度
            # 来自：h_t = o_t ⊙ tanh(C_t)
            # dh = do ⊙ tanh(C_t) + o_t ⊙ d(tanh(C_t))
            # 计算：do = dh ⊙ tanh(C_t) ⊙ o_t ⊙ (1 - o_t)
            tanh_C = np.tanh(self.C_states[t+1])
            do = dh * tanh_C * self.o_gates[t] * (1 - self.o_gates[t])
            dbo += do
            
            # 解答 20：细胞状态的梯度
            # 来自：h_t = o_t ⊙ tanh(C_t)
            # 同时加入下一时间步传回的梯度
            dC = dh * self.o_gates[t] * (1 - tanh_C**2) + dC_next
            
            # 解答 21：候选细胞状态的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            # dC = i_t ⊙ dC̃_t
            dC_tilde = dC * self.i_gates[t] * (1 - self.C_tilde_cache[t]**2)
            dbc += dC_tilde
            
            # 解答 22：输入门的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            di = dC * self.C_tilde_cache[t] * self.i_gates[t] * (1 - self.i_gates[t])
            dbi += di
            
            # 解答 23：遗忘门的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            df = dC * self.C_states[t] * self.f_gates[t] * (1 - self.f_gates[t])
            dbf += df
            
            # 解答 24：上一细胞状态的梯度
            # 关键：直接路径的梯度逐元素乘以 f_t；只有门值接近 1 时才能近似保持
            # 这条直接路径有助于缓解梯度消失，但并非无条件保证
            dC_next = dC * self.f_gates[t]
            
            # 解答 25：权重矩阵梯度
            concat = np.concatenate([self.h_states[t], self.inputs_cache[t]])
            dWo += np.outer(do, concat)
            dWc += np.outer(dC_tilde, concat)
            dWi += np.outer(di, concat)
            dWf += np.outer(df, concat)
            
            # 解答 26：上一隐藏状态的梯度
            # 汇总四组变换的贡献，因为它们都使用 h_{t-1}
            d_concat = (np.dot(self.Wf.T, df) +
                       np.dot(self.Wi.T, di) +
                       np.dot(self.Wc.T, dC_tilde) +
                       np.dot(self.Wo.T, do))
            dh_next = d_concat[:self.hidden_size]
        
        # 解答 27：梯度裁剪，防止爆炸
        # 将所有梯度裁剪至 [-5, 5]
        for dparam in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 存储梯度
        self.dWf, self.dbf = dWf, dbf
        self.dWi, self.dbi = dWi, dbi
        self.dWc, self.dbc = dWc, dbc
        self.dWo, self.dbo = dWo, dbo
        self.dWy, self.dby = dWy, dby
        
        return dh_next, dC_next
    
    def update_weights(self, learning_rate):
        """解答 28：使用梯度更新所有权重。"""
        self.Wf -= learning_rate * self.dWf
        self.bf -= learning_rate * self.dbf
        self.Wi -= learning_rate * self.dWi
        self.bi -= learning_rate * self.dbi
        self.Wc -= learning_rate * self.dWc
        self.bc -= learning_rate * self.dbc
        self.Wo -= learning_rate * self.dWo
        self.bo -= learning_rate * self.dbo
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby


# ============================================================================
# 测试
# ============================================================================

def test_lstm():
    """测试完整的 LSTM 实现。"""
    print('正在测试 LSTM 实现……')
    print("=" * 60)
    
    # 简单任务：学习重复 "hello"
    text = "hellohellohellohellohello"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"词表大小：{chars}")
    print(f"文本长度：{len(text)}")
    
    vocab_size = len(chars)
    hidden_size = 16
    
    # 创建 LSTM
    lstm = LSTMFromScratch(vocab_size, hidden_size, vocab_size)
    print(f"\n已创建 LSTM：{vocab_size} 维输入，{hidden_size} 维隐藏状态，{vocab_size} 维输出")
    
    # 训练
    seq_length = 5
    learning_rate = 0.01
    num_iterations = 200
    
    h = np.zeros(hidden_size)
    C = np.zeros(hidden_size)
    
    losses = []
    
    print(f"\n正在训练，迭代次数为 {num_iterations} 次……")
    print("-" * 60)
    
    for iteration in range(num_iterations):
        # 随机序列
        start = np.random.randint(0, len(text) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in text[start:start+seq_length]]
        targets = [char_to_idx[ch] for ch in text[start+1:start+seq_length+1]]
        
        # 前向传播
        loss = lstm.forward(inputs, targets, h, C)
        losses.append(loss)
        
        # 反向传播
        dh, dC = lstm.backward()
        
        # 更新权重
        lstm.update_weights(learning_rate)
        
        # 更新状态（原文沿用“从计算图分离”的说法；这里是手工 NumPy 梯度）
        h = lstm.h_states[-1].copy()
        C = lstm.C_states[-1].copy()
        
        if iteration % 50 == 0:
            smooth_loss = np.mean(losses[-50:]) if len(losses) >= 50 else loss
            print(f"迭代 {iteration:3d}：损失 = {smooth_loss:.4f}")
    
    print("\n" + "=" * 60)
    print('训练完成！')
    print(f"初始损失：{losses[0]:.4f}")
    print(f"最终损失：{losses[-1]:.4f}")
    print(f"改善：{(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    if losses[-1] < losses[0] * 0.5:
        print('\n损失显著下降。')
        print('你的 LSTM 正在学习。')
    else:
        print('\n损失下降不明显，请检查实现。')


if __name__ == "__main__":
    print(__doc__)
    test_lstm()
