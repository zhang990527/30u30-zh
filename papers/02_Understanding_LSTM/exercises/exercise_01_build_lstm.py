"""
练习 1：从零构建 LSTM
====================================

目标：实现含四组变换及反向传播的完整 LSTM 单元。

你的任务：
- 完成下方 TODO，补全 LSTM 实现
- 在简单序列上测试实现
- 与参考实现比较

学习目标：
1. 理解各门的数学原理
2. 观察细胞状态与隐藏状态如何交互
3. 学习梯度如何穿过 LSTM 反向传递
4. 理解 LSTM 为什么能够缓解梯度消失

用时：2-3 小时
难度：高
"""

import numpy as np


class LSTMFromScratch:
    """
    从零实现的长短期记忆（LSTM）网络。
    
    架构：
        - 遗忘门：f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
        - 输入门：i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
        - 候选细胞状态：C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
        - 输出门：o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
        - 细胞状态：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        - 隐藏状态：h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        使用随机权重初始化 LSTM。
        
        参数：
            input_size：输入向量维度
            hidden_size：LSTM 单元数
            output_size：输出向量维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # TODO 1：初始化遗忘门权重矩阵
        # 提示：形状应为 (hidden_size, hidden_size + input_size)
        # 原文称为 Xavier 初始化：np.random.randn() * 0.01；此固定缩放并非标准 Xavier 公式
        self.Wf = None  # TODO：初始化
        self.bf = None  # TODO：初始化为 np.ones(hidden_size)，思考为什么用全 1
        
        # TODO 2：初始化输入门权重矩阵
        self.Wi = None  # TODO：初始化
        self.bi = None  # TODO：初始化为 np.zeros(hidden_size)
        
        # TODO 3：初始化候选细胞状态权重矩阵
        self.Wc = None  # TODO：初始化
        self.bc = None  # TODO：初始化为 np.zeros(hidden_size)
        
        # TODO 4：初始化输出门权重矩阵
        self.Wo = None  # TODO：初始化
        self.bo = None  # TODO：初始化为 np.zeros(hidden_size)
        
        # TODO 5：初始化输出层权重
        self.Wy = None  # TODO：形状 (output_size, hidden_size)
        self.by = None  # TODO：形状 (output_size,)
        
    def sigmoid(self, x):
        """sigmoid 激活函数：σ(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        对序列执行 LSTM 前向传播。
        
        参数：
            inputs：输入索引列表（长度 T）
            targets：目标索引列表（长度 T）
            h_prev：上一隐藏状态（hidden_size,）
            C_prev：上一细胞状态（hidden_size,）
            
        返回：
            loss：序列的交叉熵损失
        """
        # 存储反向传播所需的中间值
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
            # 创建独热编码输入
            x = np.zeros(self.input_size)
            x[input_idx] = 1.0
            self.inputs_cache.append(x)
            
            # TODO 6：拼接上一隐藏状态与当前输入
            # 形状：(hidden_size + input_size,)
            concat = None  # TODO：实现
            
            # TODO 7：计算遗忘门
            # f_t = sigmoid(W_f · concat + b_f)
            f_t = None  # TODO：实现
            self.f_gates.append(f_t)
            
            # TODO 8：计算输入门
            # i_t = sigmoid(W_i · concat + b_i)
            i_t = None  # TODO：实现
            self.i_gates.append(i_t)
            
            # TODO 9：计算候选细胞状态
            # C̃_t = tanh(W_C · concat + b_C)
            C_tilde = None  # TODO：实现
            self.C_tilde_cache.append(C_tilde)
            
            # TODO 10：更新细胞状态
            # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            # 关键方程：加性更新为缓解梯度消失提供了路径
            C_t = None  # TODO：实现
            self.C_states.append(C_t)
            
            # TODO 11：计算输出门
            # o_t = sigmoid(W_o · concat + b_o)
            o_t = None  # TODO：实现
            self.o_gates.append(o_t)
            
            # TODO 12：计算隐藏状态
            # h_t = o_t ⊙ tanh(C_t)
            h_t = None  # TODO：实现
            self.h_states.append(h_t)
            
            # TODO 13：计算输出
            # y = W_y · h_t + b_y
            y = None  # TODO：实现
            
            # TODO 14：应用 softmax 得到概率
            # 提示：取指数后归一化
            probs = None  # TODO：实现 softmax
            self.y_probs.append(probs)
            
            # TODO 15：计算交叉熵损失
            # loss = -log(正确类别的概率)
            loss += None  # TODO：实现
            
        return loss
    
    def backward(self):
        """
        反向传播（随时间反向传播）。
        
        计算所有参数的梯度。
        
        返回：
            dh_next：从下一时间步传回的隐藏状态梯度
            dC_next：从下一时间步传回的细胞状态梯度
        """
        # 初始化梯度
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
        
        # 沿时间反向传播
        for t in reversed(range(len(self.inputs_cache))):
            # TODO 16：来自输出层的梯度
            dy = self.y_probs[t].copy()
            dy[targets[t]] -= 1  # softmax 与交叉熵结合后的梯度
            
            # TODO 17：输出权重的梯度
            dWy += None  # TODO：实现（外积）
            dby += None  # TODO：实现
            
            # TODO 18：隐藏状态的梯度
            dh = None  # TODO：来自输出层与下一时间步的梯度之和
            
            # TODO 19：输出门的梯度
            # dh = do ⊙ tanh(C_t) + （来自下一步的梯度）；原注释的导数符号不严谨，按链式法则核对
            # 沿 h_t = o_t ⊙ tanh(C_t) 反向传播
            do = None  # TODO：实现
            dbo += do
            
            # TODO 20：细胞状态的梯度
            # 来自：h_t = o_t ⊙ tanh(C_t)
            dC = None  # TODO：实现（包括下一步传回的 dC_next）
            
            # TODO 21：候选细胞状态的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            dC_tilde = None  # TODO：实现
            dbc += dC_tilde
            
            # TODO 22：输入门的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            di = None  # TODO：实现
            dbi += di
            
            # TODO 23：遗忘门的梯度
            # 来自：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            df = None  # TODO：实现
            dbf += df
            
            # TODO 24：上一细胞状态的梯度
            # 关键在于加性更新路径；其梯度仍要逐元素乘以遗忘门值
            dC_next = None  # TODO：实现（梯度通过遗忘门传递）
            
            # TODO 25：向权重矩阵反向传播
            concat = np.concatenate([self.h_states[t], self.inputs_cache[t]])
            dWo += None  # TODO：实现（外积）
            dWc += None  # TODO：实现
            dWi += None  # TODO：实现
            dWf += None  # TODO：实现
            
            # TODO 26：上一隐藏状态的梯度
            dh_next = None  # TODO：实现（汇总四组变换的贡献）
        
        # TODO 27：使用梯度裁剪防止爆炸
        # 提示：np.clip(grad, -5, 5)
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
        """使用梯度更新权重。"""
        # TODO 28：实现权重更新
        # w = w - learning_rate * dw
        self.Wf -= None  # TODO
        self.bf -= None  # TODO
        # ……继续更新所有参数


# ============================================================================
# 测试你的实现
# ============================================================================

def test_lstm():
    """在简单序列上测试 LSTM。"""
    print('正在测试 LSTM 实现……')
    print("=" * 60)
    
    # 简单数据：学习重复 "hello"
    text = "hellohellohello"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    hidden_size = 10
    
    # 创建 LSTM
    lstm = LSTMFromScratch(vocab_size, hidden_size, vocab_size)
    
    # 训练
    seq_length = 5
    learning_rate = 0.01
    
    h = np.zeros(hidden_size)
    C = np.zeros(hidden_size)
    
    for iteration in range(100):
        # 获取随机序列
        start = np.random.randint(0, len(text) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in text[start:start+seq_length]]
        targets = [char_to_idx[ch] for ch in text[start+1:start+seq_length+1]]
        
        # 前向传播
        loss = lstm.forward(inputs, targets, h, C)
        
        # 反向传播
        dh, dC = lstm.backward()
        
        # 更新
        lstm.update_weights(learning_rate)
        
        # 更新状态
        h = lstm.h_states[-1].copy()
        C = lstm.C_states[-1].copy()
        
        if iteration % 20 == 0:
            print(f"迭代 {iteration}：损失 = {loss:.4f}")
    
    print('\n如果损失下降，说明实现可能正确；仍需核对梯度。')
    print('请与 ../implementation.py 中的参考实现比较')


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print('说明：')
    print("1. 补全所有 TODO（搜索 'TODO'）")
    print('2. 运行本文件，测试你的实现')
    print('3. 损失应随迭代下降')
    print('4. 遇到困难时，参阅 ../solutions/solution_01_lstm_implementation.py（原文路径）')
    print("=" * 60 + "\n")
    
    # 准备好测试后取消注释
    # test_lstm()
