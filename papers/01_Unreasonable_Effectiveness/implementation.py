"""
精简字符级 RNN 实现
基于 Andrej Karpathy 的 min-char-rnn.py
为便于教学提供详细注释

这是使用纯 NumPy 编写的完整、可运行的字符级 RNN 实现。
逐行解释实现，不隐藏计算细节。
"""

import numpy as np


class CharRNN:
    """
    精简字符级循环神经网络（RNN）。
    
    网络结构：
    - 输入：独热编码的字符
    - 隐藏层：采用 tanh 激活的 RNN
    - 输出：词表上的 softmax 概率分布
    
    参数：
    -----------
    vocab_size : int
        词表中的不同字符数量
    hidden_size : int
        隐藏状态大小（越大则容量越高，但速度越慢）
    learning_rate : float
        学习过程中参数更新的步长
    seq_length : int
        每次权重更新前处理的字符数
    """
    
    def __init__(self, vocab_size, hidden_size=100, learning_rate=1e-1, seq_length=25):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        
        # 使用较小的随机值初始化权重矩阵
        # 为什么取小值？过大的值可能导致梯度爆炸
        # 为什么随机初始化？打破对称性，否则所有神经元会学到相同内容
        
        # Wxh：输入（x）到隐藏状态（h）的权重矩阵
        # 形状：(hidden_size, vocab_size)
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        
        # Whh：上一隐藏状态（h）到当前隐藏状态（h）的权重矩阵
        # 形状：(hidden_size, hidden_size)
        # 这是“记忆”连接
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # Why：隐藏状态（h）到输出（y）的权重矩阵
        # 形状：(vocab_size, hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # 偏置（每个神经元一个）
        self.bh = np.zeros((hidden_size, 1))  # 隐藏层偏置
        self.by = np.zeros((vocab_size, 1))   # 输出层偏置
        
        # Adagrad（自适应学习率）的累积量
        # Adagrad 根据历史梯度为每个参数调整学习率
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
    def forward(self, inputs, hprev):
        """
        RNN 的前向传播。
        
        参数：
        -----------
        inputs : list of int
            字符索引列表（整数编码）
        hprev : numpy array
            上一隐藏状态（形状：hidden_size x 1）
            
        返回：
        --------
        xs : dict
            每个时间步的输入向量
        hs : dict
            每个时间步的隐藏状态
        ys : dict
            每个时间步的输出，即未归一化分数（logits）
        ps : dict
            每个时间步经过 softmax 后的概率
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)  # 初始隐藏状态
        
        # 依次处理序列中的字符
        for t, char_idx in enumerate(inputs):
            # 1. 将字符索引转换为独热向量
            # 例如，vocab_size=5 且 char_idx=2 时，xs[t] = [0, 0, 1, 0, 0]
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][char_idx] = 1
            
            # 2. 计算新的隐藏状态
            # h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
            #       ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
            # 处理输入          记住过去
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) +   # 当前输入的贡献
                np.dot(self.Whh, hs[t-1]) + # 上一状态（记忆）
                self.bh                      # 偏置
            )
            
            # 3. 计算输出分数（logits）
            # y_t = Why @ h_t + by
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # 4. 将分数转换为概率（softmax）
            # Softmax：e^x / sum(e^x) → 所有概率之和为 1
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
        return xs, hs, ys, ps
    
    def loss(self, ps, targets):
        """
        计算交叉熵损失。
        
        交叉熵衡量预测与真实目标之间的差异。
        损失越低，预测越准确
        
        参数：
        -----------
        ps : dict
            每个时间步的预测概率
        targets : list of int
            真实字符的索引
            
        返回：
        --------
        loss : float
            每个字符的平均损失
        """
        loss = 0
        for t, target in enumerate(targets):
            # 交叉熵：-log(正确字符的预测概率)
            # 若预测概率 = 1.0 → 损失 = 0（完全正确）
            # 若预测概率 = 0.01 → 损失 = -log(0.01) ≈ 4.6（较差）
            loss += -np.log(ps[t][target, 0])
        return loss / len(targets)
    
    def backward(self, xs, hs, ps, targets):
        """
        反向传播：使用随时间反向传播（BPTT）计算梯度。
        
        学习所需的梯度在这里计算，包括：
        - 每个权重对误差的影响有多大
        - 应向哪个方向调整权重以减小误差
        
        参数：
        -----------
        xs, hs, ps : dict（来自前向传播）
        targets : list of int
            真实字符的索引
            
        返回：
        --------
        gradients : dict
            各权重矩阵的梯度
        """
        # 初始化梯度累积量
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])  # 从后续时间步反向传来的梯度
        
        # 沿时间反向传播（从末尾到开头）
        for t in reversed(range(len(targets))):
            # 1. 计算损失相对于输出的梯度（dy）
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # softmax 与交叉熵组合后的梯度：正确类别为 p - 1，其余类别为 p
            
            # 2. 计算输出层权重梯度
            dWhy += np.dot(dy, hs[t].T)  # 通过矩阵乘法传播梯度
            dby += dy
            
            # 3. 梯度传回隐藏状态
            dh = np.dot(self.Why.T, dy) + dhnext  # 来自输出的梯度 + 来自后续时间步的梯度
            
            # 4. 梯度经过 tanh
            # tanh'(x) = 1 - tanh²(x)
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            # 5. 计算隐藏层权重梯度
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            
            # 6. 将梯度传到前一时间步
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # 裁剪梯度以防止梯度爆炸
        # 若不裁剪，梯度可能急剧增大并导致训练发散
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_parameters(self, gradients):
        """
        使用 Adagrad 更新权重。
        
        Adagrad 为每个参数自适应调整学习率：
        - 历史梯度平方累积量较大的参数，有效学习率较小
        - 历史梯度平方累积量较小的参数，有效学习率较大
        这有助于减少振荡并加快收敛。
        
        参数：
        -----------
        gradients : tuple
            (dWxh, dWhh, dWhy, dbh, dby)
        """
        dWxh, dWhh, dWhy, dbh, dby = gradients
        
        # 更新累积量（梯度平方和）
        self.mWxh += dWxh * dWxh
        self.mWhh += dWhh * dWhh
        self.mWhy += dWhy * dWhy
        self.mbh += dbh * dbh
        self.mby += dby * dby
        
        # 更新参数（Adagrad 公式）
        # parameter -= learning_rate * gradient / sqrt(memory + eps)
        # eps 用于防止除零
        self.Wxh -= self.learning_rate * dWxh / np.sqrt(self.mWxh + 1e-8)
        self.Whh -= self.learning_rate * dWhh / np.sqrt(self.mWhh + 1e-8)
        self.Why -= self.learning_rate * dWhy / np.sqrt(self.mWhy + 1e-8)
        self.bh -= self.learning_rate * dbh / np.sqrt(self.mbh + 1e-8)
        self.by -= self.learning_rate * dby / np.sqrt(self.mby + 1e-8)
    
    def sample(self, h, seed_idx, n_chars, temperature=1.0):
        """
        从模型的概率分布采样以生成文本。
        
        参数：
        -----------
        h : numpy array
            初始隐藏状态
        seed_idx : int
            起始字符索引
        n_chars : int
            要生成的字符数
        temperature : float
            控制随机性（越低越保守，越高越随机）
            1.0 = 使用原始概率
            0.5 = 使分布更尖锐（更保守）
            2.0 = 使分布更平坦（更随机）
            
        返回：
        --------
        indices : list of int
            生成的字符索引
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for _ in range(n_chars):
            # 前向传播（与训练相同，但只有一个时间步）
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # 应用温度
            # 温度控制随机性：
            # - 温度更高 → 分布更均匀（更多样、更随机）
            # - 温度更低 → 分布更尖锐（更接近确定性选择）
            y = y / temperature
            p = np.exp(y) / np.sum(np.exp(y))
            
            # 从概率分布中采样
            # 这会引入随机性，比始终选择最大概率项更多样
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # 将采样得到的字符用作下一步输入
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            
            indices.append(idx)
        
        return indices


def prepare_data(text):
    """
    将文本转换为整数索引。
    
    返回：
    --------
    char_to_idx : dict
        字符到索引的映射
    idx_to_char : dict
        索引到字符的映射
    data : list of int
        以索引列表表示的文本
    """
    chars = sorted(list(set(text)))  # 不同的字符
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_idx[ch] for ch in text]
    return char_to_idx, idx_to_char, data


if __name__ == "__main__":
    # 演示：使用微型样例训练
    text = "hello world, this is a minimal RNN implementation. " * 10
    char_to_idx, idx_to_char, data = prepare_data(text)
    
    print(f"词表大小：{len(char_to_idx)}")
    print(f"数据长度：{len(data)} 个字符")
    print(f"样本：{text[:50]}...")
    
    # 初始化模型
    model = CharRNN(
        vocab_size=len(char_to_idx),
        hidden_size=100,
        learning_rate=1e-1,
        seq_length=25
    )
    
    print("\n正在训练……")
    print("（运行 train_minimal.py 进行完整训练并查看进度）")
