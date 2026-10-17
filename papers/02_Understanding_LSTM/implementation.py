"""
使用 NumPy 实现长短期记忆（Long Short-Term Memory，LSTM）网络
=====================================================

完整的教学用 LSTM 网络实现。

与第 1 天的基本 RNN 相比，这里更复杂，因为增加了：
- 4 组权重矩阵（遗忘门、输入门、候选细胞状态、输出门）
- 与隐藏状态并行的细胞状态
- 更复杂的反向传播

这些额外机制能够缓解梯度消失问题。

Author: 30u30 Project
License: MIT
"""

import numpy as np
import pickle


def sigmoid(x):
    """
    sigmoid 激活函数。
    
    将任意值映射到 (0, 1) 区间。
    用于门控，因为门值需要表示从“完全关闭”（0）到“完全开启”（1）的程度。
    
    参数：
        x：输入数组
        
    返回：
        取值介于 0 与 1 之间的数组
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    softmax 激活函数（数值稳定版本）。
    
    将未归一化分数（logits）转换为总和为 1 的概率。
    
    参数：
        x：输入的未归一化分数数组
        
    返回：
        概率分布
    """
    exp_x = np.exp(x - np.max(x))  # 减去最大值，以提高数值稳定性
    return exp_x / np.sum(exp_x)


class LSTM:
    """
    字符级 LSTM 语言模型。
    
    该 LSTM 学习预测序列中的下一个字符。
    它通过细胞状态与门控机制，能够比基本 RNN
    在更长时间范围内保留信息。
    
    架构：
        - 输入：独热编码的字符
        - 隐藏层：带遗忘门、输入门、输出门的 LSTM
        - 输出：在字符词表上计算 softmax
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.001):
        """
        使用随机权重初始化 LSTM。
        
        参数：
            vocab_size：词表中不重复字符的数量
            hidden_size：隐藏状态和细胞状态的维度
            seq_length：训练时展开的时间步数
            learning_rate：梯度下降步长（这里建议小于基本 RNN）
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # LSTM 有 4 组权重（遗忘门、输入门、候选细胞状态、输出门）
        # 各组都将 [hidden_state, input] 变换为相应的门值或候选状态
        # 原文称为 Xavier 初始化：按 sqrt(2 / (fan_in)) 缩放；命名差异见核对说明
        scale = np.sqrt(2.0 / (hidden_size + vocab_size))
        
        # 遗忘门：决定从细胞状态中遗忘什么
        self.Wf = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bf = np.ones((hidden_size, 1))  # 初始化为 1（初始倾向于保留信息）
        
        # 输入门：决定存储哪些新信息
        self.Wi = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        
        # 候选细胞状态：计算待加入细胞状态的候选值
        self.Wc = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        
        # 输出门：决定从细胞状态中输出什么
        self.Wo = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        
        # 输出层：将隐藏状态映射到词表维度
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
        
        # 用于自适应学习率的 Adagrad 累积量
        # 为每个参数记录梯度平方和
        self.mWf, self.mWi, self.mWc, self.mWo = [np.zeros_like(W) for W in [self.Wf, self.Wi, self.Wc, self.Wo]]
        self.mbf, self.mbi, self.mbc, self.mbo = [np.zeros_like(b) for b in [self.bf, self.bi, self.bc, self.bo]]
        self.mWhy = np.zeros_like(self.Why)
        self.mby = np.zeros_like(self.by)
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        对输入序列执行 LSTM 前向传播。

        处理 seq_length 个字符，并计算：
        1. 门激活值（遗忘门、输入门、输出门）
        2. 细胞状态更新
        3. 隐藏状态更新
        4. 输出预测
        5. 损失

        参数：
            inputs：字符索引列表（长度 = seq_length）
            targets：目标字符索引列表（希望预测的字符）
            h_prev：上一隐藏状态（hidden_size × 1）
            C_prev：上一细胞状态（hidden_size × 1）
            
        返回：
            loss：当前序列的交叉熵损失总和
            h_last：最终隐藏状态（传给下一段序列）
            C_last：最终细胞状态（传给下一段序列）
            cache：反向传播所需的所有中间值
        """
        # 存储反向传播需要的中间值
        xs, hs, Cs, ys, ps = {}, {}, {}, {}, {}
        fs, i_s, c_cands, os, zs = {}, {}, {}, {}, {}  # 门激活值
        
        # 用上一状态初始化
        hs[-1] = np.copy(h_prev)
        Cs[-1] = np.copy(C_prev)
        
        loss = 0
        
        # 沿时间执行前向传播
        for t in range(len(inputs)):
            # === 步骤 1：准备输入 ===
            # 将字符索引转换为独热向量
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # 拼接上一隐藏状态与当前输入
            # 所有门均基于这个拼接向量作出决策
            zs[t] = np.vstack([hs[t-1], xs[t]])  # 形状：(hidden_size + vocab_size, 1)
            
            # === 步骤 2：遗忘门 ===
            # “应从细胞状态中遗忘什么？”
            # f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
            # 输出：0（全部遗忘）与 1（全部保留）之间的值
            fs[t] = sigmoid(self.Wf @ zs[t] + self.bf)
            
            # === 步骤 3：输入门与候选细胞状态 ===
            # “应向细胞状态加入哪些新信息？”
            # i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)    # 允许写入多少
            # ~C_t = tanh(W_c · [h_{t-1}, x_t] + b_c)      # 候选值
            i_s[t] = sigmoid(self.Wi @ zs[t] + self.bi)
            c_cands[t] = np.tanh(self.Wc @ zs[t] + self.bc)
            
            # === 步骤 4：更新细胞状态 ===
            # “更新长期记忆”
            # C_t = f_t ⊙ C_{t-1} + i_t ⊙ ~C_t
            # 关键：令 f_t ≈ 1，可以沿这条路径保留信息
            Cs[t] = fs[t] * Cs[t-1] + i_s[t] * c_cands[t]
            
            # === 步骤 5：输出门 ===
            # “应根据细胞状态输出什么？”
            # o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
            os[t] = sigmoid(self.Wo @ zs[t] + self.bo)
            
            # === 步骤 6：隐藏状态 ===
            # “由细胞状态计算隐藏状态”
            # h_t = o_t ⊙ tanh(C_t)
            hs[t] = os[t] * np.tanh(Cs[t])
            
            # === 步骤 7：输出层 ===
            # 将隐藏状态映射到词表维度
            ys[t] = self.Why @ hs[t] + self.by
            
            # === 步骤 8：softmax ===
            # 转换为概率分布
            ps[t] = softmax(ys[t])
            
            # === 步骤 9：损失 ===
            # 交叉熵损失：-log(正确字符的概率)
            loss += -np.log(ps[t][targets[t], 0])
        
        # 缓存所有反向传播所需信息
        cache = {
            'xs': xs, 'hs': hs, 'Cs': Cs, 'ys': ys, 'ps': ps,
            'fs': fs, 'i_s': i_s, 'c_cands': c_cands, 'os': os, 'zs': zs
        }
        
        return loss, hs[len(inputs)-1], Cs[len(inputs)-1], cache
    
    def backward(self, inputs, targets, cache):
        """
        反向传播：通过随时间反向传播（BPTT）计算梯度。
        
        比基本 RNN 更复杂，因为需要：
        1. 计算细胞状态梯度（除隐藏状态梯度外）
        2. 对四组变换反向传播
        3. 更仔细地管理梯度流
        
        细胞状态提供了一条更容易传递梯度的路径。
        
        参数：
            inputs：字符索引列表
            targets：目标字符索引列表
            cache：前向传播产生的中间值
            
        返回：
            所有参数的梯度字典
        """
        xs = cache['xs']
        hs = cache['hs']
        Cs = cache['Cs']
        ps = cache['ps']
        fs = cache['fs']
        i_s = cache['i_s']
        c_cands = cache['c_cands']
        os = cache['os']
        zs = cache['zs']
        
        # 将梯度初始化为零
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dWhy = np.zeros_like(self.Why)
        
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dby = np.zeros_like(self.by)
        
        # 累积沿时间反向传递的梯度
        dh_next = np.zeros_like(hs[0])
        dC_next = np.zeros_like(Cs[0])
        
        # 按时间的逆序进行反向传播
        for t in reversed(range(len(inputs))):
            # === 输出层梯度 ===
            # 损失对输出分数的梯度（softmax + 交叉熵）
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # 这就是 softmax 与交叉熵结合后的简洁导数
            
            dWhy += dy @ hs[t].T
            dby += dy
            
            # === 反向传播到隐藏状态 ===
            # 梯度来源：
            # 1. 输出层（Why.T @ dy）
            # 2. 下一时间步（dh_next）
            dh = self.Why.T @ dy + dh_next
            
            # === 输出门梯度 ===
            # h_t = o_t ⊙ tanh(C_t)
            # ∂L/∂o_t = ∂L/∂h_t ⊙ tanh(C_t)
            do = dh * np.tanh(Cs[t])
            # 穿过 sigmoid 反向传播：σ'(x) = σ(x) * (1 - σ(x))
            do_input = do * os[t] * (1 - os[t])
            
            # === 细胞状态梯度 ===
            # 梯度来源：
            # 1. 隐藏状态：dh * o_t * tanh'(C_t)
            # 2. 下一时间步：dC_next
            dC = dh * os[t] * (1 - np.tanh(Cs[t])**2) + dC_next
            
            # === 遗忘门梯度 ===
            # C_t = f_t ⊙ C_{t-1} + ...
            # ∂L/∂f_t = ∂L/∂C_t ⊙ C_{t-1}
            df = dC * Cs[t-1]
            df_input = df * fs[t] * (1 - fs[t])
            
            # === 输入门梯度 ===
            # C_t = ... + i_t ⊙ ~C_t
            # ∂L/∂i_t = ∂L/∂C_t ⊙ ~C_t
            di = dC * c_cands[t]
            di_input = di * i_s[t] * (1 - i_s[t])
            
            # === 候选细胞状态梯度 ===
            # C_t = ... + i_t ⊙ ~C_t
            # ∂L/∂~C_t = ∂L/∂C_t ⊙ i_t
            dc_cand = dC * i_s[t]
            # 穿过 tanh 反向传播：tanh'(x) = 1 - tanh(x)^2
            dc_cand_input = dc_cand * (1 - c_cands[t]**2)
            
            # === 权重梯度（所有门与候选状态） ===
            # 各组变换的输入都是 z_t = [h_{t-1}, x_t]，因此：
            # ∂L/∂W = ∂L/∂(W·z) @ z^T
            dWf += df_input @ zs[t].T
            dWi += di_input @ zs[t].T
            dWc += dc_cand_input @ zs[t].T
            dWo += do_input @ zs[t].T
            
            # === 偏置梯度 ===
            dbf += df_input
            dbi += di_input
            dbc += dc_cand_input
            dbo += do_input
            
            # === 反向传播到 [h_{t-1}, x_t] ===
            dz = (self.Wf.T @ df_input + 
                  self.Wi.T @ di_input +
                  self.Wc.T @ dc_cand_input +
                  self.Wo.T @ do_input)
            
            # 拆分为 h_{t-1} 与 x_t 的梯度
            dh_next = dz[:self.hidden_size]
            # dx = dz[self.hidden_size:]  # 这里无需计算该梯度
            
            # === 传向上一时间步的细胞状态梯度 ===
            # 关键：梯度通过遗忘门沿细胞状态的直接路径传递
            # dC_{t-1} = ∂L/∂C_t ⊙ f_t
            # 若 f_t ≈ 1，梯度就容易沿这条路径保留
            dC_next = dC * fs[t]
        
        # 裁剪梯度，防止爆炸
        # （LSTM 缓解梯度消失，但仍可能梯度爆炸）
        for grad in [dWf, dWi, dWc, dWo, dWhy, dbf, dbi, dbc, dbo, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return {
            'Wf': dWf, 'Wi': dWi, 'Wc': dWc, 'Wo': dWo, 'Why': dWhy,
            'bf': dbf, 'bi': dbi, 'bc': dbc, 'bo': dbo, 'by': dby
        }
    
    def update_weights(self, grads):
        """
        使用 Adagrad 优化器更新权重。
        
        Adagrad 根据历史梯度，为每个参数
        自适应调整学习率。梯度较大的参数
        获得更小的有效学习率。
        
        参数：
            grads：反向传播得到的梯度字典
        """
        params = {
            'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo, 'Why': self.Why,
            'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo, 'by': self.by
        }
        
        memories = {
            'Wf': self.mWf, 'Wi': self.mWi, 'Wc': self.mWc, 'Wo': self.mWo, 'Why': self.mWhy,
            'bf': self.mbf, 'bi': self.mbi, 'bc': self.mbc, 'bo': self.mbo, 'by': self.mby
        }
        
        for key in params:
            # 累积梯度平方
            memories[key] += grads[key] * grads[key]
            
            # 使用自适应学习率更新
            # learning_rate / sqrt(梯度平方和 + epsilon)
            params[key] -= self.learning_rate * grads[key] / np.sqrt(memories[key] + 1e-8)
    
    def sample(self, h, C, seed_idx, n, temperature=1.0):
        """
        通过从 LSTM 采样来生成文本。

        参数：
            h：初始隐藏状态（hidden_size × 1）
            C：初始细胞状态（hidden_size × 1）
            seed_idx：起始字符的索引
            n：要生成的字符数
            temperature：控制随机性，越高越随机
            
        返回：
            生成的字符索引列表
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for t in range(n):
            # 拼接 h 与 x
            z = np.vstack([h, x])
            
            # 通过 LSTM 单元执行前向传播
            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            c_cand = np.tanh(self.Wc @ z + self.bc)
            C = f * C + i * c_cand
            o = sigmoid(self.Wo @ z + self.bo)
            h = o * np.tanh(C)
            
            # 输出层
            y = self.Why @ h + self.by
            
            # 应用温度
            y = y / temperature
            
            # Softmax
            p = softmax(y)
            
            # 从分布中采样
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # 更新下一步的输入
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
        
        return indices
    
    def save(self, filepath):
        """将模型参数保存到文件。"""
        params = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'learning_rate': self.learning_rate,
            'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo, 'Why': self.Why,
            'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo, 'by': self.by
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"模型已保存至 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """从文件加载模型参数。"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        model = cls(params['vocab_size'], params['hidden_size'], 
                   params['seq_length'], params['learning_rate'])
        
        model.Wf = params['Wf']
        model.Wi = params['Wi']
        model.Wc = params['Wc']
        model.Wo = params['Wo']
        model.Why = params['Why']
        model.bf = params['bf']
        model.bi = params['bi']
        model.bc = params['bc']
        model.bo = params['bo']
        model.by = params['by']
        
        print(f"模型已从以下路径加载：{filepath}")
        return model


if __name__ == "__main__":
    print('LSTM 实现')
    print("=" * 60)
    print('\n本文件提供 LSTM 类。')
    print('请使用 train_minimal.py 训练模型。')
    print('请使用 notebook.ipynb 进行交互式学习。')
    print('\n与基本 RNN 的主要区别：')
    print('  - 隐藏状态（h）之外还有细胞状态（C）')
    print('  - 四组权重矩阵：遗忘门、输入门、候选细胞状态、输出门')
    print('  - 通过门控控制信息流')
    print('  - 梯度更容易通过细胞状态传递')
