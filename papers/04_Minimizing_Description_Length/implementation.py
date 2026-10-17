"""
最小描述长度（Minimum Description Length，MDL）／贝叶斯神经网络实现
=========================================================================

含噪权重神经网络的完整教学实现。

这与普通 RNN（第 1 天）或 LSTM（第 2 天）有根本区别。
不为每个权重学习单个固定值（例如 w = 0.5），
而是为每个权重学习概率分布（例如 w ~ Gaussian(0.5, 0.1)）。

实现包含：
- 可学习均值（mu）
- 通过参数 rho 学习标准差 sigma（原文误称 sigma 为方差）
- 用重参数化技巧使梯度能够通过随机采样
- 组合误差（NLL）与复杂度（KL 散度）的损失函数

实现 Hinton 与 van Camp（1993）的核心思想：
“更简单的权重（更多噪声／不确定性）对应更好的压缩，可能改善泛化”；并非无条件保证

作者：30u30 项目
许可证：MIT
"""

import numpy as np
import pickle


def softplus(x):
    """
    Softplus 激活函数：log(1 + exp(x))。
    
    用于将无约束参数 rho 转换为正标准差 sigma。
    要求 sigma > 0；softplus 是保持正值的 ReLU 平滑近似。
    
    参数：
        x：输入数组（rho）
        
    返回：
        正值数组（sigma）
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    """Sigmoid activation for the derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ReLU activation for hidden layers."""
    return np.maximum(0, x)


class BayesianLinear:
    """
    以概率分布表示权重的单个全连接层。
    
    不直接使用 w，而是使用：
    - w_mu：权重均值
    - w_rho：控制离散程度（sigma = softplus(w_rho)，sigma 是标准差）
    
    前向传播：
    1. 由 rho 计算 sigma
    2. 采样噪声 (epsilon) ~ N(0, 1)
    3. 构造权重：w = mu + sigma * epsilon
    4. 计算输出：y = x @ w + b
    """
    
    def __init__(self, input_size, output_size, init_scale=0.1):
        self.input_size = input_size
        self.output_size = output_size
        
        # 随机初始化均值 mu，类似普通权重
        # 采用 He 初始化形式
        scale = np.sqrt(2.0 / input_size)
        self.w_mu = np.random.randn(input_size, output_size) * scale
        self.b_mu = np.zeros(output_size)
        
        # 初始化 Rho（不确定性控制参数）
        # 从较小的负值（例如 -3）开始，使
        # sigma = softplus(-3) 较小（约 0.05）。
        # 希望初始模型带有少量不确定性，而非完全随机。
        self.w_rho = np.ones((input_size, output_size)) * -3.0
        self.b_rho = np.ones(output_size) * -3.0
        
        # 存储梯度
        self.grads = {}
        
        # 反向传播缓存
        self.cache = None

    def sample_weights(self):
        """
        从学习得到的分布中采样实际权重。
        这是重参数化采样步骤；原文也称其为“比特回收”步骤，但采样不等于实际编码过程。
        """
        # 1. 将 rho 转换为 sigma（标准差）
        # sigma = log(1 + exp(rho))
        self.w_sigma = softplus(self.w_rho)
        self.b_sigma = softplus(self.b_rho)
        
        # 2. 从标准正态分布 N(0,1) 采样噪声 epsilon
        self.w_epsilon = np.random.randn(self.input_size, self.output_size)
        self.b_epsilon = np.random.randn(self.output_size)
        
        # 3. 构造含噪权重
        # w = mu + sigma * epsilon
        self.w_sample = self.w_mu + self.w_sigma * self.w_epsilon
        self.b_sample = self.b_mu + self.b_sigma * self.b_epsilon
        
        return self.w_sample, self.b_sample

    def forward(self, x):
        """
        使用采样权重进行前向传播。
        """
        # 为本次前向传播采样权重
        w, b = self.sample_weights()
        
        # 线性变换：y = xW + b
        out = x @ w + b
        
        # 缓存反向传播所需数据
        self.cache = (x, w, b)
        return out

    def kl_divergence(self):
        """
        计算本层的复杂度代价（KL 散度）。
        
        计算 KL( q(w) || p(w) )
        - q(w)：学习得到的近似后验 N(mu, sigma)，这里 sigma 指标准差
        - p(w)：先验 N(0, 1)，即标准高斯分布
        
        N(mu, sigma) 与 N(0, 1) 之间的 KL 公式（sigma 为标准差）：
        KL = 0.5 * sum( sigma^2 + mu^2 - 1 - log(sigma^2) )
        """
        # 权重的 KL
        kl_w = 0.5 * np.sum(
            self.w_sigma**2 + self.w_mu**2 - 1 - 2*np.log(self.w_sigma + 1e-8)
        )
        
        # 偏置的 KL
        kl_b = 0.5 * np.sum(
            self.b_sigma**2 + self.b_mu**2 - 1 - 2*np.log(self.b_sigma + 1e-8)
        )
        
        return kl_w + kl_b

    def backward(self, d_out):
        """
        计算 mu 与 rho 的梯度。
        
        需注意：梯度有两个来源：
        1. 数据损失（NLL）：通过采样权重反向传播
        2. 复杂度损失（KL）：直接作用于 mu 与 sigma
        
        参数：
            d_out：数据损失对本层输出的梯度（dL/dy）
        
        返回：
            d_x：对输入 x 的梯度，用于传给前一层
        """
        x, _, _ = self.cache
        
        # === 1. 来自数据损失（误差代价）的梯度 ===
        # 对重参数化公式使用链式法则推导。
        
        # 对采样权重的梯度（普通反向传播）
        # dL/dw = x.T @ dL/dy
        d_w_sample = x.T @ d_out
        d_b_sample = np.sum(d_out, axis=0)
        
        # 对均值 mu 的梯度
        # dL/dmu = dL/dw * dw/dmu = dL/dw * 1
        d_nll_w_mu = d_w_sample
        d_nll_b_mu = d_b_sample
        
        # 对 Rho 的梯度
        # dL/drho = dL/dw * dw/dsigma * dsigma/drho
        # dw/dsigma = epsilon
        # dsigma/drho = sigmoid(rho)
        d_nll_w_rho = d_w_sample * self.w_epsilon * sigmoid(self.w_rho)
        d_nll_b_rho = d_b_sample * self.b_epsilon * sigmoid(self.b_rho)
        
        # === 2. 来自 KL 散度（复杂度代价）的梯度 ===
        # KL = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
        
        # 对均值 mu 的梯度
        # dKL/dmu = mu
        d_kl_w_mu = self.w_mu
        d_kl_b_mu = self.b_mu
        
        # 对 Rho 的梯度
        # dKL/drho = dKL/dsigma * dsigma/drho
        # dKL/dsigma = sigma - 1/sigma
        d_kl_w_sigma = self.w_sigma - (1.0 / (self.w_sigma + 1e-8))
        d_kl_b_sigma = self.b_sigma - (1.0 / (self.b_sigma + 1e-8))
        
        d_kl_w_rho = d_kl_w_sigma * sigmoid(self.w_rho)
        d_kl_b_rho = d_kl_b_sigma * sigmoid(self.b_rho)
        
        # === 3. 合并与存储 ===
        # 加权系数 beta 通常由优化器处理，这里存储原始梯度。
        # 将 NLL 和 KL 梯度分开，供训练循环加权。
        
        self.grads = {
            'nll_w_mu': d_nll_w_mu, 'kl_w_mu': d_kl_w_mu,
            'nll_b_mu': d_nll_b_mu, 'kl_b_mu': d_kl_b_mu,
            'nll_w_rho': d_nll_w_rho, 'kl_w_rho': d_kl_w_rho,
            'nll_b_rho': d_nll_b_rho, 'kl_b_rho': d_kl_b_rho
        }
        
        # 对输入 x 的梯度，向前一层传递
        d_x = d_out @ self.w_sample.T
        return d_x


class MDLNetwork:
    """
    简单的两层贝叶斯神经网络。
    结构：输入 -> BayesianLinear -> ReLU -> BayesianLinear -> 输出
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 创建层
        self.layer1 = BayesianLinear(input_size, hidden_size)
        self.layer2 = BayesianLinear(hidden_size, output_size)
        
    def forward(self, x):
        """
        前向传播。
        注意：由于会采样权重，对同一 x 调用两次
        将产生略有不同的结果，这体现了不确定性。
        """
        self.h = relu(self.layer1.forward(x))
        out = self.layer2.forward(self.h)
        return out
    
    def total_kl(self):
        """Total Complexity Cost (KL) of the entire network."""
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()
    
    def backward(self, d_loss):
        """
        反向传播。
        参数：
            d_loss：数据损失（MSE/NLL）对网络输出的梯度
        """
        # 通过第 2 层反向传播
        d_h = self.layer2.backward(d_loss)
        
        # 通过 ReLU 反向传播
        d_h[self.h <= 0] = 0
        
        # 通过第 1 层反向传播
        self.layer1.backward(d_h)
        
    def update_weights(self, learning_rate, kl_weight):
        """
        使用随机梯度下降（SGD）更新参数。
        
        总梯度 = Data_Grad + (kl_weight * KL_Grad)
        """
        for layer in [self.layer1, self.layer2]:
            # 更新 Mu（均值）
            layer.w_mu -= learning_rate * (layer.grads['nll_w_mu'] + kl_weight * layer.grads['kl_w_mu'])
            layer.b_mu -= learning_rate * (layer.grads['nll_b_mu'] + kl_weight * layer.grads['kl_b_mu'])
            
            # 更新 Rho（控制标准差的参数，原文误称为方差）
            layer.w_rho -= learning_rate * (layer.grads['nll_w_rho'] + kl_weight * layer.grads['kl_w_rho'])
            layer.b_rho -= learning_rate * (layer.grads['nll_b_rho'] + kl_weight * layer.grads['kl_b_rho'])

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        运行模型 N 次，得到预测分布的采样近似。
        
        返回：
            mean_pred：预测均值
            std_pred：预测标准差（不确定性）
        """
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x))
            
        preds = np.array(preds) # 形状：(n_samples, batch_size, output_size)
        
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        
        return mean_pred, std_pred

    def save(self, filepath):
        """Save model parameters."""
        params = {
            'l1_w_mu': self.layer1.w_mu, 'l1_w_rho': self.layer1.w_rho,
            'l1_b_mu': self.layer1.b_mu, 'l1_b_rho': self.layer1.b_rho,
            'l2_w_mu': self.layer2.w_mu, 'l2_w_rho': self.layer2.w_rho,
            'l2_b_mu': self.layer2.b_mu, 'l2_b_rho': self.layer2.b_rho,
            'dims': (self.input_size, self.hidden_size, self.output_size)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"模型已保存至 {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model."""
        with open(filepath, 'rb') as f:
            p = pickle.load(f)
            
        model = cls(*p['dims'])
        
        model.layer1.w_mu, model.layer1.w_rho = p['l1_w_mu'], p['l1_w_rho']
        model.layer1.b_mu, model.layer1.b_rho = p['l1_b_mu'], p['l1_b_rho']
        model.layer2.w_mu, model.layer2.w_rho = p['l2_w_mu'], p['l2_w_rho']
        model.layer2.b_mu, model.layer2.b_rho = p['l2_b_mu'], p['l2_b_rho']
        
        print(f"模型已从以下路径加载：{filepath}")
        return model


if __name__ == "__main__":
    print('MDL／贝叶斯网络实现')
    print("=" * 60)
    print('\n本文件包含 BayesianLinear 与 MDLNetwork 类。')
    print('使用重参数化技巧实现含噪权重。')
    print('\n关键组件：')
    print('  - BayesianLinear：权重服从 N(mu, sigma) 的层，sigma 为标准差')
    print('  - sample_weights()：添加噪声，执行重参数化采样（不等同于实际比特回收编码）')
    print('  - kl_divergence()：计算复杂度代价')
    print('\n原提示使用 train_mdl.py；本目录实际训练入口为 train_minimal.py。')