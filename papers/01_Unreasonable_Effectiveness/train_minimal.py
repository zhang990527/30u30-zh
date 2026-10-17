"""
字符级 RNN 的精简训练脚本
使用自己的文本训练并生成样本。

用法：
    python train_minimal.py --data path/to/text.txt --epochs 100
    python train_minimal.py --generate --checkpoint model.pkl
"""

import numpy as np
import argparse
import pickle
import time
from implementation import CharRNN, prepare_data


def train(model, data, char_to_idx, idx_to_char, epochs=100, print_every=10):
    """
    在给定数据上训练 RNN。
    
    参数：
    -----------
    model : CharRNN
        待训练模型
    data : list of int
        以字符索引表示的训练数据
    char_to_idx, idx_to_char : dict
        字符映射
    epochs : int
        原说明称为数据遍历轮数；当前实现实际为更新次数
    print_every : int
        每 N 次迭代输出一次样本
    """
    n = 0  # 当前数据位置
    p = 0  # 迭代计数器
    hprev = np.zeros((model.hidden_size, 1))  # 初始隐藏状态
    
    smooth_loss = -np.log(1.0 / model.vocab_size) * model.seq_length  # 起始预期损失
    
    print(f"\n训练 {epochs} 次更新……")
    print(f"初始预期损失：{smooth_loss:.2f}")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 原注释称为每轮开始时重置；实际只在到达数据末尾时重置
        if n + model.seq_length + 1 >= len(data):
            n = 0
            hprev = np.zeros((model.hidden_size, 1))  # 重置隐藏状态
        
        # 准备输入序列与目标序列
        inputs = data[n:n + model.seq_length]
        targets = data[n + 1:n + model.seq_length + 1]
        
        # 定期从模型中采样
        if p % print_every == 0:
            # 生成样本文本
            sample_idx = model.sample(hprev, inputs[0], 200, temperature=0.8)
            sample_text = ''.join([idx_to_char[idx] for idx in sample_idx])
            
            elapsed = time.time() - start_time
            print(f"\n--- 迭代 {p}（Epoch 参数计数 {epoch}）| 损失：{smooth_loss:.2f} | 耗时：{elapsed:.1f}s ---")
            print(f"样本：\n{sample_text}\n")
        
        # 前向传播
        xs, hs, ys, ps = model.forward(inputs, hprev)
        
        # 计算损失
        loss = model.loss(ps, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001  # 指数移动平均
        
        # 反向传播（计算梯度）
        gradients = model.backward(xs, hs, ps, targets)
        
        # 更新参数
        model.update_parameters(gradients)
        
        # 更新隐藏状态（语义上截断历史梯度；这里是 NumPy 实现，没有自动求导图）
        hprev = hs[len(inputs) - 1]
        
        # 移动到下一段序列
        n += model.seq_length
        p += 1
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"最终损失：{smooth_loss:.2f}")
    
    return model, hprev


def generate(model, idx_to_char, seed_text, n_chars=500, temperature=0.8):
    """
    Generate text from trained model.
    
    参数：
    -----------
    model : CharRNN
        已训练的模型
    idx_to_char : dict
        索引到字符的映射
    seed_text : str
        起始文本
    n_chars : int
        要生成的字符数
    temperature : float
        采样温度（0.5 较保守，1.5 更多样）
    """
    print(f"\n正在生成 {n_chars} 个字符，温度为 {temperature}...")
    print(f"起始文本：'{seed_text}'")
    print("=" * 60)
    
    # 将起始文本转换为索引
    char_to_idx = {ch: i for i, ch in idx_to_char.items()}
    seed_indices = [char_to_idx[ch] for ch in seed_text if ch in char_to_idx]
    
    if not seed_indices:
        print("错误：起始文本中没有词表内的字符")
        return
    
    # 初始化隐藏状态
    h = np.zeros((model.hidden_size, 1))
    
    # 使用起始文本预热隐藏状态
    for idx in seed_indices[:-1]:
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1
        h = np.tanh(np.dot(model.Wxh, x) + np.dot(model.Whh, h) + model.bh)
    
    # 生成
    sample_idx = model.sample(h, seed_indices[-1], n_chars, temperature)
    generated = ''.join([idx_to_char[idx] for idx in sample_idx])
    
    print(f"{seed_text}{generated}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='训练字符级 RNN')
    parser.add_argument('--data', type=str, default='data/tiny_shakespeare.txt',
                        help='文本文件路径')
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='隐藏层大小')
    parser.add_argument('--learning-rate', type=float, default=1e-1,
                        help='学习率')
    parser.add_argument('--seq-length', type=int, default=25,
                        help='BPTT 展开序列长度')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练更新次数（保留原参数名 epochs）')
    parser.add_argument('--checkpoint', type=str, default='model.pkl',
                        help='模型保存／加载路径')
    parser.add_argument('--generate', action='store_true',
                        help='从已保存的模型生成文本')
    parser.add_argument('--seed', type=str, default='The ',
                        help='生成时的起始文本')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度')
    parser.add_argument('--gen-length', type=int, default=500,
                        help='要生成的字符数')
    
    args = parser.parse_args()
    
    if args.generate:
        # 加载模型并生成文本
        print("正在加载模型：", args.checkpoint)
        with open(args.checkpoint, 'rb') as f:
            model, char_to_idx, idx_to_char = pickle.load(f)
        
        generate(model, idx_to_char, args.seed, args.gen_length, args.temperature)
    
    else:
        # 加载数据
        print(f"正在加载数据：{args.data}...")
        try:
            with open(args.data, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"错误：找不到 {args.data}")
            print("正在创建示例数据……")
            text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.""" * 10
        
        print(f"已加载数据：{len(text)} 个字符")
        
        # 准备数据
        char_to_idx, idx_to_char, data = prepare_data(text)
        print(f"词表：{len(char_to_idx)} 个不同字符")
        print(f"样本：{text[:100]}...")
        
        # 初始化模型
        model = CharRNN(
            vocab_size=len(char_to_idx),
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            seq_length=args.seq_length
        )
        
        # 训练
        model, hprev = train(model, data, char_to_idx, idx_to_char, 
                            epochs=args.epochs, print_every=10)
        
        # 保存模型
        print(f"\n正在保存模型到 {args.checkpoint}...")
        with open(args.checkpoint, 'wb') as f:
            pickle.dump((model, char_to_idx, idx_to_char), f)
        print("模型已保存！")
        
        # 生成样本
        print("\n" + "=" * 60)
        print("最终生成样本：")
        generate(model, idx_to_char, args.seed, args.gen_length, args.temperature)


if __name__ == "__main__":
    main()
