"""
解答 2：构建关系函数（g_theta）
按 30u30 第 19 天的格式组织

依据：Santoro 等（2017）
"""

import torch
import torch.nn as nn

class RelationFunction(nn.Module):
    """
    RN 论文中描述的四层 MLP。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
