"""
关系网络（RN）实现：关系推理模块
按 30u30 第 19 天的格式组织

本模块实现 Santoro 等人（2017）描述的关系网络（RN）。
RN 通过特定的归纳偏置进行关系推理：
对集合中每个可能的对象对，
显式计算非线性函数。

主要功能：
1. 对象对生成：通过广播完成 N^2 配对（第 2.1 节）。
2. 坐标感知编码：空间任务可选拼接 (x, y)（第 3.1 节）。
3. 聚合器变体：支持 sum、mean、max，以展示计数偏置。
4. 关系瓶颈 g_theta：用于成对逻辑的共享 MLP。
5. 全局处理 f_phi：用于最终任务预测的 MLP。

参考资料：
- Santoro 等（2017）《用于关系推理的简单神经网络模块》（A Simple Neural Network Module for Relational Reasoning）
  https://arxiv.org/abs/1706.01427

作者：30u30 项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class RelationNetwork(nn.Module):
    """
    关系网络（RN）模块。
    
    架构（对应论文第 2.1 节）：
    RN(O) = f_phi(AGGREGATE_{i,j} g_theta(o_i, o_j, q))
    
    标准 RN 使用 sum 聚合器，因为它能保留
    计数所需的基数信息，而 mean、max 往往会丢失此信息。
    """
    
    def __init__(
        self, 
        object_dim: int, 
        relation_dim: int = 256, 
        output_dim: int = 10, 
        question_dim: Optional[int] = None,
        aggregator: str = 'sum',
        use_coordinates: bool = False,
        dropout_p: float = 0.5
    ):
        """
        初始化可配置的关系网络。

        参数：
            object_dim：每个输入对象向量的维度。
            relation_dim：g_theta 和 f_phi MLP 的隐藏维度。
            output_dim：最终输出维度，例如类别数。
            question_dim：（可选）问题／上下文向量维度。
            aggregator：关系聚合策略，取 sum、mean 或 max。
            use_coordinates：若为 True，期望输入为 (batch, N, D+2)，最后 2 维为 (x,y)。
            dropout_p：f_phi 中的随机失活概率（默认 0.5，原仓库称来自论文）。
        """
        super(RelationNetwork, self).__init__()
        
        self.aggregator = aggregator.lower()
        self.use_coords = use_coordinates
        
        # g_theta：应用于每个对象对 (o_i, o_j) 的共享 MLP
        # 输入大小：2 * object_dimension + 可选的 question_dimension
        g_input_dim = 2 * object_dim
        if question_dim is not None:
            g_input_dim += question_dim
            
        self.g_theta = nn.Sequential(
            nn.Linear(g_input_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU()
        )
        
        # f_phi：应用于聚合后 g_theta 输出的全局 MLP
        self.f_phi = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(relation_dim // 2, output_dim)
        )

    def generate_pairs(
        self, 
        objects: torch.Tensor, 
        question: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        通过广播从对象集合生成 N^2 个对象对。

        参数：
            objects：输入张量，形状为 (batch, num_objects, object_dim)
            question：可选上下文张量，形状为 (batch, question_dim)

        返回：
            pairs：拼接后的对象对张量，形状为 (batch, num_objects^2, pair_dim)
        """
        batch_size, num_objects, object_dim = objects.size()
        
        # 1. 扩展对象以构造笛卡尔积
        # o_i 形状：(B, N, N, D)
        o_i = objects.unsqueeze(2).repeat(1, 1, num_objects, 1)
        # o_j 形状：(B, N, N, D)
        o_j = objects.unsqueeze(1).repeat(1, num_objects, 1, 1)
        
        # 2. 拼接对象对 (o_i, o_j)
        pairs = torch.cat([o_i, o_j], dim=-1)
        
        # 3. 若有问题上下文，将其拼接到每个对象对
        if question is not None:
            # q_expanded 形状：(B, N, N, Q_dim)
            q_expanded = question.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_objects, 1)
            pairs = torch.cat([pairs, q_expanded], dim=-1)
            
        # 重塑为 (B, N^2, pair_dim)，方便 g_theta 高效批处理
        return pairs.view(batch_size, num_objects * num_objects, -1)

    def aggregate(self, g_out: torch.Tensor) -> torch.Tensor:
        """
        按指定策略聚合成对关系。
        
        参数：
            g_out: (batch, num_pairs, relation_dim)
            
        返回：
            aggregated: (batch, relation_dim)
        """
        if self.aggregator == 'sum':
            return g_out.sum(dim=1)
        elif self.aggregator == 'mean':
            return g_out.mean(dim=1)
        elif self.aggregator == 'max':
            return g_out.max(dim=1)[0]
        else:
            raise ValueError(f"不支持的聚合器：{self.aggregator}")

    def forward(
        self, 
        objects: torch.Tensor, 
        question: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        关系网络的前向传播。

        参数：
            objects: (batch, num_objects, object_dim)
            question：可选，形状为 (batch, question_dim)

        返回：
            logits: (batch, output_dim)
        """
        # 第 1 步：生成 N^2 个对象对
        pairs = self.generate_pairs(objects, question)
        
        # 第 2 步：对每个对象对应用共享权重的 g_theta
        batch_size, num_pairs, pair_dim = pairs.size()
        # 将时间／集合维展平到批次维，以提高效率
        pairs_flat = pairs.view(batch_size * num_pairs, pair_dim)
        g_out_flat = self.g_theta(pairs_flat)
        
        # 恢复形状：(batch, num_pairs, relation_dim)
        g_out = g_out_flat.view(batch_size, num_pairs, -1)
        
        # 第 3 步：全局聚合
        # 默认求和，以保留基数信息
        aggregated = self.aggregate(g_out)
        
        # 第 4 步：通过 f_phi 进行预测
        output = self.f_phi(aggregated)
        
        return output


def add_coordinates(objects: torch.Tensor) -> torch.Tensor:
    """
    向每个对象向量拼接归一化的 (x, y) 坐标。
    
    对空间推理很重要，例如 CLEVR 风格任务（第 3.1 节）。
    若对象特征相同且没有位置，模型就无法区分它们的位置。
    
    参数：
        objects: (batch, N, D)
    
    返回：
        objects_with_coords: (batch, N, D+2)
    """
    batch_size, n_objects, _ = objects.size()
    
    # 在 [-1, 1] 范围创建坐标网格
    coords = torch.linspace(-1, 1, n_objects).to(objects.device)
    # 对一维序列或预先提取的集合：
    x_coords = coords.view(1, n_objects, 1).repeat(batch_size, 1, 1)
    y_coords = torch.zeros_like(x_coords) 
    
    return torch.cat([objects, x_coords, y_coords], dim=-1)


def feature_map_to_objects(f_map: torch.Tensor) -> torch.Tensor:
    """
    将 CNN 特征图转为“对象”集合（第 3.1 节）。
    
    在 Santoro 论文中，最终 k x k 特征图中的每个像素／网格单元
    都被视为独立对象。
    
    参数：
        f_map：CNN 特征，形状为 (batch, channels, k, k)
        
    返回：
        objects：对象集合，形状为 (batch, k*k, channels)
    """
    batch, channels, k, _ = f_map.size()
    # 1. 置换为 (batch, k, k, channels)
    # 2. 将空间维展平为大小为 k*k 的集合
    objects = f_map.permute(0, 2, 3, 1).contiguous()
    return objects.view(batch, k * k, channels)


if __name__ == "__main__":
    # 进一步的冒烟检查：比较 sum 与 mean 聚合器
    obj_num, obj_d = 5, 10
    
    rn_sum = RelationNetwork(object_dim=obj_d, aggregator='sum')
    rn_mean = RelationNetwork(object_dim=obj_d, aggregator='mean')
    
    dummy_objs = torch.ones(1, obj_num, obj_d) # 所有对象相同
    
    out_sum = rn_sum(dummy_objs)
    out_mean = rn_mean(dummy_objs)
    
    print(f"RN（求和）输出幅值：{torch.norm(out_sum).item():.2f}")
    print(f"RN（均值）输出幅值：{torch.norm(out_mean).item():.2f}")
    print('\n说明：求和幅值应更大，因为它随 N^2 个对象对扩展。')
