# 第 6 章本地化报告

范围：`papers/06_Complexodynamics` 内 6 个 Markdown、14 个 Python、1 个 Notebook，共 21 个文件。已翻译文档正文、全部 232 处原有 Python 文档字符串/自然语言注释、练习提示、安全输出和图标签、Notebook 全部 7 个 Markdown 单元及 6 个代码单元中的展示文本。代码标识符、文件名、字典键、数据样本、公式和数值保留。原英文 Markdown 标题的显式 slug 锚点已保留。

译名：complextropy 为“复杂熵”；sophistication 为“精致复杂度”；logical depth 为“逻辑深度”；coarse-graining 为“粗粒化”；two-part code 为“两段式编码”。首次出现附英文，说明前两项为本项目译名，不声称为官方名称。

来源：原仓库来源为 Scott Aaronson 2011 年博文 *The First Law of Complexodynamics*，不是论文。`https://scottaaronson.blog/?p=762` 及作者旧域名 `https://www.scottaaronson.com/blog/?p=762`、`https://scottaaronson.com/blog/?p=762` 均因代理 403 无法读取。未宣称原文/评论编号/后续历史/截至 2024 年开放状态已独立核验。README 与回顾说明明确标记限制，保留原始出处链接供后续核查。

## 验证

- 保存编辑前整章副本：`/workspace/.localization-30u30/ch06-baseline/`。
- 14 个 Python 均通过语法解析；将字符串常量统一归一化后，全部 AST 结构与基线完全相同。非字符串语法、数值、字节样本、公式运算无变化。
- Notebook 6 个代码单元去除 IPython magic 后 AST 归一化与基线相同；`nbformat.validate` 通过。
- 使用统一 `/workspace/.30u30-runtime/activate.sh` 环境，通过 `nbclient` 实际执行 Notebook 全部 6 个代码单元；执行结果存于 `/workspace/.localization-30u30/chapter06-executed.ipynb`，未写回仓库。
- 原版与译版 `run_simulation(grid_size=16,n_steps=100,swaps_per_step=10,measure_interval=10,seed=42)` 的所有返回键、标量、列表与快照逐项完全一致。
- `python setup.py` 通过依赖导入及冒烟检查。
- 无绘图 CLI：`python train_minimal.py --grid-size 16 --steps 100 --swaps-per-step 10 --measure-interval 10 --no-plot` 成功。
- Markdown 英文自然段残留审计未发现遗漏；保留作者姓名、英文对照、数学表达式、命令、标识符、原始信号数据和 matplotlib 枚举参数。
- 原有练习 `NotImplementedError` 及算法缺陷全部保留，没有补写解答或修改教学逻辑。

## 原仓库问题及译注

1. `grid_entropy` 只看全局 0/1 比例；交换守恒比例。实测全部 11 个时刻熵均为 1.0，不能展示“单调增加”。README、相关 docstring、Notebook 和图标题已明确解释；逻辑不改。
2. 平均局部熵可波动，不能保证单条随机轨迹逐步单调。
3. 原始 gzip 大小、边界比例不保证先升后降；实际短模拟 gzip 从 27 增至 55，最大值在终点，没有内部隆起。预设结论输出已标记为原脚本预期，需要结合实测。
4. `batch_swap` 随机选择像素/方向并跳过无效尝试，与每次均匀选择异色边的 `swap_step` 在时间参数化和边界行为上不同。`times` 计交换尝试数，不是成功交换数。
5. 两段式编码残差经过 uint8 量化，仓库未提供经验证的无损解码器，因此“精确重建”只是原教学意图。
6. CHEATSHEET 的所谓“向量化、互不相邻批量交换”示例实际仍为顺序循环，且不保证像素对互不相邻；原代码保留并加说明。
7. README 中 sophistication 的集合条件与回顾中的历史性概括需要原始来源进一步核查；未把这些表述提升为已证明结论。

未运行：50,000 步完整模拟及全部 7 张独立可视化图（Notebook 已实际绘图）；不声称完成独立领域专家语义认证。无新增语法/Notebook/代表性模拟失败。外部来源阻塞仍在，整章技术含义以明确证据边界交付。
