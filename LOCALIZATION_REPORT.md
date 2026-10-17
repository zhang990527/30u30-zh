# 本地化验收报告（进度快照）

**尚未完成全仓库汉化，请勿将当前分支按最终版本合并。**

基线提交：`0edf4bf45fbfb052d5f5760929f28998e1631de3`。开始时工作区无已有改动。当前工作为可审阅的中间版本，后续章节与已改动文件均可能还需翻译、术语复核及运行验证。

## 范围与进度

初始主体盘点：182 个 Markdown、418 个 Python、30 个 Notebook、2 个 HTML、1 个 JavaScript，约 450 万字符。结构检查另外纳入样式与示例文本，共 674 个原有文本文件。

本快照有 **219 个原有文件发生改动**。这仅是改动数，不是翻译完成数或准确率。逐目录计数见 [进度清单](localization/progress.tsv)。Day 6、Day 7 已完成各自一轮文本处理与自检，不能将其扩大为全仓库完成声明。其他范围包含未完成文件。

范围包括文档、Notebook Markdown、注释、文档字符串、显示文案和图表标签。训练语料、原始模型生成样本、代码标识符、接口、路径及历史 Notebook 输出保留原义；法律许可原文目前保留，尚待配中文说明。

并行任务在此快照前触发平台用量上限而中断；不能据此声称剩余章节已处理。原始来源访问也有局部限制，具体“已取得”与“未取得”需逐项记录。

## 已执行的检查

- 修改前：Day 1 CLI 短训练 20 次更新，Notebook 全部 9 个代码单元执行成功。
- 修改后：Day 1 固定随机种子下，前向结果、损失、反向梯度、参数更新及 40 字符采样，与原提交逐项完全相同。
- Day 6：16×16 网格、100 步固定种子模拟，译前译后的返回键和值逐项一致；setup、CLI 及 Notebook 的 6 个代码单元通过。详见 [章节记录](localization/chapter_06.md)。
- Day 7：10×10 网格、20 步 CLI 通过；20×20 固定种子快照及测量的译前译后结果相同。
- 全局静态检查：非文本 Python AST 结构差异 0；新增 Python 语法错误 0；原有语法错误 6；新增 Markdown 本地链接错误 0。检查忽略字符串内容，因此不能排除字符串语义、协议或数据误改，字符串变更仍需人工审查。
- Notebook 单元数量、类型和历史输出进行结构比较；可解析代码单元进行非文本 AST 比较。未解析的 magic 单元单列，不声称已全部执行。
- `pip check` 通过。已准备 CPU PyTorch 与 Noto CJK 字体，但不代表所有章节训练都验证通过。

复核命令：

```bash
python localization/audit.py --output /tmp/30u30-localization-audit.json
```

脚本会检查对照提交中的所有相关文件，列出字符串变更、数学片段待审项、原有链接问题。它不能证明译文准确，不使用汉字存在率或文件改动率作为完成率。

## 已取得的一手材料

- Karpathy，2015-05-21，*The Unreasonable Effectiveness of Recurrent Neural Networks*：[作者网站源文件](https://raw.githubusercontent.com/karpathy/karpathy.github.io/master/_posts/2015-05-21-rnn-effectiveness.markdown)。核实原文实验为 LSTM，而本章代码为基本 RNN。
- Krizhevsky、Sutskever、Hinton，2012，AlexNet：[NeurIPS 正式论文](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)。核对集成规模、Top-5 错误率和 ReLU 实验条件。
- Pointer Networks：[NeurIPS 2015 正式论文](https://papers.nips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)，核对公式、隐藏层设置及优化条件。
- [ResNet 官方仓库](https://github.com/KaimingHe/deep-residual-networks)、[ResNet v2 官方仓库](https://github.com/KaimingHe/resnet-1k-layers)、[空洞卷积官方仓库](https://github.com/fyu/dilation)的 README。
- [关系记忆官方源码](https://raw.githubusercontent.com/google-deepmind/sonnet/master/sonnet/python/modules/relational_memory.py)。
- [Dropout 的 JMLR 页面](https://jmlr.org/papers/v15/srivastava14a.html)、[GroundHog 的 NMT 实验说明](https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt)、[Annotated Transformer 作者仓库](https://github.com/harvardnlp/annotated-transformer)。
- Day 22 的处理代理已取得 Deep Speech 2 的 `1512.02595v1`，并核对作者名单、实验表和章节编号；该章尚未完成全量翻译与验收。

尚未逐篇读取的材料不得标成“已核验”。译名表见 [LOCALIZATION_TERMS.md](LOCALIZATION_TERMS.md)。没有外部领域专家独立认证，不将模型自审描述为专家审校。

## 已发现的原仓库问题

- Day 1：`--epochs` 实际是更新次数；速查表 `--lr`／`--length` 与接口不符；序列总损失初始化与逐字符平均损失尺度不一致；直接指数计算仍可能溢出。未修改算法。
- Day 6：复杂度代理量不保证出现先升后降；不能由代理量图形推出复杂性的普遍定律。
- Day 7：批量交换可能共享索引；20×20、seed 42 下初始粒子数 200，100 步中出现 192–214，说明原实现没有始终守恒。译前译后行为一致，未静默修复。
- AlexNet：15.3% 对应 7 个 CNN 的集成；5 个 CNN 是 16.4%。84.7% 是 Top-5 正确率，不能泛称一般准确率。ReLU 的 6 倍比较有特定 CIFAR-10 实验条件。
- Day 16：集合编码的逐元素输出是置换等变；读取—处理—写入与训练入口存在文码差异；Notebook 的交叉熵输入和二次 softmax 问题尚属原缺陷。
- Day 17–20：部分文档入口不存在、邻章链接或练习表不匹配；原练习占位表达式可能不能解析。
- 静态基线的 6 个语法错误：Day 20 的 5 个练习与 Day 27 的 `implementation.py`。未以禁用检查或修改算法方式掩盖。

## 阻断项与剩余工作

仍需处理未翻译章节、Python/Notebook 残留文本、网站注释和显示文案，并逐项复核已改动文件。还需全仓公式／数值／引用审计、字符串安全审查、完整术语一致性审校和更多代表性运行检查。历史输出及图片中的英文不能被算作已经汉化。

创建 PR 时，Git 推送预检成功，但 `gh api` 与直接 HTTPS 请求 `api.github.com` 均被代理以 403 拒绝。已在环境配置草稿中追加该域名；保存草稿不代表运行时访问已恢复。需要在环境设置保存网络变更后重试 PR 创建。
