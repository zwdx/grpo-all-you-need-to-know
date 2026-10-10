[上一章](chapter-06.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-08.md)

---

<a name="chapter-7"></a>

# 第 7 章 序列级方法与变体谱系

**本章目录**

- [7.1 变体爆炸的背景与分类框架](#s7-1)
- [7.2 GSPO：序列级重要性比率的数学动机](#s7-2)
- [7.3 CISPO：保留全部 Token 梯度](#s7-3)
- [7.4 其他值得了解的变体](#s7-4)
- [7.5 谱系图与演化逻辑](#s7-5)
- [7.6 选型建议](#s7-6)
- [7.7 本章小结与自测](#s7-7)

<a name="s7-1"></a>

## 7.1 变体爆炸的背景与分类框架

<a name="s7-1-1"></a>

### 7.1.1 为什么 GRPO 变体如此之多

自 2024 年 GRPO 提出以来，围绕它的变体在 2025–2026 年间呈爆发式增长。这一现象有三重原因：

第一，**RLVR（Reinforcement Learning with Verifiable Rewards）成为推理模型训练的主流范式**。当可验证奖励取代奖励模型成为主要信号源后，GRPO 作为"无需 critic 的组采样策略优化"天然适配 RLVR 的设置（见第 4 章），各团队在落地时不可避免地遇到各种工程与理论病症，进而提出针对性修正。

第二，**多数改动的实现成本极低**。GRPO 的核心目标函数紧凑，许多改进只需修改损失计算的一两行代码（例如将 clip 上界调高、去掉 std 归一化、将 token 级比率换成序列级比率）。低实现门槛使得"打补丁→发论文"的周期极短。

第三，**问题本身是多维度的**。GRPO 同时涉及优势估计、裁剪机制、优化粒度、采样策略四个独立组件，每个组件都有改进空间，且不同组件的改动往往可以组合。这导致变体数量随组合数快速增长。

面对这个局面，读者最需要的不是逐个记忆变体细节，而是建立一个**分类坐标系**，在看到任何新变体时能迅速定位它"改了 GRPO 的哪个组件、治了什么病"。

<a name="s7-1-2"></a>

### 7.1.2 四组件分类框架

我们将 GRPO 的目标函数拆解为四个可独立替换的组件，并据此对所有已知变体进行分类：

| 组件编号 | 组件名称 | 核心问题 | 代表变体 |
|:---:|:---:|:---|:---|
| ① | **优势估计**（Advantage） | 是否除 std、留一法 vs 全组均值、多奖励解耦归一化 | Dr.GRPO（去长度/std 偏差）、RLOO（留一法基线）、GDPO 类 |
| ② | **裁剪机制**（Clip） | 对称 vs 解耦、硬 clip vs 软门控 vs detached 权重 | DAPO（clip-higher）、CISPO（detached IS 权重）、SAPO（软门控）、BAPO（自适应裁剪） |
| ③ | **优化粒度**（Granularity） | token 级 vs 序列级 vs 前缀级 | GSPO（序列级比率）、LUSPO（序列级+长度无偏） |
| ④ | **采样与数据**（Sampling） | 动态过滤、难度课程、off-policy 复用 | DAPO（动态采样）、GVPO（无 IS off-policy） |

**关键观察**：多数变体是在 ①~④ 上做局部替换，彼此**可组合**。例如 DAPO 同时改了 ②（clip-higher）、③（token-level loss）和 ④（动态采样）；GSPO 改了 ③（序列级粒度），但也可与 ① 中的 Dr.GRPO 修正组合使用。这个视角能让读者在阅读新论文时迅速建立结构化理解，而非陷入碎片化的细节记忆。

以下各节按此框架展开，重点讲解 ③ 和 ② 中影响力最大的两支——GSPO 与 CISPO。

<a name="s7-2"></a>

## 7.2 GSPO：序列级重要性比率的数学动机

GSPO（Group Sequence Policy Optimization）由阿里巴巴 Qwen 团队提出（Zheng et al., 2025, arXiv:2507.18071），是 GRPO 家族中将优化粒度从 token 级提升到序列级的代表性工作。本节从数学机理出发，解释为什么这种粒度提升是必要的，以及它带来了什么本质变化。

<a name="s7-2-1"></a>

### 7.2.1 Token 级 IS 比率的三重问题

回顾 GRPO 的做法（见第 3 章）：对组内第 $`i`$ 条回答的第 $`t`$ 个 token，独立计算重要性采样比率

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}$$

并在每个 token 上独立执行 PPO-style clip。这个设计有三个深层问题。

**问题一：方差随序列长度累积。** 如 PPO 文档第 5 章所述，重要性采样估计量的方差在比值分布重尾时可以无界。GRPO 虽然用单步比率避免了轨迹级比率的指数级方差增长，但一条长回答包含 $`|o_i|`$ 个 token，每个 token 的 $`r_{i,t}`$ 都是独立的高方差随机变量。整条序列的等效更新是这些 token 比率的联合作用，其联合方差随 $`|o_i|`$ 增大而累积。这在长 CoT（Chain-of-Thought）场景下尤为严重。

**问题二：优化单元与奖励单元的粒度错配。** 这是 GSPO 的核心批判。在 GRPO 中，奖励 $`R(q, o_i)`$ 是**序列级**的——整条回答获得一个分数，优势 $`\hat{A}_i`$ 对同一序列的所有 token 相同（见第 2 章）。然而裁剪与优化却是 **token 级**的：同一条回答中，某些 token 的 $`r_{i,t}`$ 超出 clip 范围被截断，另一些则没有。这意味着实际生效的梯度与"这条回答整体应该被强化/削弱多少"的语义意图不一致。一条本应被整体强化的好回答，可能因为其中几个 token 的比率碰巧偏高而被部分 clip 掉，导致学习信号被扭曲。

**问题三：MoE 架构下的路由抖动灾难。** GSPO 论文明确指出，GRPO 在 Mixture-of-Experts（MoE）模型上的训练极不稳定。机理如下：MoE 的路由器（router）在每个 token 上动态选择激活哪些专家。策略更新会改变路由决策，而路由改变又会让 token 级的 log 概率发生与"内容质量"无关的剧烈跳变——同一个 token 在不同专家组合下的条件概率可能差异巨大。于是 $`r_{i,t}`$ 出现与优化目标无关的巨大波动，导致梯度爆炸或训练崩溃。这种抖动是 MoE 架构固有的，与策略本身的优劣无关，因此属于需要被消除的噪声。

<a name="s7-2-2"></a>

### 7.2.2 GSPO 的序列级比率

GSPO 的解法是将优化粒度从 token 级提升到序列级。定义序列级重要性比率为

$$s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}\right)$$

即 token 级比率的**几何平均**。长度归一化因子 $`1/|o_i|`$ 的作用至关重要：若不归一，$`\log s_i`$ 将是 $`|o_i|`$ 项 log-ratio 之和，其量级随序列长度线性增长，导致 $`s_i`$ 数值爆炸或下溢，无法用于稳定的 clip 优化。归一化后，$`s_i`$ 的数值范围与序列长度解耦，使得同一套超参可以适用于不同长度的回答。

GSPO 的目标函数在序列级做 clip：

$$J^{GSPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\,\hat{A}_i,\ \text{clip}(s_i(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_i\right)\right]$$

其中 $`\hat{A}_i`$ 仍为序列级优势（如 GRPO 的组均值基线或 Dr.GRPO 的修正形式）。注意此处 clip 作用于**整条回答**：一条回答要么整体被强化、要么整体被削弱、要么整体被 clip，优化单元与奖励单元完全对齐。（注：上述目标函数是序列级形式的直接类比，具体细节以原论文为准。）

<a name="s7-2-3"></a>

### 7.2.3 梯度分析：GSPO vs GRPO

要理解 GSPO 与 GRPO 的本质差异，必须对比两者的梯度结构。

对 GSPO，由于 $`s_i(\theta) = \exp\left(\frac{1}{|o_i|}\sum_t \log r_{i,t}(\theta)\right)`$，由链式法则可得

$$\nabla_\theta s_i(\theta) = s_i(\theta) \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$$

因此在未触发 clip 的情形下，GSPO 对参数 $`\theta`$ 的梯度为

$$\nabla_\theta J^{GSPO} \propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G s_i(\theta)\,\hat{A}_i \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

**关键特征**：$`s_i(\theta)`$ 是一个**序列级标量**，它对同一序列内所有 token 的梯度施加相同的加权系数。

对比 GRPO 的梯度（见第 3 章）：

$$\nabla_\theta J^{GRPO} \propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} r_{i,t}(\theta)\,\hat{A}_i \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

**关键区别**：GRPO 让每个 token 用自己的 $`r_{i,t}`$ 独立加权。当某个 token 的 $`r_{i,t}`$ 因路由抖动或其他噪声异常偏高时，该 token 的梯度会被过度放大；而在 GSPO 中，这种局部异常被几何平均平滑掉了。**GSPO 用一个序列级标量统一加权所有 token 的梯度，GRPO 则让每个 token 用自己的比率加权**——前者方差更低、对局部噪声更鲁棒，后者可能对个别 token 过度反应。这是理解两者差异的关键。

<a name="s7-2-4"></a>

### 7.2.4 ε 的量级差异

由于 $`s_i(\theta)`$ 是 token 级比率的几何平均，根据大数定律的性质，其偏离 1 的幅度天然比单个 $`r_{i,t}`$ 小得多。直觉上，多个独立随机变量的几何平均比方差最大的那个个体更接近期望值。因此 GSPO 的 clip 阈值 $`\varepsilon`$ 需要设得比 GRPO 显著更小——若 GRPO 使用 $`\varepsilon = 0.2`$，GSPO 可能需要小若干个数量级的 $`\varepsilon`$ 才能达到等效的约束强度。**具体数值请以原论文为准，不要直接套用 GRPO 的超参。** 若错误地使用 GRPO 量级的 $`\varepsilon`$，序列级 clip 几乎永远不会触发，等同于退化为无 clip 的 REINFORCE。

<a name="s7-2-5"></a>

### 7.2.5 GSPO 的实践影响

GSPO 论文报告其在 Qwen3 系列模型的 RL 训练中得到了应用，尤其在 MoE 架构上显著提升了训练稳定性。这验证了上述理论分析：当 token 级比率因架构特性（如 MoE 路由）而产生与优化目标无关的剧烈波动时，序列级聚合是一种有效的降噪手段。对于非 MoE 的 dense 模型，GSPO 的收益可能不如 MoE 场景显著，但其"粒度对齐"的设计理念仍具参考价值。

<a name="s7-3"></a>

## 7.3 CISPO：保留全部 Token 梯度

CISPO（Clipped IS-weight Policy Optimization）出自 MiniMax-M1 技术报告（arXiv:2506.13585），代表了 GRPO 变体谱系中与 GSPO 截然不同的另一条路线：它不改变优化粒度，而是改变裁剪机制本身，目标是**保留所有 token 的梯度**。

<a name="s7-3-1"></a>

### 7.3.1 Hard Clip 的信息丢弃问题

承接 PPO 文档第 6 章的六情形分析：PPO-style hard clip 一旦生效（即 $`r_{i,t}`$ 超出 $`[1-\varepsilon, 1+\varepsilon]`$ 范围），该 token 对目标函数的梯度**完全归零**。这在短回答场景中或许可以接受，但在长 CoT 场景下代价极大。

一条上万 token 的推理回答中，真正决定推理质量的往往是少数"关键转折 token"——例如反思词 "wait"、"let me recheck"、"alternatively" 等。这些 token 的出现标志着模型在进行自我纠错或路径切换，是高质量推理的核心信号。然而，正因为它们是低概率事件（模型在旧策略下不太可能生成这些词），其重要性比率 $`r_{i,t}`$ 容易远超 clip 上界，从而被 hard clip 完全丢弃梯度。结果是：模型学到了大量"常规 token"的信号，却丢失了最关键的"反思 token"的信号，推理能力难以持续提升。

<a name="s7-3-2"></a>

### 7.3.2 CISPO 的解法：Detached IS 权重

CISPO 的核心思想是：**不对目标函数做 PPO 式 hard ratio clipping，而是把重要性采样权重裁剪后作为 detached（stop-gradient）系数**。其损失形式类似 REINFORCE / policy gradient：

$$L^{CISPO} = -\mathbb{E}\left[\frac{1}{\sum_i |o_i|}\sum_i \sum_t \text{sg}\!\left(\hat{r}^{IS}_{i,t}\right)\,\hat{A}_i\,\log\pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

其中 $`\hat{r}^{IS}_{i,t} = \text{clip}(r^{IS}_{i,t},\, 0,\, \varepsilon^{IS}_{high})`$ 是被裁剪的 IS 权重，$`\text{sg}(\cdot)`$ 表示 stop-gradient（即该部分不参与求导）。（注：这是依据"CISPO 裁剪 IS 权重作为 detached 系数而非 hard ratio clipping"这一核心描述的公式重构，**具体形式请以 MiniMax-M1 技术报告为准**。）

<a name="s7-3-3"></a>

### 7.3.3 梯度对比：CISPO vs PPO/GRPO

CISPO 与 PPO/GRPO 在梯度层面的本质区别可以通过直接求导看出。

对 CISPO，由于 $`\text{sg}(\hat{r}^{IS}_{i,t})`$ 被视为常数，梯度为

$$\nabla_\theta L^{CISPO} = -\mathbb{E}\left[\frac{1}{\sum_i |o_i|}\sum_i \sum_t \text{sg}\!\left(\hat{r}^{IS}_{i,t}\right)\,\hat{A}_i\,\nabla_\theta \log\pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

**关键特征**：$`\nabla_\theta \log\pi_\theta`$ 项对**所有 token 都非零**。clip 只影响"这个 token 的权重多大"（通过 $`\text{sg}(\hat{r}^{IS}_{i,t})`$ 控制），而不影响"这个 token 有没有梯度"。即使某个 token 的原始比率 $`r^{IS}_{i,t}`$ 远超 clip 上界，它仍然以被裁剪后的权重 $`\varepsilon^{IS}_{high}`$ 参与梯度更新，而非被完全丢弃。

对比 PPO/GRPO 的梯度：当 $`r_{i,t} > 1+\varepsilon`$ 且 $`\hat{A}_i > 0`$ 时，$`\min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t})\hat{A}_i) = (1+\varepsilon)\hat{A}_i`$，该项对 $`\theta`$ 的梯度为零（因为 clip 后的值是常数）。**Hard clip 直接杀死梯度，CISPO 的 detached 权重只是缩放梯度**——这是两者的本质区别。

<a name="s7-3-4"></a>

### 7.3.4 为什么这对长 CoT 重要

CISPO 的设计实现了"保信号"与"控方差"的折中：

- **保信号**：稀有但关键的 token（如反思词）不会因为比率超限而失去学习信号。它们的梯度被保留，只是权重被限制在一个合理范围内。
- **控方差**：clip 上界 $`\varepsilon^{IS}_{high}`$（通常设得比 PPO 的 $`1+\varepsilon`$ 大得多，如 10 量级——**具体以原论文为准**）仍能防止单个 token 的权重爆炸导致方差失控。相比完全不 clip 的 REINFORCE，CISPO 通过宽松的上界提供了必要的安全阀。

这种设计特别适合长 CoT 推理模型的 RL 训练，其中关键信号的稀疏性与 IS 比率的极端值并存。MiniMax-M1 技术报告将 CISPO 用于其长 CoT 推理模型的训练，验证了这一思路的有效性。

<a name="s7-4"></a>

## 7.4 其他值得了解的变体

以下变体各有定位，简要概述以帮助读者建立完整谱系认知。

<a name="s7-4-1"></a>

### 7.4.1 RLOO

RLOO（REINFORCE Leave-One-Out，Ahmadian et al., 2024, arXiv:2402.14740）使用留一法基线（见第 2 章），无 clip 无 KL，是 GRPO 的极简近亲。实践中与 GRPO 效果常接近，TRL 框架有独立实现。适合想理解"组内基线的最小可行版本"的读者，也适合作为消融实验的 baseline。

<a name="s7-4-2"></a>

### 7.4.2 Dr.GRPO

Dr.GRPO（arXiv:2503.20783，第 5 章已详述）去掉长度归一化 $`(1/|o_i|)`$ 与 std 归一化，消除两种系统性偏差。在谱系中属于"改优势估计 + 改聚合方式"类。当观察到回答长度失控或难度依赖的训练不稳定时，Dr.GRPO 的修正是首选起点。

<a name="s7-4-3"></a>

### 7.4.3 GVPO

GVPO（Group Variance Policy Optimization，NeurIPS 2025，作业帮与港科大广州）的核心特点是**避免使用重要性采样**，损失等价于 MSE 形式，理论上给出唯一最优解保证，支持无 IS 的 off-policy 训练从而可复用历史数据，对超参 $`\beta`$ 不敏感。（注：此条信息来自二手资料，**以原论文为准**。）GVPO 的意义在于绕开了 IS 方差问题这条根本路径，而非在 IS 框架内做修补。

<a name="s7-4-4"></a>

### 7.4.4 BAPO

BAPO（Balanced Policy Optimization）采用自适应裁剪以平衡正负样本的梯度贡献、防止熵崩塌。其目标与 DAPO 的 clip-higher 相近（都是解决正负样本不对称问题），但机制不同：BAPO 动态调整 clip 范围而非固定解耦上下界。

<a name="s7-4-5"></a>

### 7.4.5 SAPO

SAPO（Soft Adaptive Policy Optimization）用软门控函数替代硬 clip，提供更平滑的梯度衰减曲线。有分析指出它在某些参数设置下退化到类 GSPO 的序列级形式，暗示序列级优化与软裁剪之间存在深层联系。

<a name="s7-4-6"></a>

### 7.4.6 长度偏差的后续工作

LUSPO（Length-Unbiased Sequence Policy Optimization）指出 GSPO 本身仍存在长度偏差并加以修正。这说明"长度偏差"是贯穿 token 级与序列级方法的共同难题（呼应第 5 章），并非切换到序列级就能自动解决。

<a name="s7-4-7"></a>

### 7.4.7 关于变体数量的声明

**这个领域变体极多且仍在快速演进**（2025–2026 年间新变体持续出现），本章只覆盖影响力最大的几支。其他已知名称包括 REINFORCE++-Baseline、GEPO、ESPO、MO-GRPO（多目标 reward hacking 缓解）、DAGRPC 等。读者遇到新变体时，应使用 7.1 节的四类框架去定位它改了什么组件，并以其原论文与主流框架（verl、TRL、OpenRLHF、NeMo-RL、AReaL）的实现为准。

<a name="s7-5"></a>

## 7.5 谱系图与演化逻辑

<a name="s7-5-1"></a>

### 7.5.1 演化树

以下文本树展示从 PPO 出发的主要演化路径，每条边标注"改了什么组件、治什么病"：

```
PPO (2017)
└── 去掉 critic，用组内基线 ──→ GRPO (2024) / RLOO (2024)
     │
     ├── 去偏差：长度归一化 + std 归一化 [改①] ──→ Dr.GRPO (2025)
     │
     ├── 工程精修：clip-higher + 动态采样 + token loss + 超长塑形 + 去KL [改②③④] ──→ DAPO (2025)
     │
     ├── 粒度提升到序列级 [改③] ──→ GSPO (2025)
     │    └── 修长度偏差 [改①] ──→ LUSPO
     │
     ├── 保全部 token 梯度：detached IS 权重 [改②] ──→ CISPO (2025)
     │
     └── 绕开 IS：MSE 损失 + off-policy [改④] ──→ GVPO (2025)
```

<a name="s7-5-2"></a>

### 7.5.2 时间线表格

| 算法 | 年份 | 机构 | 核心改动 | 主要动机 |
|:---|:---:|:---|:---|:---|
| PPO | 2017 | OpenAI | 重要性采样 + clip 的策略优化 | 稳定 on-policy 训练 |
| RLOO | 2024 | Cohere 等 | 留一法基线，无 clip 无 KL | 简化 RLHF 训练 |
| GRPO | 2024 | DeepSeek | 组采样 + 组均值基线，去 critic | 适配 LLM RLVR |
| DAPO | 2025 | ByteDance 等 | clip-higher + 动态采样 + token loss + 超长塑形 | 长 CoT 训练稳定性 |
| Dr.GRPO | 2025 | — | 去长度/std 归一化 | 消除系统性偏差 |
| CISPO | 2025 | MiniMax | Detached IS 权重替代 hard clip | 保留关键 token 梯度 |
| GSPO | 2025 | Alibaba Qwen | 序列级 IS 比率（几何平均） | MoE 训练稳定性 |
| GVPO | 2025 | Zuoyebang / HKUST(GZ) | 无 IS，MSE 损失 | 绕开 IS 方差问题 |

（注：表中"—"表示机构信息未在核实清单中给出，以原论文为准。）

<a name="s7-6"></a>

## 7.6 选型建议

<a name="s7-6-1"></a>

### 7.6.1 决策路径

以下是面向实践者的选型决策路径，按优先级排列：

1. **任务有可靠 verifier、模型是 dense 架构、回答长度中等** → **GRPO 或 DAPO** 起步。DAPO 在长 CoT 上更稳，是当下更常用的默认选择。
2. **模型是 MoE 架构** → 优先考虑 **GSPO**。Token 级比率在 MoE 上有路由抖动问题（见 7.2.1），序列级比率能有效平滑。
3. **长 CoT 且观察到"关键 token 学不到"、熵快速下降** → 考虑 **CISPO**（保梯度）或 DAPO 的 clip-higher。两者机制不同但目标相近，可做消融对比。
4. **回答长度差异极大、观察到长度失控** → 必须处理长度偏差：**Dr.GRPO** 的去归一化修正 + DAPO 的超长软惩罚塑形。
5. **想复用历史 rollout 数据 / 担心 IS 方差** → 考虑 **GVPO** 类无 IS 方法。但注意第 2 章指出组内基线本身不适合跨 batch 复用，需配合相应的基线修正。
6. **资源受限、只想跑通验证想法** → **RLOO**（最简单，无 clip 无 KL）或直接用 GRPO 的小 $`G`$ 配置。

<a name="s7-6-2"></a>

### 7.6.2 横向对比大表

| 维度 | PPO | GRPO | RLOO | Dr.GRPO | DAPO | GSPO | CISPO | GVPO |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 是否需要 Critic | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 优势估计方式 | GAE | 组均值基线 | 留一法基线 | 去归一化组均值 | 组均值+动态过滤 | 组均值基线 | 组均值基线 | MSE 等价 |
| 优化粒度 | Token 级 | Token 级 | Token 级 | Token 级 | Token 级 | **序列级** | Token 级 | Token 级 |
| Clip 机制 | 对称硬 clip | 对称硬 clip | 无 | 对称硬 clip | **解耦 clip-higher** | 序列级硬 clip | **Detached IS 权重** | 无 IS |
| 是否含 KL | ✅ | ✅（可去） | ❌ | ✅（可去） | ❌ | ✅（可去） | ✅（可去） | ❌ |
| 长度偏差处理 | N/A | 有偏差 | 有偏差 | **显式修正** | 超长软惩罚 | 部分缓解 | 有偏差 | 以原论文为准 |
| 适用场景 | 通用 RL | RLVR 默认 | 极简 baseline | 长度/难度偏差 | 长 CoT | **MoE** | **长 CoT 稀疏信号** | Off-policy 复用 |
| 实现复杂度 | 高 | 中 | 低 | 中 | 中高 | 中 | 中 | 以原论文为准 |

（注：表中标注"以原论文为准"的条目表示核实清单中未给出确切信息，请勿直接引用。）

<a name="s7-6-3"></a>

### 7.6.3 务实建议

**不要盲目追新变体。** 多数变体的收益在特定设置下才显著，且很多改动可组合。建议从 DAPO 或 GRPO 的成熟框架实现起步，先跑通并建立自己的基线与诊断能力（见第 8 章），再针对观察到的具体病症引入对应改动，**每次只改一处并做消融**。变体选择应由诊断驱动，而非由论文发表时间驱动。

<a name="s7-7"></a>

## 7.7 本章小结与自测

<a name="h233"></a>

### 要点回顾

1. GRPO 变体可按四组件分类：优势估计、裁剪机制、优化粒度、采样与数据。多数变体是局部替换且可组合。
2. GSPO 用序列级 IS 比率（token 级比率的几何平均）解决粒度错配与 MoE 路由抖动问题，梯度由序列级标量统一加权。
3. CISPO 用 detached IS 权重替代 hard clip，保留所有 token 的梯度，特别适合长 CoT 中稀疏关键信号的学习。
4. 序列级比率经长度归一化后波动范围远小于 token 级比率，ε 需显著调小。
5. 长度偏差是贯穿 token 级与序列级方法的共同难题，Dr.GRPO 与 LUSPO 分别从不同角度修正。
6. 选型应由诊断驱动：dense 模型用 DAPO/GRPO，MoE 用 GSPO，长 CoT 稀疏信号用 CISPO，长度失控用 Dr.GRPO。

<a name="h234"></a>

### 自测题

**题目 1**：GSPO 的序列级比率 $`s_i(\theta)`$ 为何能稳定 MoE 模型的 RL 训练？请从梯度的角度解释。

<details>
<summary>查看参考答案</summary>

参考答案：MoE 路由器在每个 token 上动态选择专家，策略更新导致路由变化，使 token 级 log 概率产生与内容质量无关的剧烈跳变。GRPO 的 token 级比率 $`r_{i,t}`$ 直接反映这种跳变，导致个别 token 的梯度被异常放大或缩小。GSPO 的 $`s_i(\theta)`$ 是 token 级比率的几何平均，$`\nabla_\theta s_i = s_i \cdot (1/|o_i|)\sum_t \nabla_\theta \log\pi_\theta(o_{i,t}|\cdot)`$，用一个序列级标量统一加权所有 token 的梯度，局部路由抖动被平均效应平滑掉，因此梯度方差更低、训练更稳定。

</details>

**题目 2**：CISPO 与 PPO hard clip 在梯度上有什么本质区别？为什么这对长 CoT 重要？

<details>
<summary>查看参考答案</summary>

参考答案：PPO hard clip 在比率超出范围时梯度完全归零，该 token 的学习信号被丢弃。CISPO 将裁剪后的 IS 权重作为 detached 系数（stop-gradient），梯度始终来自 $`\nabla_\theta \log\pi_\theta`$ 项，对所有 token 非零。clip 只控制权重大小，不决定有无梯度。在长 CoT 中，关键转折 token（如反思词）往往是低概率事件，其比率易超 clip 上界；CISPO 保留了这些稀有但关键 token 的学习信号，同时通过宽松的上界防止方差失控。

</details>

**题目 3**：为什么 GSPO 的 clip 阈值 $`\varepsilon`$ 不能直接沿用 GRPO 的值（如 0.2）？

<details>
<summary>查看参考答案</summary>

参考答案：$`s_i(\theta)`$ 是 $`|o_i|`$ 个 token 级比率的几何平均，其偏离 1 的幅度天然比单个 $`r_{i,t}`$ 小得多。若使用 GRPO 量级的 $`\varepsilon`$，序列级 clip 几乎永远不会触发，GSPO 退化为无 clip 的 REINFORCE，失去约束策略更新幅度的安全机制。因此 $`\varepsilon`$ 必须显著调小以匹配序列级比率的波动范围。具体数值以原论文为准。

</details>

**题目 4**：使用 7.1 节的四组件分类框架，分析 DAPO 同时改了哪些组件？

<details>
<summary>查看参考答案</summary>

参考答案：DAPO 改了三个组件：② 裁剪机制（clip-higher，解耦上下界 $`\varepsilon_{low}=0.2, \varepsilon_{high}=0.28`$）、③ 优化粒度（token-level loss 聚合）、④ 采样与数据（动态采样过滤全对全错组）。此外还移除了 KL 项。这体现了变体之间的可组合性。

</details>

**题目 5**：GVPO 与 GRPO/GSPO/CISPO 的根本区别是什么？它解决了什么问题？

<details>
<summary>查看参考答案</summary>

参考答案：GVPO 的根本区别在于**完全避免使用重要性采样**，损失等价于 MSE 形式。GRPO/GSPO/CISPO 都在 IS 框架内做改进（改粒度、改裁剪方式），但都无法根除 IS 估计量本身的方差问题。GVPO 绕开了 IS 这条路径，理论上给出唯一最优解保证，并支持 off-policy 训练以复用历史数据。（注：以上信息来自二手资料，以原论文为准。）

</details>

---

[上一章](chapter-06.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-08.md)
