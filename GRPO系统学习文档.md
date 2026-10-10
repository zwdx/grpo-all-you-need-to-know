<a id="h1"></a>

# GRPO 系统学习文档

<!-- toc:start -->
[返回项目首页](README.md) · [阅读路线](#user-content-h5) · [全文通用符号约定](#user-content-h6)

## 可点击目录

- [第 1 章 从 PPO 到 GRPO：动机与问题设定](#user-content-chapter-1)
- [第 2 章 组内基线的理论基础](#user-content-chapter-2)
- [第 3 章 GRPO 算法本体](#user-content-chapter-3)
- [第 4 章 奖励信号与可验证奖励（RLVR）](#user-content-chapter-4)
- [第 5 章 GRPO 的已知偏差与 Dr.GRPO 修正](#user-content-chapter-5)
- [第 6 章 DAPO：面向长链推理的四项关键改进](#user-content-chapter-6)
- [第 7 章 序列级方法与变体谱系](#user-content-chapter-7)
- [第 8 章 工程实现与实践](#user-content-chapter-8)

<details>
<summary>展开小节目录</summary>

- [1.1 开场定位：GRPO 是什么](#user-content-s1-1)
- [1.2 LLM 后训练的 MDP 视角](#user-content-s1-2)
  - [1.2.1 状态、动作与策略](#user-content-s1-2-1)
  - [1.2.2 奖励的特殊性：outcome-level 稀疏奖励](#user-content-s1-2-2)
  - [1.2.3 关键推论：所有 token 共享同一个回报值](#user-content-s1-2-3)
  - [1.2.4 折扣因子 $`\gamma`$ 的处境](#user-content-s1-2-4)
- [1.3 PPO 在 LLM 场景下的四个负担](#user-content-s1-3)
  - [1.3.1 显存负担：四模型并存](#user-content-s1-3-1)
  - [1.3.2 训练负担：critic 与 actor 的资源竞争](#user-content-s1-3-2)
  - [1.3.3 价值函数在长序列上的固有困难](#user-content-s1-3-3)
  - [1.3.4 超参与工程负担](#user-content-s1-3-4)
- [1.4 GRPO 的核心洞察：以采样换模型](#user-content-s1-4)
- [1.5 PPO 与 GRPO 的系统对比](#user-content-s1-5)
- [1.6 GRPO 的适用边界](#user-content-s1-6)
  - [1.6.1 适合的场景](#user-content-s1-6-1)
  - [1.6.2 不适合或需谨慎的场景](#user-content-s1-6-2)
  - [1.6.3 与 RLVR 的关系](#user-content-s1-6-3)
- [1.7 全文档阅读路线](#user-content-s1-7)
- [1.8 本章小结与自测](#user-content-s1-8)
  - [要点回顾](#user-content-h33)
  - [自测题](#user-content-h34)
- [2.1 从 PPO 文档的基线理论出发](#user-content-s2-1)
  - [2.1.1 基线的两个基本事实](#user-content-s2-1-1)
  - [2.1.2 本章要回答的问题](#user-content-s2-1-2)
- [2.2 蒙特卡洛基线：$`V^\pi(q)`$ 的采样估计](#user-content-s2-2)
  - [2.2.1 LLM RL 中的价值函数与采样估计](#user-content-s2-2-1)
  - [2.2.2 无偏性证明](#user-content-s2-2-2)
  - [2.2.3 方差分析](#user-content-s2-2-3)
  - [2.2.4 基线误差向梯度的传导](#user-content-s2-2-4)
- [2.3 留一法基线（Leave-One-Out）与 RLOO](#user-content-s2-3)
  - [2.3.1 动机：消除自相关偏差](#user-content-s2-3-1)
  - [2.3.2 留一法基线的定义](#user-content-s2-3-2)
  - [2.3.3 无偏性的严格证明](#user-content-s2-3-3)
  - [2.3.4 留一法基线的方差](#user-content-s2-3-4)
  - [2.3.5 RLOO：REINFORCE Leave-One-Out](#user-content-s2-3-5)
- [2.4 GRPO 用的是哪一种基线](#user-content-s2-4)
  - [2.4.1 原论文的明确选择](#user-content-s2-4-1)
  - [2.4.2 偏差的实践影响](#user-content-s2-4-2)
  - [2.4.3 论文与实现的差异](#user-content-s2-4-3)
  - [2.4.4 偏差的定量表达：$`(G-1)/G`$ 缩放定理](#user-content-s2-4-4)
- [2.5 除以标准差的含义与代价](#user-content-s2-5)
  - [2.5.1 数学含义：逐 prompt 的自适应尺度归一化](#user-content-s2-5-1)
  - [2.5.2 好处](#user-content-s2-5-2)
  - [2.5.3 代价：难度偏差与非线性破坏](#user-content-s2-5-3)
  - [2.5.4 全同奖励下的零梯度现象](#user-content-s2-5-4)
- [2.6 与 Critic 基线的系统性对比](#user-content-s2-6)
  - [2.6.1 自适应性的深入讨论](#user-content-s2-6-1)
  - [2.6.2 结论性判断](#user-content-s2-6-2)
- [2.7 GAE 在 GRPO 中的命运](#user-content-s2-7)
  - [2.7.1 GAE 的回顾](#user-content-s2-7-1)
  - [2.7.2 GRPO 设定下的退化](#user-content-s2-7-2)
  - [2.7.3 代价：信用分配的丧失](#user-content-s2-7-3)
- [2.8 采样数 $`G`$ 的选取：数学分析](#user-content-s2-8)
  - [2.8.1 边际收益递减](#user-content-s2-8-1)
  - [2.8.2 小 $`G`$ 的统计风险](#user-content-s2-8-2)
  - [2.8.3 实用区间](#user-content-s2-8-3)
  - [2.8.4 $`G`$ 与 Prompt 多样性的权衡](#user-content-s2-8-4)
- [2.9 本章小结与自测](#user-content-s2-9)
  - [本章要点](#user-content-h73)
  - [自测题](#user-content-h74)
- [3.1 符号与设定](#user-content-s3-1)
- [3.2 GRPO 完整目标函数](#user-content-s3-2)
  - [3.2.1 DeepSeekMath 原论文形式](#user-content-s3-2-1)
  - [3.2.2 逐项拆解](#user-content-s3-2-2)
- [3.3 KL 项的无偏估计器](#user-content-s3-3)
  - [3.3.1 问题陈述](#user-content-s3-3-1)
  - [3.3.2 Schulman k3 估计器及其无偏性证明](#user-content-s3-3-2)
  - [3.3.3 非负性证明](#user-content-s3-3-3)
  - [3.3.4 与其他估计器的对比](#user-content-s3-3-4)
  - [3.3.5 k3 估计器的梯度分析](#user-content-s3-3-5)
- [3.4 GRPO 与 PPO 的逐项对照](#user-content-s3-4)
- [3.5 完整算法伪代码](#user-content-s3-5)
  - [3.5.1 关键超参与典型取值](#user-content-s3-5-1)
  - [3.5.2 K=1 时 clip 失效的关键观察](#user-content-s3-5-2)
- [3.6 计算与显存分析](#user-content-s3-6)
  - [3.6.1 训练中需驻留的模型](#user-content-s3-6-1)
  - [3.6.2 显存粗略分解](#user-content-s3-6-2)
  - [3.6.3 时间成本的转移](#user-content-s3-6-3)
  - [3.6.4 何时 GRPO 更划算](#user-content-s3-6-4)
- [3.7 一个极简的数值算例](#user-content-s3-7)
  - [3.7.1 二值奖励、半对半错](#user-content-s3-7-1)
  - [3.7.2 全对：梯度消失](#user-content-s3-7-2)
  - [3.7.3 三对一错：不对称惩罚](#user-content-s3-7-3)
- [3.8 本章小结与自测](#user-content-s3-8)
  - [要点回顾](#user-content-h100)
  - [自测题](#user-content-h101)
- [4.1 为什么奖励设计是 GRPO 的第一性问题](#user-content-s4-1)
- [4.2 两类奖励源的严格区分](#user-content-s4-2)
  - [4.2.1 奖励模型（Reward Model, RM）](#user-content-s4-2-1)
  - [4.2.2 规则/可验证奖励（Rule-based / Verifiable Reward）](#user-content-s4-2-2)
  - [4.2.3 两类奖励的对比](#user-content-s4-2-3)
  - [4.2.4 为什么 GRPO 的成功集中在规则奖励场景](#user-content-s4-2-4)
- [4.3 RLVR 范式](#user-content-s4-3)
  - [4.3.1 定义与数据构成](#user-content-s4-3-1)
  - [4.3.2 RLVR 与经典 RLHF 的关键差异](#user-content-s4-3-2)
  - [4.3.3 代表性工作](#user-content-s4-3-3)
- [4.4 Outcome Reward 与 Process Reward 的数学区别](#user-content-s4-4)
  - [4.4.1 Outcome Reward（结果奖励）](#user-content-s4-4-1)
  - [4.4.2 Process Reward（过程奖励）](#user-content-s4-4-2)
  - [4.4.3 当前主流选择](#user-content-s4-4-3)
- [4.5 Reward Hacking 的机理与表现](#user-content-s4-5)
  - [4.5.1 形式化定义](#user-content-s4-5-1)
  - [4.5.2 RM 场景的典型 Hacking](#user-content-s4-5-2)
  - [4.5.3 RLVR 场景的 Hacking](#user-content-s4-5-3)
  - [4.5.4 缓解手段的机理](#user-content-s4-5-4)
- [4.6 二值奖励下 GRPO 的特殊动力学](#user-content-s4-6)
  - [4.6.1 组内统计量的解析表达](#user-content-s4-6-1)
  - [4.6.2 信号强度的退化分析](#user-content-s4-6-2)
  - [4.6.3 数值表](#user-content-s4-6-3)
  - [4.6.4 两个重要实践结论](#user-content-s4-6-4)
- [4.7 奖励的尺度、稀疏性与组内归一化的相互作用](#user-content-s4-7)
  - [4.7.1 尺度不变性的证明](#user-content-s4-7-1)
  - [4.7.2 KL 项打破尺度不变性](#user-content-s4-7-2)
  - [4.7.3 多目标奖励的组合](#user-content-s4-7-3)
- [4.8 奖励与 GRPO 假设的一致性检查清单](#user-content-s4-8)
- [4.9 本章小结与自测](#user-content-s4-9)
  - [要点总结](#user-content-h133)
  - [自测题](#user-content-h134)
- [5.1 为什么 GRPO 会有系统性偏差](#user-content-s5-1)
- [5.2 长度偏差（response-level length bias）的严格推导](#user-content-s5-2)
  - [5.2.1 从 GRPO 目标函数的内层求和入手](#user-content-s5-2-1)
  - [5.2.2 梯度分析](#user-content-s5-2-2)
  - [5.2.3 分情况讨论：不对称的后果](#user-content-s5-2-3)
  - [5.2.4 为什么这在数学推理中有害](#user-content-s5-2-4)
  - [5.2.5 Dr.GRPO 的修正：去掉 $`(1/|o_i|)`$](#user-content-s5-2-5)
  - [5.2.6 长度偏差也存在于开源 PPO 实现](#user-content-s5-2-6)
- [5.3 难度偏差（question-level difficulty bias）的严格推导](#user-content-s5-3)
  - [5.3.1 从组内 std 归一化入手](#user-content-s5-3-1)
  - [5.3.2 与第 4 章信号强度分析的矛盾](#user-content-s5-3-2)
  - [5.3.3 二值奖励下的严格刻画](#user-content-s5-3-3)
  - [5.3.4 Dr.GRPO 的修正：去掉 $`(1/\text{std})`$](#user-content-s5-3-4)
  - [5.3.5 数值稳定性与零梯度的澄清](#user-content-s5-3-5)
- [5.4 loss 聚合方式（token-mean vs sequence-mean）](#user-content-s5-4)
  - [5.4.1 三种常见聚合方式](#user-content-s5-4-1)
  - [5.4.2 一个实践中的微妙辨析](#user-content-s5-4-2)
  - [5.4.3 后续研究](#user-content-s5-4-3)
- [5.5 Dr.GRPO 的完整修正版目标函数](#user-content-s5-5)
  - [5.5.1 合并两处修正](#user-content-s5-5-1)
  - [5.5.2 与标准策略梯度的关系](#user-content-s5-5-2)
  - [5.5.3 实验结论（定性）](#user-content-s5-5-3)
- [5.6 其他被讨论的 GRPO 偏差与争议](#user-content-s5-6)
  - [5.6.1 Advantage 的符号-长度耦合](#user-content-s5-6-1)
  - [5.6.2 二值奖励下组均值中心化的梯度枯竭](#user-content-s5-6-2)
  - [5.6.3 过校准与过度自信](#user-content-s5-6-3)
  - [5.6.4 On-policy vs Off-policy GRPO](#user-content-s5-6-4)
- [5.7 偏差修正的取舍与争议](#user-content-s5-7)
  - [5.7.1 "无偏"与"长度不变"的不可能性](#user-content-s5-7-1)
  - [5.7.2 实践建议](#user-content-s5-7-2)
- [5.8 本章小结与自测](#user-content-s5-8)
  - [要点总结](#user-content-h167)
  - [自测题](#user-content-h168)
- [6.1 DAPO 的定位与整体思路](#user-content-s6-1)
- [6.2 Clip-Higher：解耦上下裁剪阈值](#user-content-s6-2)
  - [6.2.1 熵崩塌的定义与危害](#user-content-s6-2-1)
  - [6.2.2 对称 clip 如何加速熵崩塌](#user-content-s6-2-2)
  - [6.2.3 Clip-Higher 的解法](#user-content-s6-2-3)
  - [6.2.4 为什么下界不一起放宽](#user-content-s6-2-4)
  - [6.2.5 对诊断指标的影响](#user-content-s6-2-5)
- [6.3 Dynamic Sampling：动态采样过滤无效 prompt](#user-content-s6-3)
  - [6.3.1 动机：零梯度样本的算力浪费](#user-content-s6-3-1)
  - [6.3.2 DAPO 的动态采样机制](#user-content-s6-3-2)
  - [6.3.3 为什么这有效：隐式课程学习](#user-content-s6-3-3)
  - [6.3.4 代价与注意点](#user-content-s6-3-4)
  - [6.3.5 伪代码](#user-content-s6-3-5)
- [6.4 Token-Level Policy Gradient Loss](#user-content-s6-4)
  - [6.4.1 动机：序列内平均的长度偏差](#user-content-s6-4-1)
  - [6.4.2 DAPO 的 token-level 聚合](#user-content-s6-4-2)
  - [6.4.3 与 Dr.GRPO 的关系](#user-content-s6-4-3)
  - [6.4.4 对 long-CoT 的关键意义](#user-content-s6-4-4)
- [6.5 Overlong Reward Shaping：超长奖励塑形](#user-content-s6-5)
  - [6.5.1 动机：截断带来的奖励噪声](#user-content-s6-5-1)
  - [6.5.2 DAPO 的两部分解法](#user-content-s6-5-2)
  - [6.5.3 软惩罚的数学动机](#user-content-s6-5-3)
  - [6.5.4 堵住 reward hacking 路径](#user-content-s6-5-4)
- [6.6 DAPO 移除 KL 项的决策](#user-content-s6-6)
- [6.7 DAPO 完整目标函数与算法](#user-content-s6-7)
  - [6.7.1 完整目标函数](#user-content-s6-7-1)
  - [6.7.2 算法伪代码](#user-content-s6-7-2)
  - [6.7.3 DAPO 相对 GRPO 的改动总结](#user-content-s6-7-3)
- [6.8 DAPO 的工程系统与可复现性](#user-content-s6-8)
- [6.9 本章小结与自测](#user-content-s6-9)
  - [要点回顾](#user-content-h200)
  - [自测题](#user-content-h201)
- [7.1 变体爆炸的背景与分类框架](#user-content-s7-1)
  - [7.1.1 为什么 GRPO 变体如此之多](#user-content-s7-1-1)
  - [7.1.2 四组件分类框架](#user-content-s7-1-2)
- [7.2 GSPO：序列级重要性比率的数学动机](#user-content-s7-2)
  - [7.2.1 Token 级 IS 比率的三重问题](#user-content-s7-2-1)
  - [7.2.2 GSPO 的序列级比率](#user-content-s7-2-2)
  - [7.2.3 梯度分析：GSPO vs GRPO](#user-content-s7-2-3)
  - [7.2.4 ε 的量级差异](#user-content-s7-2-4)
  - [7.2.5 GSPO 的实践影响](#user-content-s7-2-5)
- [7.3 CISPO：保留全部 Token 梯度](#user-content-s7-3)
  - [7.3.1 Hard Clip 的信息丢弃问题](#user-content-s7-3-1)
  - [7.3.2 CISPO 的解法：Detached IS 权重](#user-content-s7-3-2)
  - [7.3.3 梯度对比：CISPO vs PPO/GRPO](#user-content-s7-3-3)
  - [7.3.4 为什么这对长 CoT 重要](#user-content-s7-3-4)
- [7.4 其他值得了解的变体](#user-content-s7-4)
  - [7.4.1 RLOO](#user-content-s7-4-1)
  - [7.4.2 Dr.GRPO](#user-content-s7-4-2)
  - [7.4.3 GVPO](#user-content-s7-4-3)
  - [7.4.4 BAPO](#user-content-s7-4-4)
  - [7.4.5 SAPO](#user-content-s7-4-5)
  - [7.4.6 长度偏差的后续工作](#user-content-s7-4-6)
  - [7.4.7 关于变体数量的声明](#user-content-s7-4-7)
- [7.5 谱系图与演化逻辑](#user-content-s7-5)
  - [7.5.1 演化树](#user-content-s7-5-1)
  - [7.5.2 时间线表格](#user-content-s7-5-2)
- [7.6 选型建议](#user-content-s7-6)
  - [7.6.1 决策路径](#user-content-s7-6-1)
  - [7.6.2 横向对比大表](#user-content-s7-6-2)
  - [7.6.3 务实建议](#user-content-s7-6-3)
- [7.7 本章小结与自测](#user-content-s7-7)
  - [要点回顾](#user-content-h233)
  - [自测题](#user-content-h234)
- [8.1 框架选型与上手路径](#user-content-s8-1)
  - [8.1.1 主流框架对比](#user-content-s8-1-1)
  - [8.1.2 TRL 最小可运行示例](#user-content-s8-1-2)
- [8.2 从零实现 GRPO 的核心逻辑](#user-content-s8-2)
  - [8.2.1 奖励函数：规则奖励与防 hacking](#user-content-s8-2-1)
  - [8.2.2 Rollout 与组采样：old_logprob 的正确获取](#user-content-s8-2-2)
  - [8.2.3 优势计算：严格按 prompt 分组](#user-content-s8-2-3)
  - [8.2.4 将序列级优势广播到每个 token](#user-content-s8-2-4)
  - [8.2.5 损失计算：比率、clip、k3 KL、聚合](#user-content-s8-2-5)
  - [8.2.6 Mask 的正确处理](#user-content-s8-2-6)
- [8.3 显存、并行与 Rollout 加速](#user-content-s8-3)
  - [8.3.1 显存分解](#user-content-s8-3-1)
  - [8.3.2 省显存的手段及其代价](#user-content-s8-3-2)
  - [8.3.3 并行策略](#user-content-s8-3-3)
  - [8.3.4 Rollout 与训练的耦合方式](#user-content-s8-3-4)
- [8.4 训练诊断指标](#user-content-s8-4)
  - [8.4.1 指标速查表](#user-content-s8-4-1)
  - [8.4.2 GRPO 特有的诊断逻辑](#user-content-s8-4-2)
  - [8.4.3 症状驱动的排查顺序清单](#user-content-s8-4-3)
- [8.5 常见坑清单](#user-content-s8-5)
- [8.6 学习资源与进阶路径](#user-content-s8-6)
  - [8.6.1 论文清单（按推荐阅读顺序）](#user-content-s8-6-1)
  - [8.6.2 代码库](#user-content-s8-6-2)
  - [8.6.3 一条务实的学习路径](#user-content-s8-6-3)
- [8.7 本章小结与全文档收尾](#user-content-s8-7)
  - [学完这两份文档之后该做什么](#user-content-h261)

</details>

<!-- toc:end -->


> 从 PPO 到组相对策略优化（Group Relative Policy Optimization）的完整理论路径 —— 《PPO 系统学习文档》的续篇

---

<a id="h2"></a>

## 关于这份文档

这份文档是《PPO 系统学习文档》的续篇，目标是：**让一个已经理解了 PPO 全部推导的人，能够真正理解 GRPO 为什么可以去掉 critic、它在数学上损失了什么换来了什么、它的目标函数每一项怎么来的，以及 2024–2026 年间围绕它爆发的大量变体各自在改什么。**

写作定位与上一份文档一致：**偏数学与理论推导**。坚持回答三个问题：

- 这个公式**从哪里出发**推导出来？
- 中间做了**什么近似或假设**？近似掉的那部分去了哪里？
- 为什么**必须是这个形式**，换一种写法会出什么问题？

凡是实现与理论有出入的地方（论文公式与社区实现的差异、归一化位置、KL 项的位置、clip 是否生效），文档都会显式标注。

**阅读前提**：本文档假定你已读完《PPO 系统学习文档》，尤其是第 3 章（策略梯度定理与基线）、第 4 章（GAE）、第 5 章（重要性采样与 TRPO）、第 6 章（PPO-Clip 与 RLHF 应用）。文中大量推导直接建立在这些结论之上，不再重复证明。

文档由 8 章组成，采用依赖链结构：**后一章直接使用前一章的结论**。

---

<a id="h3"></a>

## 全文逻辑主线

GRPO 不是一堆工程技巧的堆砌，它源于一个非常具体的观察，然后引发一连串后果。整份文档沿着这条链展开：

```
PPO 需要 critic，而 critic 在 LLM 场景下又贵又难训
          ↓  （第 1 章：四个负担 + LLM RL 的 MDP 视角）
关键观察：奖励是 outcome-level 的，所有 token 共享同一个回报值
          ↓  （第 1 章：G_t ≡ R(q,o)，于是 critic 的状态相关性优势被削弱）
既然 V^π(q) = E_{o~π}[R(q,o)]，那就可以对同一 prompt 采样 G 次直接估计它
          ↓  （第 2 章：组均值是 V^π 的无偏 MC 估计，方差 σ²(q)/G）
用「G 倍采样算力」换掉「一个需要训练和存储的 critic」
          ↓  （第 2 章：留一法严格无偏；全组均值优势 = 留一法优势 × (G-1)/G）
写出完整目标函数：逐 token 比率 × 序列级常量优势 + 显式 KL 项
          ↓  （第 3 章：k3 无偏 KL 估计器；K=1 时 clip 失效退化为 REINFORCE）
但奖励的性质现在成了唯一的学习信号来源
          ↓  （第 4 章：RLVR、二值奖励下 p≈0.5 信号最强、reward hacking）
去掉 critic 的同时，也去掉了 token 级信用分配，且两个归一化项引入偏差
          ↓  （第 5 章：长度偏差「对就简短、错就啰嗦」；难度偏差「边缘题被放大」→ Dr.GRPO）
长链推理场景下还有熵崩塌、零梯度样本浪费算力、超长截断噪声
          ↓  （第 6 章：DAPO 四项技术各治一病）
token 级比率与序列级奖励的粒度错配，在 MoE 上直接导致训练崩溃
          ↓  （第 7 章：GSPO 序列级比率 / CISPO 保全部 token 梯度 / 四类分类框架）
理论落地时到处是与公式不一致的细节
          ↓  （第 8 章：TRL/verl、显存与并行、诊断指标、14 条常见坑）
```

一句话概括 GRPO 的定位：**它用「对同一 prompt 的 G 次采样」这个蒙特卡洛估计量，替代了 PPO 中需要训练的 critic 网络；代价是失去了 token 级信用分配能力，收益是省掉一半显存与整个 critic 的训练开销。** 这个交易在 outcome reward + 可验证奖励的推理任务上极其划算，在 process reward 或奖励昂贵的场景则未必——第 2.6 节给出了明确的适用边界判断。

---

<a id="h4"></a>

## 各章依赖关系

| 章节 | 标题 | 依赖 | 核心产出 |
|---|---|---|---|
| 第 1 章 | 从 PPO 到 GRPO：动机与问题设定 | PPO 文档全部 | LLM RL 的 MDP 视角、PPO 四负担、PPO↔GRPO 对照表、适用边界 |
| 第 2 章 | 组内基线的理论基础 | 第 1 章；PPO 文档第 3 章 | 组均值基线无偏性与方差、留一法证明、(G-1)/G 缩放定理、与 critic 的六维对比 |
| 第 3 章 | GRPO 算法本体 | 第 1、2 章；PPO 文档第 6 章 | 完整目标函数逐项拆解、k3 KL 估计器证明、伪代码与超参、显存定量分析、数值算例 |
| 第 4 章 | 奖励信号与 RLVR | 第 1–3 章 | RM vs 规则奖励、outcome/process reward、reward hacking 机理、二值奖励动力学、尺度不变性 |
| 第 5 章 | GRPO 的已知偏差与 Dr.GRPO 修正 | 第 2–4 章 | 长度偏差与难度偏差的严格推导、loss 聚合方式、Dr.GRPO 修正版 |
| 第 6 章 | DAPO | 第 2–5 章；PPO 文档第 6 章 | clip-higher、动态采样、token-level loss、超长软惩罚、移除 KL 的决策 |
| 第 7 章 | 序列级方法与变体谱系 | 第 1–6 章；PPO 文档第 5 章 | 变体四类分类框架、GSPO、CISPO、演化树、选型决策路径、横向对比大表 |
| 第 8 章 | 工程实现与实践 | 全部；PPO 文档第 7 章 | 框架选型、可运行参考实现、显存与并行、诊断指标、14 条常见坑、学习资源 |

---

<a id="h5"></a>

## 阅读路线

**完整路线（推荐）**：第 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 章顺序读完。

**最短路线（只想搞懂 GRPO 公式）**：第 1 章（MDP 视角与四负担）→ 第 3 章全部 → 第 2.2、2.4 节（组均值基线与 (G-1)/G 缩放）。这条线能让你完整理解 GRPO 在算什么、为什么可以不算 critic。

**实践优先路线（想先跑起来）**：第 8.1、8.2 节（框架选型与参考实现）→ 第 3 章（对照公式）→ 第 4.6 节（数据难度配比，直接决定训练能否进行）→ 第 8.4、8.5 节（诊断与排坑）→ 第 5、6 章（观察到长度失控或熵崩塌时再读）。

**只关心前沿与选型**：第 1.5 节（对照表）→ 第 7 章全部 → 第 5、6 章小结。第 7.1 节的四类分类框架和 7.6 节的横向对比大表是为你准备的。

**关心理论严谨性（想搞清楚 GRPO 到底有没有偏）**：第 2 章全部 → 第 5 章全部 → 第 3.5 节（K=1 时 clip 失效）→ 第 7.2 节（GSPO 梯度分析）。

---

<a id="h6"></a>

## 全文通用符号约定

以下为贯穿全文的记号。**与《PPO 系统学习文档》保持一致**；各章可能有局部补充记号，以章内首次定义为准。

<a id="h7"></a>

### 基本对象

| 符号 | 含义 |
|---|---|
| $`q`$ | prompt（提示词），在 MDP 视角下即初始状态 $`s_0`$ |
| $`o`$ / $`o_i`$ | 一个完整回答（response / completion），即一条轨迹 |
| $`o_{i,t}`$ | 第 $`i`$ 个回答的第 $`t`$ 个 token |
| $`o_{i,<t}`$ | 第 $`i`$ 个回答中 $`t`$ 之前的所有 token（前缀） |
| $`\|o_i\|`$ | 第 $`i`$ 个回答的 token 数（长度） |
| $`G`$ | 组大小（group size），对同一 prompt 采样的回答数 |
| $`B`$ | prompt 批大小（一个 batch 里的 prompt 数） |
| $`K`$ | 内层更新的 epoch 数 |

<a id="h8"></a>

### 策略与模型

| 符号 | 含义 |
|---|---|
| $`\pi_\theta`$ | 当前策略，即正在训练的语言模型 |
| $`\pi_{\theta_{old}}`$ | 采样时的旧策略（rollout 用的模型快照） |
| $`\pi_{ref}`$ | 参考模型（reference model），通常是 SFT 后的初始模型，仅用于 KL 约束 |
| $`R_\phi`$ | 奖励模型（reward model），若使用 RM 而非规则奖励 |

<a id="h9"></a>

### 奖励与优势

| 符号 | 含义 |
|---|---|
| $`r_i`$ / $`R(q, o_i)`$ | 第 $`i`$ 个回答的奖励（标量，整条回答一个分数） |
| $`V^\pi(q)`$ | 从 prompt $`q`$ 出发的期望回报，$`= \mathbb{E}_{o\sim\pi}[R(q,o)]`$ |
| $`\sigma^2(q)`$ | 同一 prompt 下回答奖励的内在方差，$`= \mathrm{Var}_{o\sim\pi}[R(q,o)]`$ |
| $`\hat{A}_i`$ | 第 $`i`$ 个回答的优势估计（序列级常量） |
| $`\hat{A}_{i,t}`$ | token 级优势，GRPO 中 $`\hat{A}_{i,t} = \hat{A}_i`$ 对所有 $`t`$ 成立 |

<a id="h10"></a>

### 优化相关

| 符号 | 含义 |
|---|---|
| $`r_{i,t}(\theta)`$ | 逐 token 重要性采样比率，$`= \pi_\theta(o_{i,t}\|q,o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}\|q,o_{i,<t})`$ |
| $`s_i(\theta)`$ | GSPO 的序列级比率（token 级比率的几何平均，见第 7 章） |
| $`\varepsilon`$ | clip 半径；DAPO 解耦为 $`\varepsilon_{low}`$、$`\varepsilon_{high}`$ |
| $`\beta`$ | KL 惩罚系数 |
| $`D_{KL}[\pi_\theta \| \pi_{ref}]`$ | KL 散度（定义与性质见 PPO 文档第 1 章） |
| $`\gamma`$ | 折扣因子，LLM RL 实践中通常取 $`1`$ |

<a id="h11"></a>

### 记号说明

- **向量与矩阵**：小写粗体为向量，大写粗体为矩阵；本文档以标量推导为主。
- **期望**：$`\mathbb{E}_{x\sim p}[\cdot]`$ 表示对 $`x`$ 在分布 $`p`$ 下取期望；下标省略时表示上下文默认分布。
- **估计量**：带 $`\hat{\cdot}`$ 的量为估计量（如 $`\hat{A}_i`$、$`\hat{D}_{KL}`$），不带者为真值。
- **梯度**：$`\nabla_\theta`$ 表示对策略参数 $`\theta`$ 求梯度。

---

<a id="h12"></a>

## 关于本文档的准确性说明

GRPO 及其变体是一个**演进极快**的领域（2024–2026 年间新变体持续涌现）。本文档遵循以下原则：

1. **论文与实现的差异被显式标注**。最典型的两处：GRPO 原论文（DeepSeekMath）的目标函数**不含 clip**，而社区实现普遍叠加了 PPO 式 clip；KL 项在原论文中写在损失里，部分实现则塑形进奖励。这些差异在第 3 章明确指出。
2. **凡未经核实的数值一律标注「以原论文为准」**。超参、实验分数、论文编号只在有多源确认时给出具体值。
3. **争议被如实呈现**。包括：去掉 std 归一化是否真的更优（第 5.7 节）、「无偏」与「长度不变」能否同时满足（第 5.7 节）、DAPO 移除 KL 是否普适（第 6.6 节）、以及 PPO/GRPO 的成功在多大程度上来自算法本身而非工程细节（承接 PPO 文档第 6.6 节）。
4. **变体覆盖是有选择的**。第 7 章只覆盖影响力最大的几支，并给出一个四类分类框架，使你遇到任何新变体时都能自行定位它改了什么。

---

<a id="chapter-1"></a>

# 第 1 章 从 PPO 到 GRPO：动机与问题设定

<a id="s1-1"></a>

## 1.1 开场定位：GRPO 是什么

组相对策略优化（Group Relative Policy Optimization, GRPO）是近端策略优化（Proximal Policy Optimization, PPO）的一个变体，其核心改动可以用一句话概括：**把「用 critic 网络估计基线」替换为「用同一 prompt 的多次采样结果的组内统计量作基线」，从而彻底去掉 critic 网络**。这一方法由 Shao et al. (2024) 在《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》(arXiv:2402.03300) 中提出，后因被 DeepSeek-R1 采用而在大语言模型（Large Language Model, LLM）后训练（post-training）社区中广为人知。

如果你已经完整阅读了 PPO 文档，那么你对策略梯度定理、广义优势估计（Generalized Advantage Estimation, GAE）、重要性采样比率裁剪（clip）以及信任域约束的全部推导都已掌握。本章的任务不是重复这些内容，而是回答一个更上游的问题：**在 LLM 后训练这个特定场景下，PPO 的哪些设计假设不再自然成立？GRPO 又是如何利用该场景的特殊结构来简化算法的？** 理解这个"为什么"，比记住 GRPO 的公式更重要——公式本身将在第 3 章给出完整推导。

<a id="s1-2"></a>

## 1.2 LLM 后训练的 MDP 视角

要从 PPO 过渡到 GRPO，首先需要把 LLM 的文本生成过程严格表述为一个马尔可夫决策过程（Markov Decision Process, MDP）。这一步是从经典强化学习（Reinforcement Learning, RL）文献到 LLM 后训练文献的关键桥梁。

<a id="s1-2-1"></a>

### 1.2.1 状态、动作与策略

考虑一个 LLM 接收用户提问（prompt）$`q`$ 并生成回答（output）$`o = (o_1, o_2, \ldots, o_{|o|})`$ 的过程。我们将其建模为如下 MDP：

- **状态（state）**：在时刻 $`t`$，状态为已生成的前缀，即 $`s_t = (q, o_{<t}) = (q, o_1, o_2, \ldots, o_{t-1})`$。特别地，初始状态 $`s_0 = q`$（尚未生成任何 token）。
- **动作（action）**：在时刻 $`t`$，动作为选择下一个 token $`o_t`$，动作空间为整个词表 $`\mathcal{V}`$。
- **策略（policy）**：参数化的语言模型 $`\pi_\theta(o_t \mid q, o_{<t})`$，即在给定 prompt 和已生成前缀的条件下，输出词表上的概率分布。这正是自回归语言模型的标准定义。
- **轨迹（trajectory）**：一条完整的回答 $`o = (o_1, \ldots, o_{|o|})`$ 就是一条从 $`s_0 = q`$ 出发、经过 $`|o|`$ 步转移后到达终止状态的轨迹。

注意这里的"时刻"对应的是 token 生成的步数，而非物理时间。一条轨迹的长度 $`|o|`$ 是可变的，由模型的停止条件（生成 EOS token 或达到最大长度）决定。

<a id="s1-2-2"></a>

### 1.2.2 奖励的特殊性：outcome-level 稀疏奖励

LLM 后训练中的奖励信号与经典 RL 有本质区别。在机器人控制、游戏等经典场景中，奖励通常在每一步都有定义（dense reward），或者至少在若干中间里程碑处给出（shaped reward）。但在 LLM 后训练中，**绝大多数情况下奖励只在最后一个 token 给出**，这被称为结果奖励（outcome reward）或稀疏奖励（sparse reward）。形式化地：

```math
r_t = \begin{cases}
0, & t < |o| \\
R(q, o), & t = |o|
\end{cases}
```

其中 $`R(q, o)`$ 是对整条回答的质量评分，可以来自规则验证器（如数学题的答案校验）、奖励模型（reward model）或人类反馈。中间所有 token 的即时奖励均为零。

这一结构看似简单，却蕴含了一个极其重要的推论。

<a id="s1-2-3"></a>

### 1.2.3 关键推论：所有 token 共享同一个回报值

定义从时刻 $`t`$ 出发的回报（return / reward-to-go）为：

```math
G_t = \sum_{k=t}^{|o|} \gamma^{k-t} r_k
```

其中 $`\gamma \in [0, 1]`$ 是折扣因子（discount factor）。将上述稀疏奖励结构代入：当 $`t < |o|`$ 时，$`r_t = r_{t+1} = \cdots = r_{|o|-1} = 0`$，唯一非零项是 $`r_{|o|} = R(q, o)`$，因此：

```math
G_t = \gamma^{|o|-t} \cdot R(q, o), \quad \forall \, t \leq |o|
```

在实践中，LLM RL 通常取 $`\gamma = 1`$（理由见下一小节），此时上式进一步简化为：

```math
G_t = R(q, o), \quad \forall \, t \leq |o|
```

**这意味着：在同一条轨迹内，无论从哪个 token 位置 $`t`$ 出发，回报值都完全相同，都等于整条回答的最终奖励 $`R(q, o)`$。**

这个结论的直接后果是：如果我们定义优势函数（advantage function）为 $`A(s_t, a_t) = Q(s_t, a_t) - V(s_t)`$，那么在 $`\gamma = 1`$ 且 outcome reward 的设定下，$`Q(s_t, a_t)`$ 对所有 $`t`$ 都等于同一个值 $`R(q, o)`$（因为从任意位置出发的期望回报都由最终奖励决定）。于是，**整条序列的优势估计可以退化为一个序列级的标量**，而不需要像经典 RL 那样在每个时刻分别计算不同的优势值。这正是 GRPO 能够为整条序列的每个 token 赋予同一个优势值 $`\hat{A}_{i,t} = \hat{A}_i`$ 的理论依据。相比之下，PPO 在经典 RL 中依赖 GAE 递推得到每个时刻不同的优势估计（见 PPO 文档第 4 章），这在 outcome reward 场景下其实是一种过度设计。

<a id="s1-2-4"></a>

### 1.2.4 折扣因子 $`\gamma`$ 的处境

在经典 RL 中，$`\gamma < 1`$ 有两个作用：保证无穷 horizon 下回报收敛，以及编码时间偏好（即时奖励优于延迟奖励）。但在 LLM 后训练中：

1. **Episode 天然有限**：每条回答的长度 $`|o|`$ 是有限的（通常上限为数千至数万 token），不存在无穷 horizon 的收敛问题。
2. **无真正的时间偏好**：我们并不认为回答的前半部分比后半部分"更有价值"；评价标准是整条回答的整体质量。

因此，LLM RL 实践中普遍取 $`\gamma = 1`$。此时 GAE 的递推公式（见 PPO 文档第 4 章）：

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad \hat{A}_t^{\text{GAE}} = \sum_{l=0}^{|o|-t} (\gamma \lambda)^l \delta_{t+l}
```

在 $`\gamma = 1`$ 且 $`r_t = 0`$（$`t < |o|`$）的条件下退化为：

```math
\delta_t = V(s_{t+1}) - V(s_t) \quad (t < |o|), \quad \delta_{|o|} = R(q,o) - V(s_{|o|})
```

GAE 变成了一系列相邻价值函数差分的加权和。如果 critic 拟合完美（$`V(s_t) = \mathbb{E}[R(q,o) \mid s_t]`$），这些差分之和恰好 telescoping 消去，$`\hat{A}_t^{\text{GAE}} = R(q,o) - V(s_0)`$，与直接蒙特卡洛估计一致。但如果 critic 有误差，这些误差会沿递推累积，反而引入额外的偏差和方差。**当 $`\gamma = 1`$ 且奖励集中在终点时，critic 对优势估计的"方差缩减"收益被其自身拟合误差所抵消，甚至可能适得其反。** 这从另一个角度削弱了 critic 在 LLM 场景下的必要性。

<a id="s1-3"></a>

## 1.3 PPO 在 LLM 场景下的四个负担

理解了 LLM 后训练的 MDP 结构后，我们可以精确地分析 PPO 在此场景下面临的具体困难。以下逐条量化分析四个核心负担。

<a id="s1-3-1"></a>

### 1.3.1 显存负担：四模型并存

PPO 在 LLM 后训练中需要同时在 GPU 显存中维护四个模型：

| 模型 | 作用 | 参数量级 | 是否可省 |
|------|------|----------|----------|
| Policy（actor）$`\pi_\theta`$ | 生成回答、计算策略梯度 | 与基座模型相同（7B–数百B） | 不可省 |
| Reference $`\pi_{\text{ref}}`$ | 计算 KL 散度以防止策略偏离过远 | 与 policy 相同 | 若使用规则奖励且不需要 KL 约束则可省，但实践中几乎总是保留 |
| Critic（value）$`V_\phi`$ | 估计状态价值以计算优势 | 通常与 policy 同量级 | **GRPO 的核心贡献：可省** |
| Reward model $`R_\psi`$ | 对生成的回答打分 | 通常与 policy 同量级或略小 | 若使用规则奖励（如数学验证器）则可省 |

在最典型的配置下（使用奖励模型 + KL 约束），四个模型全部需要驻留显存。对于 7B 参数的模型，仅模型权重（FP16）就需要约 $`4 \times 14 \text{GB} = 56 \text{GB}`$，再加上优化器状态（AdamW 需额外 2 倍权重大小用于 momentum 和 variance）、激活值和 KV cache，实际显存需求远超此数。

GRPO 去掉 critic 后，模型数从 4 降为 3。如果使用规则奖励（rule-based reward）替代奖励模型，则进一步降为 2（policy + reference）。**去掉 critic 节省的不仅是模型权重本身的显存，还包括其优化器状态和训练过程中的激活值开销。** 对于大模型而言，这一节省往往是能否在有限硬件上完成训练的决定性因素。

<a id="s1-3-2"></a>

### 1.3.2 训练负担：critic 与 actor 的资源竞争

Critic 不是免费附赠的——它自身也需要训练。在 PPO 的每个更新周期中，critic 通过最小化价值函数的均方误差损失来更新参数 $`\phi`$：

```math
\mathcal{L}_{\text{critic}}(\phi) = \mathbb{E}_{(s_t, G_t)} \left[ \left( V_\phi(s_t) - G_t \right)^2 \right]
```

这带来两个问题：

1. **优化资源竞争**：actor 和 critic 共享计算资源（GPU 时间、显存带宽）。在实际训练中，critic 的损失收敛速度往往慢于 actor，导致整体训练效率受限于 critic。
2. **误差传播**：critic 拟合不准会通过优势估计直接污染 actor 的梯度方向。正如 PPO 文档第 6 章讨论的 explained_variance 指标所示，当 critic 的解释方差较低时，优势估计的信噪比很差，actor 的更新方向几乎是随机的。在 LLM 尺度下，由于序列极长且奖励稀疏，critic 的拟合难度远高于经典 RL 任务，这一问题尤为突出。

<a id="s1-3-3"></a>

### 1.3.3 价值函数在长序列上的固有困难

LLM 的回答长度可达数千甚至数万 token。critic 需要在每一个 token 位置 $`t`$ 都给出准确的价值估计 $`V(s_t) = \mathbb{E}[R(q, o) \mid q, o_{<t}]`$。这本质上是一个条件期望估计问题，其难度随以下条件增长：

- **序列长度**：越长意味着 critic 需要整合越多的上下文信息才能做出准确预测。
- **奖励稀疏性**：由于中间没有奖励信号，critic 无法利用逐步的反馈来校正自己的估计，只能依赖终端奖励的反向传播。
- **语义复杂性**：$`V(s_t)`$ 依赖于对"当前已生成的部分回答距离正确答案还有多远"的理解，这本身就是一个极其困难的推理问题。

更严重的是，critic 的估计误差会沿 GAE 递推累积。回顾 GAE 公式，$`\hat{A}_t^{\text{GAE}}`$ 是 $`\delta_t, \delta_{t+1}, \ldots`$ 的加权和，每一项 $`\delta_k`$ 都包含 $`V(s_k)`$ 和 $`V(s_{k+1})`$ 的估计误差。在长序列下，即使单步误差很小，累积效应也可能使优势估计严重失真。

<a id="s1-3-4"></a>

### 1.3.4 超参与工程负担

PPO 在 LLM 尺度下的超参数敏感度远高于经典 RL：

- **Critic 专属超参**：critic 的学习率、warmup 步数、与 actor 的更新比例（如每更新 actor 一次需更新 critic 几次）都需要单独调节。
- **PPO 核心超参**：clip 范围 $`\epsilon`$、KL 惩罚系数 $`\beta`$、GAE 的 $`\lambda`$ 等在 LLM 尺度下的最优值与经典 RL 差异很大，且对训练稳定性影响极大。
- **工程复杂度**：四模型的分布式并行策略（data parallelism、tensor parallelism、pipeline parallelism）的组合远比三模型或两模型复杂，调试成本高。

这些负担叠加在一起，使得 PPO 在 LLM 后训练中的工程门槛极高。GRPO 通过去掉 critic，一举消除了上述大部分问题。

<a id="s1-4"></a>

## 1.4 GRPO 的核心洞察：以采样换模型

经过前两节的铺垫，GRPO 的核心洞察现在应该显得自然而必然：

> **既然 LLM 后训练中的奖励是 outcome-level 的（整条序列一个分数），那么"这条回答好不好"是一个可以通过多次采样直接经验估计的量——不需要学一个 critic 去预测它。**

具体来说，对于同一个 prompt $`q`$，我们从当前策略 $`\pi_\theta`$ 采样 $`G`$ 个回答 $`\{o_1, o_2, \ldots, o_G\}`$，分别获得奖励 $`\{r_1, r_2, \ldots, r_G\}`$（其中 $`r_i = R(q, o_i)`$）。回忆状态价值函数的定义：

```math
V^\pi(s_0 = q) = \mathbb{E}_{o \sim \pi(\cdot \mid q)} [R(q, o)]
```

组均值 $`\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i`$ 正是 $`V^\pi(q)`$ 的一个蒙特卡洛（Monte Carlo）估计量。根据大数定律，当 $`G \to \infty`$ 时 $`\bar{r} \to V^\pi(q)`$（几乎处处收敛）。对于有限的 $`G`$，$`\bar{r}`$ 是 $`V^\pi(q)`$ 的无偏估计（unbiased estimator），即 $`\mathbb{E}[\bar{r}] = V^\pi(q)`$。组内标准差 $`\sigma_r = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \bar{r})^2}`$ 则可用于归一化，消除不同 prompt 之间奖励尺度的差异。

于是，第 $`i`$ 条回答的优势估计为：

```math
\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}
```

这就是"组相对"（Group Relative）名称的由来：每条回答的优势是相对于同组其他回答的统计量而言的。

**GRPO 的本质是一个 compute-for-memory 的交易：用 $`G`$ 倍的推理算力，换掉一个需要训练和存储的 critic 网络。** 这个交易在以下条件下划算：

1. **推理成本相对可控**：LLM 的推理（inference）比训练（training）便宜得多，因为推理不需要反向传播和优化器状态。$`G`$ 次推理的额外开销通常远小于训练一个同量级 critic 的开销。
2. **$`G`$ 不需要太大**：实践表明 $`G = 4 \sim 16`$ 即可获得稳定的优势估计（详见第 2 章的方差分析）。
3. **任务有明确的奖励信号**：当奖励是二值的（对/错）或有明确定义的评分规则时，少量采样即可提供有意义的区分度。

关于组均值基线的无偏性证明、方差分析以及与 REINFORCE 基线和留一法（Leave-One-Out, RLOO）基线的理论比较，将在第 2 章详细展开。

<a id="s1-5"></a>

## 1.5 PPO 与 GRPO 的系统对比

下表逐维度对比 PPO 与 GRPO 的关键差异，作为全文档的导航图：

| 维度 | PPO | GRPO |
|------|-----|------|
| 优势估计方式 | GAE：基于 critic 的 TD 残差加权求和 | 组内归一化：$`\hat{A}_i = (r_i - \bar{r}) / \sigma_r`$ |
| 是否需要 critic | 是，且需与 actor 同步训练 | 否，完全去掉 |
| 是否需要 GAE | 是，需选择 $`\lambda`$ | 否 |
| KL 项的位置与形式 | 通常在奖励中塑形：$`r'_t = r_t - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]`$ | 直接写入损失函数作为独立正则项（见第 3 章） |
| Clip 机制 | 对重要性采样比率 $`\rho_t = \pi_\theta / \pi_{\text{old}}`$ 做对称裁剪 $`[\!1-\epsilon, 1+\epsilon\!]`$ | 同样对逐 token 比率做裁剪，但优势为序列级标量（见第 3 章） |
| 每 prompt 采样数 | 通常 1（on-policy）或少数几条 | $`G`$ 条（典型值 4–16） |
| 显存中的模型数 | 4（policy + ref + critic + RM）或 3（规则奖励时） | 3（policy + ref + RM）或 2（规则奖励时） |
| 适用场景 | 通用 RL，dense/sparse reward 均可 | 最适合 outcome-level 可验证奖励的 LLM 任务 |

几点补充说明：

- **KL 项的处理差异**值得特别注意。PPO 文档中讨论了将 KL 惩罚融入奖励的做法（reward shaping），这在理论上等价于修改 MDP 的奖励函数。GRPO 则倾向于将 KL 作为损失函数中的独立正则项，这种处理方式在数学上更清晰，也更容易调节（见第 3 章的完整目标函数）。
- **采样数的差异**反映了两种算法的根本哲学不同：PPO 信赖 critic 提供的"先验"价值估计来减少方差，因此每次只需少量样本；GRPO 不信赖任何 learned baseline，转而通过增加采样数来直接降低蒙特卡洛估计的方差。
- 表中未列出的一个重要区别是**优势估计的粒度**：PPO 的优势是 token-level 的（每个 token 有不同的 $`\hat{A}_t`$），而 GRPO 的优势是 sequence-level 的（同一条序列的所有 token 共享同一个 $`\hat{A}_i`$）。这一区别直接源于 §1.2.3 的推导。

<a id="s1-6"></a>

## 1.6 GRPO 的适用边界

诚实地划定 GRPO 的能力边界，比列举其优点更重要。

<a id="s1-6-1"></a>

### 1.6.1 适合的场景

- **有明确可验证奖励的任务**：数学推理（答案可自动验证）、代码生成（可通过测试用例验证）、逻辑推理（有确定性评判标准）。这类任务的奖励信号干净、噪声低，少量采样即可获得有意义的组内区分度。这也是 GRPO 在 RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）范式下最成功的原因（详见第 4 章）。
- **能负担 $`G`$ 倍采样的算力预算**：$`G`$ 倍推理开销虽远小于训练 critic，但当模型极大或 prompt 极多时仍不可忽视。
- **Prompt 分布足够多样**：组内基线的有效性依赖于同一 prompt 下不同回答之间存在有意义的质量差异。如果某个 prompt 的所有采样回答几乎一样好（或一样差），组内标准差趋近于零，优势估计退化。

<a id="s1-6-2"></a>

### 1.6.2 不适合或需谨慎的场景

- **奖励昂贵**：如果每次评分都需要调用一个大型奖励模型或请求人类反馈，$`G`$ 倍采样的奖励计算成本可能成为瓶颈。此时 PPO 的 critic（一次前向即可给出所有 token 的价值估计）反而更高效。
- **无法对同一 prompt 多次采样的场景**：某些任务中每个 prompt 只有唯一的"正确"交互路径，重复采样只会得到近乎相同的回答，组内统计量失去意义。
- **奖励噪声大**：当奖励信号本身方差很高（如人类偏好标注的一致性很低）时，组内均值和标准差都是高噪声估计，优势估计的信噪比会很差。
- **$`G`$ 过小**：这是一个数学事实而非工程建议——当 $`G = 2`$ 时，组内标准差的自由度仅为 1，估计极不稳定；$`\hat{A}_i`$ 只能取 $`\pm 1/\sqrt{2}`$ 两个值，完全丧失了连续优势信号的表达能力。实践中 $`G \geq 4`$ 是最低要求，$`G \geq 8`$ 更为稳妥。

<a id="s1-6-3"></a>

### 1.6.3 与 RLVR 的关系

GRPO 的最大成功来自于它与 RLVR 范式的天然契合。RLVR 的核心思想是：与其训练一个可能产生 reward hacking 的奖励模型，不如直接使用可程序化验证的规则奖励（如数学答案匹配、代码执行结果）。在这种设定下，奖励计算廉价且无噪声，恰好满足 GRPO 对"低成本、高质量奖励信号"的需求。RLVR 的详细讨论见第 4 章。

<a id="s1-7"></a>

## 1.7 全文档阅读路线

本档共 8 章，以下是针对不同目标的推荐阅读路径：

- **只想理解 GRPO 的公式与算法流程** → 第 1 章（本章）+ 第 3 章。本章建立直觉和问题设定，第 3 章给出完整目标函数、伪代码与超参数说明。
- **想理解"去掉 critic 在数学上为什么成立"** → 第 2 章。该章从 REINFORCE 基线理论出发，严格证明组均值基线的无偏性，并与留一法（RLOO）基线进行方差比较。
- **想实际上手训练** → 第 3 章 + 第 8 章。第 3 章提供算法细节，第 8 章覆盖 TRL/verl 框架的工程实现、显存管理、诊断指标与常见坑。
- **关心前沿变体与改进** → 第 5 章（GRPO 的已知偏差与 Dr.GRPO 修正）、第 6 章（DAPO）、第 7 章（GSPO、CISPO 及其他变体的谱系与选型建议）。
- **关心奖励设计与 RLVR** → 第 4 章。涵盖规则奖励 vs 奖励模型、outcome/process reward、reward hacking 等核心议题。

<a id="s1-8"></a>

## 1.8 本章小结与自测

<a id="h33"></a>

### 要点回顾

1. **GRPO 的核心改动**是用同一 prompt 的 $`G`$ 次采样结果的组内统计量替代 critic 网络来估计基线，从而将 PPO 的四模型架构简化为三模型（或两模型）。
2. **LLM 后训练的 MDP 结构**具有特殊性：状态是已生成前缀，动作是下一个 token，奖励几乎总是 outcome-level 稀疏奖励。在 $`\gamma = 1`$ 下，同一条轨迹内所有 token 的回报值相同，这使得序列级优势估计成为可能。
3. **PPO 在 LLM 场景下面临四重负担**：显存（四模型并存）、训练（critic 与 actor 资源竞争）、价值估计困难（长序列 + 稀疏奖励导致 critic 误差累积）、超参与工程复杂度。
4. **GRPO 是以采样换模型的交易**：用 $`G`$ 倍推理成本换取去掉 critic 的收益。该交易在推理便宜、$`G`$ 适中、奖励明确时最为划算。
5. **GRPO 有明确的适用边界**：最适合有可验证奖励的任务（RLVR 范式），不适合奖励昂贵、无法多次采样、奖励噪声大或 $`G`$ 过小的场景。

<a id="h34"></a>

### 自测题

**题目 1**：在 LLM 后训练的 MDP 中，若取 $`\gamma = 0.99`$ 而非 $`\gamma = 1`$，§1.2.3 中"所有 token 共享同一个回报值"的结论是否仍然成立？如果不成立，写出 $`G_t`$ 的表达式，并说明这对 GRPO 的序列级优势假设意味着什么。

参考答案：不成立。当 $`\gamma < 1`$ 时，$`G_t = \gamma^{|o|-t} R(q, o)`$，不同 $`t`$ 对应的 $`G_t`$ 因折扣因子幂次不同而不同。这意味着序列级优势假设不再精确成立，GRPO 需要要么接受由此引入的偏差，要么在优势计算中补偿折扣效应。实践中取 $`\gamma = 1`$ 正是为了避免这一问题。

**题目 2**：假设某 LLM 后训练任务使用 PPO，policy 和 critic 均为 70B 参数模型，reference 和 reward model 均为 7B。粗略估算去掉 critic 后节省的显存占比（仅考虑模型权重，忽略优化器状态和激活值）。

参考答案：总权重 = 70 + 70 + 7 + 7 = 154B 参数。去掉 critic 后 = 70 + 7 + 7 = 84B 参数。节省占比 ≈ (154 - 84) / 154 ≈ 45.5%。注意若同时考虑优化器状态（critic 的 AdamW 需额外 2×70B 参数量的存储），实际节省比例更高。

**题目 3**：为什么在 outcome reward 且 $`\gamma = 1`$ 的设定下，GAE 的方差缩减收益会被 critic 的拟合误差所抵消？请用 GAE 的递推公式简要说明。

参考答案：GAE 中 $`\delta_t = r_t + V(s_{t+1}) - V(s_t)`$。在 outcome reward 下 $`r_t = 0`$（$`t < |o|`$），故 $`\delta_t = V(s_{t+1}) - V(s_t)`$ 完全是 critic 估计的差分。若 critic 有误差 $`\epsilon_t`$，则 $`\delta_t`$ 中包含 $`(\epsilon_{t+1} - \epsilon_t)`$，这些误差沿 GAE 加权求和时不会像真实信号那样 telescoping 消去，而是以随机游走的方式累积，可能导致 $`\hat{A}^{\text{GAE}}`$ 的方差反而大于直接使用蒙特卡洛回报 $`R(q,o)`$ 的方差。

**题目 4**：GRPO 中 $`G = 2`$ 时，$`\hat{A}_i = (r_i - \bar{r}) / \sigma_r`$ 只能取哪两个值？为什么这对训练有害？

参考答案：当 $`G = 2`$ 时，$`\bar{r} = (r_1 + r_2)/2`$，$`\sigma_r = |r_1 - r_2|/2`$（总体标准差）。代入可得 $`\hat{A}_1 = \pm 1/\sqrt{2}`$，$`\hat{A}_2 = \mp 1/\sqrt{2}`$（符号取决于 $`r_1 > r_2`$ 还是 $`r_1 < r_2`$）。优势信号退化为二值，丧失了连续区分度，梯度更新方向的信息量极低，训练效率大幅下降。

**题目 5**：如果一个任务的奖励需要通过调用 GPT-4 级别的模型来计算（每次调用耗时数秒、费用数美分），GRPO 相比 PPO 是否仍有优势？请从 compute-for-memory 交易的角度分析。

参考答案：不一定有优势。GRPO 的核心交易是用 $`G`$ 倍推理换掉 critic。当奖励计算本身很昂贵时，$`G`$ 倍采样意味着 $`G`$ 倍奖励计算成本，这可能远超训练一个 critic 的成本。此时 PPO 的 critic 虽然增加了训练开销，但每次只需一次前向即可给出价值估计，避免了重复的昂贵奖励查询。GRPO 的优势前提是奖励计算廉价（如规则验证器），这正是 RLVR 范式的核心特征。

<a id="chapter-2"></a>

# 第 2 章 组内基线的理论基础

<a id="s2-1"></a>

## 2.1 从 PPO 文档的基线理论出发

在 PPO 文档中（见 PPO 文档第 3 章），我们已经建立了关于策略梯度基线（baseline）的两个基本事实。为避免重复推导，此处仅做简要回顾，随后提出本章要回答的核心问题。

<a id="s2-1-1"></a>

### 2.1.1 基线的两个基本事实

**事实 (a)：基线的无偏性。** 对于任意只依赖状态 $`s`$ 而不依赖动作 $`a`$ 的函数 $`b(s)`$，从策略梯度估计量中减去它不改变梯度的期望值。即：

```math
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s)\right] = b(s) \cdot \mathbf{0} = \mathbf{0}
```

其中最后一步使用了似然比技巧（likelihood ratio trick）的核心结论 $`\mathbb{E}[\nabla_\theta \log \pi_\theta] = \mathbf{0}`$（见 PPO 文档第 1 章）。因此，$`\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (G - b(s))] = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G]`$，减去基线不影响梯度期望的无偏性。

**事实 (b)：最优基线的形式。** 使梯度估计方差最小的标量基线为：

```math
b^*(s) = \frac{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 \cdot G\right]}{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\right]}
```

这是被梯度范数平方加权的回报期望，而非简单的 $`\mathbb{E}[G]`$。实践中，由于 $`b^*(s)`$ 难以精确计算，通常用状态价值函数 $`V^\pi(s) = \mathbb{E}[G \mid s]`$ 作为近似替代。PPO 正是通过训练一个 critic 网络来拟合 $`V^\pi(s)`$，进而构造广义优势估计（GAE）。

<a id="s2-1-2"></a>

### 2.1.2 本章要回答的问题

在 LLM 强化学习场景中，状态 $`s`$ 对应 prompt $`q`$（加上已生成的前缀），而一条完整回答 $`o`$ 就是一条完整轨迹。如第 1 章所述，outcome reward 设定下所有 token 共享同一个 reward-to-go，这意味着 $`V^\pi(q)`$——从 prompt $`q`$ 出发的期望回报——恰好就是"该 prompt 下模型生成回答的平均质量"。

这自然引出一个关键问题：**$`V^\pi(q)`$ 这个量能不能不靠训练一个神经网络来得到，而靠采样直接估计？如果能，代价是什么？**

具体而言，我们想知道：

1. 用采样均值代替训练的 critic，梯度估计的偏差和方差分别如何变化？
2. 采样估计引入的自相关（self-inclusion）问题有多严重？能否消除？
3. 这种替代在什么条件下是合理的，在什么条件下会失效？

本章将逐一回答这些问题。

<a id="s2-2"></a>

## 2.2 蒙特卡洛基线：$`V^\pi(q)`$ 的采样估计

<a id="s2-2-1"></a>

### 2.2.1 LLM RL 中的价值函数与采样估计

在 LLM RL 的 MDP 框架中（见 PPO 文档第 1 章及本文档第 1 章），prompt $`q`$ 充当初始状态 $`s_0`$，语言模型的生成过程构成一条完整轨迹 $`o = (y_1, y_2, \ldots, y_T)`$，奖励 $`R(q, o)`$ 通常在序列末尾给出（outcome reward）。因此，状态价值函数的定义简化为：

```math
V^\pi(q) = \mathbb{E}_{o \sim \pi(\cdot|q)}[R(q, o)]
```

这里的关键观察是：因为整条回答 $`o`$ 是从策略 $`\pi(\cdot|q)`$ 一次性采样得到的完整对象，$`V^\pi(q)`$ 就是对"该 prompt 下回答质量的期望"的一个简洁表达。不需要像传统 RL 那样对每个中间状态分别估计价值。

给定当前策略 $`\pi_\theta`$，我们可以从 $`\pi_\theta(\cdot|q)`$ 中独立采样 $`G`$ 个回答 $`\{o_1, o_2, \ldots, o_G\}`$，并构造 $`V^\pi(q)`$ 的蒙特卡洛估计量：

```math
\hat{V}_G(q) = \frac{1}{G} \sum_{i=1}^{G} R(q, o_i)
```

记 $`r_i = R(q, o_i)`$ 为第 $`i`$ 个样本的奖励，则 $`\hat{V}_G(q) = \frac{1}{G}\sum_{i=1}^G r_i`$。

<a id="s2-2-2"></a>

### 2.2.2 无偏性证明

**命题 2.1**（$`\hat{V}_G`$ 的无偏性）。$`\mathbb{E}[\hat{V}_G(q)] = V^\pi(q)`$。

**证明：**

```math
\begin{aligned}
\mathbb{E}[\hat{V}_G(q)] &= \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} r_i\right] \\
&= \frac{1}{G} \sum_{i=1}^{G} \mathbb{E}[r_i] \quad \text{（期望的线性性）} \\
&= \frac{1}{G} \sum_{i=1}^{G} \mathbb{E}_{o_i \sim \pi(\cdot|q)}[R(q, o_i)] \quad \text{（每个 $o_i$ 独立同分布于 $\pi(\cdot|q)$）} \\
&= \frac{1}{G} \sum_{i=1}^{G} V^\pi(q) \\
&= V^\pi(q)
\end{aligned}
```

证毕。$`\square`$

<a id="s2-2-3"></a>

### 2.2.3 方差分析

由于 $`o_1, \ldots, o_G`$ 相互独立，$`\hat{V}_G`$ 的方差为：

```math
\text{Var}[\hat{V}_G(q)] = \text{Var}\left[\frac{1}{G}\sum_{i=1}^G r_i\right] = \frac{1}{G^2}\sum_{i=1}^G \text{Var}[r_i] = \frac{\sigma^2(q)}{G}
```

其中我们定义了核心量：

```math
\sigma^2(q) \triangleq \text{Var}_{o \sim \pi(\cdot|q)}[R(q, o)]
```

$`\sigma^2(q)`$ 是**同一 prompt 下回答质量的内在方差**。它是本章后续所有分析的基石。直觉上：

- 当任务对模型来说很"简单"（几乎所有回答都正确）或很"难"（几乎所有回答都错误）时，$`\sigma^2(q)`$ 很小；
- 当任务恰好在模型能力边界附近（部分回答对、部分错），$`\sigma^2(q)`$ 很大；
- 对于二值奖励（$`r \in \{0, 1\}`$），若模型在该 prompt 上的正确率为 $`p`$，则 $`\sigma^2(q) = p(1-p)`$，在 $`p = 0.5`$ 时取最大值 $`0.25`$。

方差以 $`1/G`$ 速率衰减意味着：要将基线估计的标准差减半，需要将采样数 $`G`$ 增至 4 倍。这是一个经典的蒙特卡洛收敛速率。

<a id="s2-2-4"></a>

### 2.2.4 基线误差向梯度的传导

现在考虑用 $`\hat{V}_G(q)`$ 替代 $`V^\pi(q)`$ 后，梯度估计量的性质。定义优势估计量：

```math
\hat{A}_i = r_i - \hat{V}_G(q) = r_i - \frac{1}{G}\sum_{j=1}^G r_j
```

对应的梯度估计量为：

```math
\hat{g} = \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i
```

虽然 $`\hat{V}_G`$ 本身是 $`V^\pi`$ 的无偏估计，但 **$`\hat{A}_i`$ 作为整体用于梯度估计时是有偏的**。原因在于 $`\hat{V}_G`$ 中包含了 $`r_i`$ 自身——基线与被评估样本之间存在自相关（self-inclusion）。

为了看清这一点，将 $`\hat{A}_i`$ 展开：

```math
\hat{A}_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j = r_i - \frac{r_i}{G} - \frac{1}{G}\sum_{j \neq i} r_j = \frac{G-1}{G} r_i - \frac{1}{G}\sum_{j \neq i} r_j
```

代入梯度估计量并取期望：

```math
\begin{aligned}
\mathbb{E}[\hat{g}] &= \frac{1}{G}\sum_{i=1}^G \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \left(\frac{G-1}{G} r_i - \frac{1}{G}\sum_{j \neq i} r_j\right)\right] \\
&= \frac{1}{G}\sum_{i=1}^G \left(\frac{G-1}{G} \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i] - \frac{1}{G}\sum_{j \neq i} \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q)] \cdot \mathbb{E}[r_j]\right)
\end{aligned}
```

注意第二项中，由于 $`o_i`$ 与 $`o_j`$（$`j \neq i`$）独立，期望可以分解。又因 $`\mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q)] = \mathbf{0}`$，第二项整体为零。因此：

```math
\mathbb{E}[\hat{g}] = \frac{G-1}{G} \cdot \frac{1}{G}\sum_{i=1}^G \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i] = \frac{G-1}{G} \cdot \mathbb{E}[\nabla_\theta \log \pi_\theta(o|q) \cdot R(q,o)]
```

而真实的策略梯度为 $`\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(o|q) \cdot R(q,o)]`$。因此：

```math
\mathbb{E}[\hat{g}] = \frac{G-1}{G} \cdot \nabla_\theta J(\theta)
```

**这表明全组均值基线导致的梯度估计存在一个 $`(G-1)/G`$ 的缩放偏差。** 我们将在 2.4 节详细讨论这个结果的深层含义。目前只需记住：自相关确实引入了偏差，但其形式是一个纯粹的常数缩放因子。

<a id="s2-3"></a>

## 2.3 留一法基线（Leave-One-Out）与 RLOO

<a id="s2-3-1"></a>

### 2.3.1 动机：消除自相关偏差

上一节的分析揭示了问题的根源：$`\hat{V}_G`$ 中包含 $`r_i`$ 自身，导致基线与被评估样本不独立。解决方案很自然——**在评估第 $`i`$ 个样本时，将其从基线计算中剔除**。这就是留一法（Leave-One-Out, LOO）基线的核心思想。

<a id="s2-3-2"></a>

### 2.3.2 留一法基线的定义

定义第 $`i`$ 个样本的留一法基线为：

```math
b_i^{\text{LOO}} = \frac{1}{G-1} \sum_{j \neq i} r_j
```

对应的优势估计量为：

```math
\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}} = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j
```

<a id="s2-3-3"></a>

### 2.3.3 无偏性的严格证明

**命题 2.2**（留一法梯度估计的无偏性）。定义梯度估计量 $`\hat{g}^{\text{LOO}} = \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}`$，则 $`\mathbb{E}[\hat{g}^{\text{LOO}}] = \nabla_\theta J(\theta)`$。

**证明：** 由期望的线性性，只需证明对每个 $`i`$：

```math
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot R(q, o_i)\right]
```

将 $`\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}}`$ 代入左边：

```math
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot (r_i - b_i^{\text{LOO}})\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i\right] - \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot b_i^{\text{LOO}}\right]
```

关键在于第二项。注意 $`b_i^{\text{LOO}} = \frac{1}{G-1}\sum_{j \neq i} r_j`$ **只依赖于 $`\{o_j\}_{j \neq i}`$**，而 $`o_i`$ 与 $`\{o_j\}_{j \neq i}`$ 相互独立（因为它们是从 $`\pi(\cdot|q)`$ 中独立采样的）。因此：

```math
\begin{aligned}
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot b_i^{\text{LOO}}\right] &= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbb{E}_{o_i}\left[\nabla_\theta \log \pi_\theta(o_i|q) \mid \{o_j\}_{j \neq i}\right]\right] \\
&= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbb{E}_{o_i}\left[\nabla_\theta \log \pi_\theta(o_i|q)\right]\right] \quad \text{（独立性）} \\
&= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbf{0}\right] \quad \text{（似然比技巧）} \\
&= \mathbf{0}
\end{aligned}
```

因此：

```math
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i\right]
```

对所有 $`i`$ 求平均即得 $`\mathbb{E}[\hat{g}^{\text{LOO}}] = \nabla_\theta J(\theta)`$。证毕。$`\square`$

**关键洞察：** 上述证明的核心条件是 $`b_i^{\text{LOO}}`$ 与 $`o_i`$ 的**独立性**。这不是充分条件的弱化版本，而是充要条件：任何与 $`o_i`$ 独立的基线都不会引入偏差；任何包含 $`o_i`$ 信息的基线（如全组均值）都会引入偏差。留一法恰好是"使用尽可能多的样本信息同时保持独立性"的最优选择。

<a id="s2-3-4"></a>

### 2.3.4 留一法基线的方差

$`b_i^{\text{LOO}}`$ 基于 $`G-1`$ 个独立样本的均值，因此其方差为：

```math
\text{Var}[b_i^{\text{LOO}}] = \frac{\sigma^2(q)}{G-1}
```

与之对比，全组均值基线的方差为 $`\sigma^2(q)/G`$。留一法基线的方差略大，比值为 $`G/(G-1)`$。当 $`G=8`$ 时，比值为 $`8/7 \approx 1.14`$；当 $`G=64`$ 时，比值为 $`64/63 \approx 1.016`$。这是为无偏性付出的代价——在实践中几乎可以忽略。

更完整地看，留一法梯度估计量的方差涉及 $`\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}}`$ 的方差。由于 $`r_i`$ 与 $`b_i^{\text{LOO}}`$ 独立：

```math
\text{Var}[\hat{A}_i^{\text{LOO}}] = \text{Var}[r_i] + \text{Var}[b_i^{\text{LOO}}] = \sigma^2(q) + \frac{\sigma^2(q)}{G-1} = \sigma^2(q) \cdot \frac{G}{G-1}
```

相比之下，全组均值优势 $`\hat{A}_i = r_i - \hat{V}_G`$ 的方差为（利用 $`r_i`$ 与 $`\hat{V}_G`$ 的协方差 $`\text{Cov}(r_i, \hat{V}_G) = \sigma^2/G`$）：

```math
\text{Var}[\hat{A}_i] = \text{Var}[r_i] + \text{Var}[\hat{V}_G] - 2\text{Cov}(r_i, \hat{V}_G) = \sigma^2 + \frac{\sigma^2}{G} - \frac{2\sigma^2}{G} = \sigma^2 \cdot \frac{G-1}{G}
```

有趣的是，全组均值优势的方差反而**更小**（因子 $`(G-1)/G < G/(G-1)`$），但这正是以偏差为代价换来的。这构成了一个经典的偏差-方差权衡（bias-variance tradeoff）。

<a id="s2-3-5"></a>

### 2.3.5 RLOO：REINFORCE Leave-One-Out

RLOO（REINFORCE Leave-One-Out）由 Ahmadian et al. (2024) 在论文《Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs》(arXiv:2402.14740) 中提出。它的定义极为简洁：

```math
\nabla_\theta J_{\text{RLOO}}(\theta) \approx \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \left(r_i - \frac{1}{G-1}\sum_{j \neq i} r_j\right)
```

RLOO 本质上就是 **REINFORCE + 留一法基线**：没有重要性比率（importance ratio）、没有 clip 操作、没有 KL 散度正则项。它是 GRPO 的"极简近亲"。

**RLOO 与 GRPO 的关系可以用一个公式概括：**

```math
\text{GRPO} \approx \text{RLOO} + \text{PPO 式 clip} + \text{KL 正则项} + \text{除以组内 std}
```

具体而言：
- RLOO 提供无偏的优势估计基础；
- GRPO 在此基础上加入了 PPO 风格的 clip 机制以限制策略更新幅度；
- GRPO 加入了 KL 散度项以防止策略偏离参考模型过远；
- GRPO 额外除以了组内标准差进行归一化（这一点的理论后果将在 2.5 节分析）。

在实践中，RLOO 在 TRL 库中有独立实现，且在多个基准上与 GRPO 效果接近。这说明 PPO 式的 clip 和 KL 项虽然有用，但并非 GRPO 成功的唯一原因——**留一法基线本身就提供了足够好的梯度信号**。

<a id="s2-4"></a>

## 2.4 GRPO 用的是哪一种基线

<a id="s2-4-1"></a>

### 2.4.1 原论文的明确选择

这是一个很多资料混淆的关键点。**DeepSeekMath 原论文（Shao et al., 2024）中定义的 GRPO 使用的是全组均值基线，而非留一法基线。** 原论文的优势定义为：

```math
\hat{A}_i^{\text{GRPO}} = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
```

其中 $`\text{mean}`$ 和 $`\text{std}`$ 均基于全部 $`G`$ 个样本计算。因此，严格来说，原始 GRPO 的优势估计是**有偏的**，偏差量级为 $`O(1/G)`$（如 2.2.4 节所示）。

<a id="s2-4-2"></a>

### 2.4.2 偏差的实践影响

这个偏差在实践中影响有限，原因有二：

1. **$`G`$ 通常足够大。** 实践中 $`G`$ 取值在 8 到 64 之间，$`(G-1)/G`$ 的范围是 $`0.875`$ 到 $`0.984`$，相对于梯度估计本身的随机噪声而言微不足道。
2. **clip 机制本身也引入偏差。** PPO 的 clip 操作是一种有偏的梯度修正，GRPO 继承了这一特性。在这种背景下，基线的 $`O(1/G)`$ 偏差并不构成额外的理论负担。

<a id="s2-4-3"></a>

### 2.4.3 论文与实现的差异

需要注意的是，部分现代实现提供了不同的基线选项。例如，TRL 库的某些配置支持 `scale_rewards` 参数和不同的 baseline 类型选择，有些实现确实采用了留一法变体。读者在阅读代码时应注意核对具体实现与原论文的差异。

<a id="s2-4-4"></a>

### 2.4.4 偏差的定量表达：$`(G-1)/G`$ 缩放定理

现在我们给出 2.2.4 节结论的一个更清晰的表述。

**命题 2.3**（全组均值优势与留一法优势的精确关系）。在不除以 std 的前提下，全组均值基线得到的优势恰好是留一法优势的 $`(G-1)/G`$ 倍：

```math
r_i - \text{mean}(\{r_1, \ldots, r_G\}) = \frac{G-1}{G} \left(r_i - b_i^{\text{LOO}}\right)
```

**证明：** 首先写出全组均值与留一法均值的关系：

```math
\text{mean}(\{r_1, \ldots, r_G\}) = \frac{1}{G}\sum_{j=1}^G r_j = \frac{1}{G}\left(r_i + \sum_{j \neq i} r_j\right) = \frac{r_i}{G} + \frac{G-1}{G} \cdot \frac{1}{G-1}\sum_{j \neq i} r_j = \frac{r_i}{G} + \frac{G-1}{G} \cdot b_i^{\text{LOO}}
```

因此：

```math
\begin{aligned}
r_i - \text{mean} &= r_i - \frac{r_i}{G} - \frac{G-1}{G} \cdot b_i^{\text{LOO}} \\
&= \frac{G-1}{G} \cdot r_i - \frac{G-1}{G} \cdot b_i^{\text{LOO}} \\
&= \frac{G-1}{G} \left(r_i - b_i^{\text{LOO}}\right)
\end{aligned}
```

证毕。$`\square`$

**这个结论的含义非常深刻：** 全组均值基线相对于留一法基线的偏差，表现为一个纯粹的常数缩放因子 $`(G-1)/G`$。在梯度层面，这等价于将学习率乘以 $`(G-1)/G`$，而**完全不改变梯度的方向**。换句话说，「GRPO 的基线偏差」实际上只是一个微小的等效学习率偏移，而非方向性错误。

**重要限制条件：** 上述等价性仅在以下条件下严格成立：
1. **不除以 std**：一旦引入 $`\text{std}(\{r_1, \ldots, r_G\})`$ 作为分母，由于 std 本身也依赖于 $`r_i`$，变换变为非线性，常数缩放的论证不再适用。
2. **不做 clip**：clip 操作是非线性的，$`(G-1)/G`$ 的缩放可能在 clip 的边界处产生不同的截断行为。

因此，完整的 GRPO（含 std 归一化和 clip）的偏差分析比命题 2.3 更复杂。但命题 2.3 仍然提供了一个有价值的基准参照：在最简情形下，偏差的本质是良性的。

<a id="s2-5"></a>

## 2.5 除以标准差的含义与代价

<a id="s2-5-1"></a>

### 2.5.1 数学含义：逐 prompt 的自适应尺度归一化

GRPO 的优势定义为：

```math
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
```

从统计学角度看，这是对组内奖励进行**标准化（standardization）**：减去均值再除以标准差，使得不同 prompt 的优势分布近似统一到零均值、单位方差的尺度上。如果组内奖励近似正态分布，标准化后的优势约服从 $`\mathcal{N}(0, 1)`$。

<a id="s2-5-2"></a>

### 2.5.2 好处

这种归一化带来两个实际好处：

1. **跨 prompt 梯度尺度一致。** 不同 prompt 的奖励方差可能差异巨大：一个简单的算术题可能所有回答都得 1 分（$`\sigma \approx 0`$），而一道竞赛题的回答质量参差不齐（$`\sigma`$ 较大）。不归一化时，前者的梯度贡献极小，后者主导训练。归一化后，每个 prompt 的梯度贡献量级相当。
2. **隐式梯度归一化。** 在小 batch 设置下，除以 std 相当于对每个 prompt 的梯度做了自适应缩放，有助于稳定训练动态，减少对全局学习率的敏感性。

<a id="s2-5-3"></a>

### 2.5.3 代价：难度偏差与非线性破坏

除以 std 的代价同样显著：

**第一，破坏了常数缩放论证。** 如 2.4.4 节所述，在不除 std 时，全组均值基线与留一法基线只差一个常数因子。但 std 的计算也包含 $`r_i`$：

```math
\hat{\sigma} = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}
```

这使得 $`\hat{A}_i = (r_i - \bar{r})/\hat{\sigma}`$ 成为 $`r_i`$ 的非线性函数。命题 2.3 的简洁结论不再成立，偏差分析变得更加复杂。

**第二，引入难度偏差（difficulty bias）。** 考虑 $`\hat{A}_i`$ 的量级与 $`\hat{\sigma}`$ 的关系：

```math
|\hat{A}_i| \propto \frac{|r_i - \bar{r}|}{\hat{\sigma}(q)}
```

当 $`\hat{\sigma}(q)`$ 很小时（即该 prompt 对模型来说太简单或太难，组内奖励几乎全同），$`\hat{A}_i`$ 会被放大；当 $`\hat{\sigma}(q)`$ 很大时（模型在该 prompt 上表现不稳定），$`\hat{A}_i`$ 会被压缩。这意味着：

- **"极端难度"的 prompt 获得过大权重**：这些 prompt 本应提供的训练信号最弱（因为模型要么已经掌握、要么完全不会），却被赋予了最大的梯度贡献。
- **"适中难度"的 prompt 被压制**：这些 prompt 恰恰是最有训练价值的（模型处于学习前沿），却被归一化缩小了贡献。

这个偏差的完整分析与修正方案将在第 5 章（Dr.GRPO）中详细讨论。此处只需理解其数学机理。

**第三，数值不稳定性。** 当 $`\hat{\sigma}(q) \to 0`$ 时，$`\hat{A}_i`$ 趋向无穷大。实践中必须加入保护项：

```math
\hat{A}_i = \frac{r_i - \bar{r}}{\hat{\sigma} + \epsilon}
```

其中 $`\epsilon`$ 通常取 $`10^{-8}`$ 量级。但当 $`\hat{\sigma}`$ 本身就在 $`\epsilon`$ 量级时，这个保护并不能真正解决问题——它只是把爆炸的上限从无穷大变成了一个很大的有限值。

<a id="s2-5-4"></a>

### 2.5.4 全同奖励下的零梯度现象

一个特别重要的特殊情况是：**当组内所有奖励完全相同时**（例如二值奖励下全部答对或全部答错），$`\hat{\sigma} = 0`$ 且 $`r_i - \bar{r} = 0`$，此时所有 $`\hat{A}_i = 0`$（即使不加 $`\epsilon`$ 保护，$`0/0`$ 在实践中也被处理为 0）。

**这意味着该 prompt 对梯度的贡献完全为零。**

这个现象的训练动力学后果是深远的：随着模型逐渐变强，越来越多的 prompt 变成"全对"状态，这些 prompt 不再提供任何训练信号。梯度信号的有效来源逐渐枯竭，训练效率下降。这正是第 6 章 DAPO 引入"动态采样"（dynamic sampling）要解决的核心问题——通过过滤掉全同奖励的 prompt 并补充新的有区分度的样本，维持训练信号的持续供给。此处我们只需记住这个数学事实及其直觉含义。

<a id="s2-6"></a>

## 2.6 与 Critic 基线的系统性对比

本节是本章的核心结论章节。我们从多个维度系统对比「组内采样基线」与「训练的 critic 基线」。

| 维度 | 训练的 Critic 基线 | 组内采样基线（GRPO/RLOO） |
|------|-------------------|------------------------|
| **偏差** | 因函数近似而有偏；偏差大小取决于 critic 网络的容量、训练数据和收敛程度，难以量化控制 | LOO 形式下严格无偏；全组均值形式下有 $`O(1/G)`$ 的缩放偏差（命题 2.3），可控且良性 |
| **方差** | 来自 critic 自身的估计噪声，随训练过程动态变化，难以预测 | $`\sigma^2(q)/G`$（LOO 为 $`\sigma^2(q)/(G-1)`$），形式确定，可通过增大 $`G`$ 精确控制 |
| **自适应性** | **状态相关**：能对每个 token 位置给出不同的 $`V(s_t)`$，理论上支持精细的信用分配 | **序列级**：整条回答共享同一个基线值，无法区分序列内部不同位置的贡献 |
| **计算成本** | 需要额外的 critic 模型：训练开销（反向传播）、推理开销（前向传播）、显存占用（模型参数+优化器状态） | 仅需 $`G`$ 倍采样推理，无额外模型、无训练开销；代价是 $`G`$ 倍的生成算力 |
| **泛化能力** | 作为函数逼近器，能对未见过的状态给出价值估计，具有外推能力 | 完全依赖当前 batch 的采样，无任何泛化能力；每个新 batch 都必须重新采样，无法复用历史数据 |
| **收敛性与稳定性** | Actor-critic 耦合系统：critic 训练不稳定会通过优势估计传导至 actor，可能导致训练崩溃 | 无反馈回路：基线估计独立于任何可学习参数，不存在耦合不稳定性；但信号更粗糙 |
| **数据效率** | 可利用 off-policy 数据更新 critic，跨 epoch 复用采样数据（PPO 的多 epoch 训练） | 严格的 on-policy：每次策略更新后必须重新采样，旧数据的基线估计失效 |

<a id="s2-6-1"></a>

### 2.6.1 自适应性的深入讨论

表格中"自适应性"一行是**最本质的区别**，需要展开讨论。

Critic 的核心优势在于它是状态相关的：对于序列中的每个前缀 $`s_t = (q, y_1, \ldots, y_{t-1})`$，critic 能给出不同的价值估计 $`V(s_t)`$。这使得 GAE 能够进行精细的信用分配——区分"哪个 token 做得好、哪个做得差"。

然而，如第 1 章所述，在 outcome reward 设定下，奖励只在序列末尾给出，所有中间步的即时奖励为零。此时 reward-to-go 对所有 token 都是相同的（等于整个序列的奖励 $`R(q, o)`$）。在这种特殊结构下，真实的状态价值函数 $`V^\pi(s_t)`$ 对所有前缀 $`s_t`$ 近似相等（都约等于 $`V^\pi(q)`$），因为无论从哪个前缀出发，最终的期望回报都由完整回答的质量决定。

**这意味着：在 outcome reward 设定下，critic 的状态相关性优势被大幅削弱。** Critic 实际上退化为一个近似常数函数，其提供的信息与序列级基线相差无几。这正是 GRPO 能在 LLM RL 中成功的深层理论原因——它放弃了一个在当前设定下本就价值有限的能力，换来了巨大的计算和工程简化。

**但这个论证是有条件的。** 如果奖励是 process-level 的（即每个 token 或每步都有独立的奖励信号，详见第 4 章），那么不同前缀的 reward-to-go 确实不同，critic 的状态相关性就有了实质价值。在这种情况下，GRPO 的序列级基线假设不再成立，使用 critic 可能是必要的。这个条件依赖关系是理解 GRPO 适用范围的关键。

<a id="s2-6-2"></a>

### 2.6.2 结论性判断

综合以上分析，我们给出以下判断：

> **在 outcome reward + 短到中等长度 + 有可靠验证器（verifier）的 LLM 推理任务上，组内采样基线优于训练的 critic 基线。** 这是因为在此设定下 critic 的状态相关性优势被削弱，而其计算成本和训练不稳定性代价却依然存在。
>
> **在 process reward、奖励计算昂贵、或无法多次采样的场景下，critic 仍是必要的。** 此时序列级基线的假设不成立，或者 $`G`$ 倍采样的成本不可承受。

<a id="s2-7"></a>

## 2.7 GAE 在 GRPO 中的命运

<a id="s2-7-1"></a>

### 2.7.1 GAE 的回顾

在 PPO 文档第 4 章中，广义优势估计（Generalized Advantage Estimation, GAE）定义为：

```math
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
```

其中 TD 残差 $`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$，$`\gamma`$ 是折扣因子，$`\lambda \in [0, 1]`$ 控制偏差-方差权衡。

<a id="s2-7-2"></a>

### 2.7.2 GRPO 设定下的退化

在 GRPO 的典型设定中：

1. **$`\gamma = 1`$**：LLM 生成任务是有限步的，不需要折扣。
2. **奖励只在末尾给出**：$`r_t = 0`$ 对所有 $`t < T`$，$`r_T = R(q, o)`$。
3. **没有训练的 critic**：GRPO 不使用 $`V(s_t)`$。

在这些条件下考察 TD 残差。对于中间步 $`t < T`$：

```math
\delta_t = 0 + V(s_{t+1}) - V(s_t)
```

如果 $`V`$ 近似为常数（如 2.6.1 节论证的，在 outcome reward 下这是合理的），则 $`\delta_t \approx 0`$。所有信息集中在最后一步：

```math
\delta_T = R(q, o) - V(s_T)
```

因此 GAE 退化为：$`\hat{A}_t^{\text{GAE}} \approx (\gamma\lambda)^{T-t} \cdot \delta_T`$。当 $`\gamma = 1`$ 且 $`\lambda = 1`$ 时，所有时间步的优势都等于 $`\delta_T = R(q, o) - V(s_T)`$，即一个与 $`t`$ 无关的常数。

**这正是 GRPO 的做法：直接定义序列级常量优势 $`\hat{A}_{i,t} = \hat{A}_i`$ 对所有 $`t`$。** GRPO 丢弃 GAE 不是一个随意的简化，而是在 outcome reward 设定下 GAE 自然退化的结果。

<a id="s2-7-3"></a>

### 2.7.3 代价：信用分配的丧失

这个简化的理论代价是放弃了**信用分配（credit assignment）** 能力。GRPO 无法区分一条回答中"哪个 token 做得好、哪个做得差"，只能对整条回答统一奖惩。

在短回答（几十到几百 token）的场景下，这个问题不太严重。但在长链推理（long Chain-of-Thought, long-CoT）中，一条回答可能包含上万 token，其中只有少数关键步骤决定了最终答案的正确性。GRPO 对所有 token 赋予相同的优势信号，相当于把关键步骤的信号稀释到了大量无关 token 上，降低了学习效率。

后续的研究方向——包括过程奖励模型（Process Reward Model, PRM）、token 级信用分配方法——正是在弥补这个理论缺口。这些内容将在第 4 章和第 7 章中进一步讨论。

<a id="s2-8"></a>

## 2.8 采样数 $`G`$ 的选取：数学分析

<a id="s2-8-1"></a>

### 2.8.1 边际收益递减

从 $`\text{Var}[\hat{V}_G] = \sigma^2/G`$ 出发，$`G`$ 的边际收益遵循经典的 $`1/G`$ 规律：

- $`G`$ 从 4 增到 8：方差减半，标准差降至原来的 $`1/\sqrt{2} \approx 0.71`$；
- $`G`$ 从 8 增到 16：方差再减半，但绝对改善量小于上一档；
- $`G`$ 从 64 增到 128：方差同样减半，但计算成本翻倍，性价比显著降低。

这意味着 $`G`$ 的选择存在一个实用饱和点，超过该点后增大 $`G`$ 的收益不足以抵消成本。

<a id="s2-8-2"></a>

### 2.8.2 小 $`G`$ 的统计风险

当 $`G`$ 过小时，std 的估计极不可靠。对于正态分布样本，样本标准差 $`\hat{\sigma}`$ 的相对标准误（coefficient of variation）约为：

```math
\frac{\text{SE}[\hat{\sigma}]}{\sigma} \approx \frac{1}{\sqrt{2(G-1)}}
```

当 $`G = 2`$ 时，相对标准误为 $`1/\sqrt{2} \approx 0.707`$，即 std 估计量的波动幅度与其真值同量级。当 $`G = 4`$ 时，相对标准误为 $`1/\sqrt{6} \approx 0.408`$，仍然很大。当 $`G = 8`$ 时，相对标准误为 $`1/\sqrt{14} \approx 0.267`$，开始进入可用范围。

这解释了为什么实践中 $`G`$ 很少低于 4：不仅是因为基线估计不准，更是因为 **std 归一化在小 $`G`$ 下极度不稳定**。

<a id="s2-8-3"></a>

### 2.8.3 实用区间

论文与实践中的常用范围：

- DeepSeekMath 使用 $`G = 8`$ 量级的采样；
- 多数后续工作使用 $`G \in [8, 64]`$；
- $`G > 64`$ 的情况较少见，通常只在奖励方差极大或需要极高精度时使用。

<a id="s2-8-4"></a>

### 2.8.4 $`G`$ 与 Prompt 多样性的权衡

在固定算力预算下，总采样数 $`N = N_q \times G`$（$`N_q`$ 为 prompt 数量）是固定的。这构成了一个权衡：

- **大 $`G`$、少 prompt**：每个 prompt 的优势估计更准确，但 prompt 多样性下降，batch 内的覆盖面窄，可能导致策略更新的泛化性不足。
- **小 $`G`$、多 prompt**：prompt 多样性高，覆盖面广，但每个 prompt 的优势估计噪声大，单步更新的方向可能不准。

实践中通常采取折中策略。此外，第 6 章介绍的 DAPO 动态采样技术改变了这个权衡：它允许在不同 prompt 上使用不同的有效 $`G`$（过滤掉全同奖励的 prompt 后补充新样本），从而在一定程度上解耦了 $`G`$ 与 prompt 多样性的矛盾。

<a id="s2-9"></a>

## 2.9 本章小结与自测

<a id="h73"></a>

### 本章要点

1. **蒙特卡洛基线的无偏性与方差**：$`\hat{V}_G(q) = \frac{1}{G}\sum r_i`$ 是 $`V^\pi(q)`$ 的无偏估计，方差为 $`\sigma^2(q)/G`$，其中 $`\sigma^2(q)`$ 是同一 prompt 下回答质量的内在方差。
2. **全组均值基线的自相关偏差**：用包含 $`r_i`$ 的全组均值作基线会引入 $`(G-1)/G`$ 的缩放偏差（命题 2.3），本质上是等效学习率偏移而非方向性错误，但该结论仅在不含 std 归一化和 clip 时严格成立。
3. **留一法基线的无偏性**：$`b_i^{\text{LOO}}`$ 与 $`o_i`$ 独立，保证了梯度估计的严格无偏性（命题 2.2），代价是方差略增为 $`\sigma^2/(G-1)`$。
4. **除以 std 的双刃剑效应**：实现了跨 prompt 的梯度尺度统一，但引入了难度偏差和非线性，破坏了常数缩放论证；全同奖励下梯度信号完全消失。
5. **组内基线 vs critic 的条件优越性**：在 outcome reward 设定下，critic 的状态相关性退化为近似常数，组内基线因此成为合理替代；在 process reward 设定下此论证不成立。
6. **GAE 的退化与信用分配的丧失**：在 outcome reward + $`\gamma=1`$ 下 GAE 退化为序列级常量，GRPO 据此丢弃 GAE，代价是无法进行 token 级信用分配。

<a id="h74"></a>

### 自测题

**题目 1.** 设 $`G=8`$，某 prompt 下模型回答的正确率 $`p=0.5`$（二值奖励），计算蒙特卡洛基线 $`\hat{V}_G`$ 的标准差。若要将该标准差降至 0.05 以下，至少需要多大的 $`G`$？

参考答案：$`\sigma^2 = p(1-p) = 0.25`$，$`\text{SD}[\hat{V}_8] = \sqrt{0.25/8} \approx 0.177`$。要求 $`\sqrt{0.25/G} < 0.05`$，即 $`G > 0.25/0.0025 = 100`$，故至少需要 $`G = 101`$（取整数上界）。

**题目 2.** 证明命题 2.3 中的关系式 $`r_i - \bar{r} = \frac{G-1}{G}(r_i - b_i^{\text{LOO}})`$，并解释为什么这个关系在引入 std 归一化后不再成立。

参考答案：证明见 2.4.4 节。引入 std 后，$`\hat{A}_i = (r_i - \bar{r})/\hat{\sigma}`$，而 $`\hat{\sigma}`$ 本身依赖于 $`r_i`$（通过 $`(r_i - \bar{r})^2`$ 项），使得 $`\hat{A}_i`$ 成为 $`r_i`$ 的非线性函数。常数缩放 $`(G-1)/G`$ 只对分子成立，对整个比值不成立，因为分母 $`\hat{\sigma}`$ 在全组均值和留一法两种情形下的值也不同。

**题目 3.** 在一个二值奖励的任务中，若某 prompt 的 8 个采样回答全部正确（$`r_1 = \cdots = r_8 = 1`$），GRPO 的优势 $`\hat{A}_i`$ 是多少？这对训练意味着什么？

参考答案：$`\bar{r} = 1`$，$`\hat{\sigma} = 0`$，$`r_i - \bar{r} = 0`$，故 $`\hat{A}_i = 0`$（实践中 $`0/\epsilon = 0`$）。该 prompt 对梯度贡献为零，不提供任何训练信号。随着模型变强，越来越多 prompt 进入此状态，有效训练信号枯竭。

**题目 4.** 比较全组均值基线和留一法基线在 $`G=8`$ 时的方差比，并说明为什么实践中这个差异通常可以忽略。

参考答案：$`\text{Var}[b^{\text{LOO}}]/\text{Var}[\bar{r}] = G/(G-1) = 8/7 \approx 1.14`$，即留一法方差仅大 14%。当 $`G`$ 更大时比值更接近 1。相对于梯度估计本身的高方差（来自策略采样的随机性），14% 的基线方差增量几乎不可察觉。

**题目 5.** 在什么条件下 GRPO 的序列级基线假设会失效？此时应该用什么替代方案？

参考答案：当奖励是 process-level（每步/token 都有独立奖励信号）时，不同前缀的 reward-to-go 不同，序列级基线无法捕捉这种差异。此时应使用训练的 critic 提供状态相关的价值估计，或采用过程奖励模型配合 token 级信用分配方法。

**题目 6.** 解释为什么 GRPO 可以在 LLM RL 中成功丢弃 GAE，而在传统 RL（如机器人控制）中不行。关键的结构差异是什么？

参考答案：LLM RL 通常使用 outcome reward（奖励仅在序列末尾给出），且 $`\gamma=1`$，此时所有 token 的 reward-to-go 相同，GAE 退化为序列级常量。传统 RL 中奖励分布在每一步（dense reward），不同状态的 value 差异显著，GAE 的信用分配功能不可或缺。关键结构差异是奖励的时间分布模式：sparse outcome reward vs. dense step reward。

<a id="chapter-3"></a>

# 第 3 章 GRPO 算法本体

<a id="s3-1"></a>

## 3.1 符号与设定

本章是全文档的核心，完整拆解 GRPO（Group Relative Policy Optimization，组相对策略优化）的目标函数、KL 估计器及其与 PPO 的结构差异。读者应已读完本文档第 1、2 章及 PPO 文档，因此以下符号直接沿用前文约定，仅做必要汇总。

- **Prompt 分布**：$`q \sim P(Q)`$，其中 $`Q`$ 为 prompt 空间。
- **回答（completion）**：对每个 prompt $`q`$，从旧策略 $`\pi_{\theta_{old}}`$ 独立采样 $`G`$ 个回答 $`\{o_1, o_2, \dots, o_G\}`$。第 $`i`$ 个回答记为序列 $`o_i = (o_{i,1}, o_{i,2}, \dots, o_{i,|o_i|})`$，其中 $`|o_i|`$ 表示该回答的 token 数。**注意不同回答的长度一般不同**，即 $`|o_i| \neq |o_j|`$ 当 $`i \neq j`$。
- **旧策略（old policy）**：$`\pi_{\theta_{old}}`$ 是生成本批 rollout 时所使用的策略参数。在 on-policy 设定下 $`\theta_{old} = \theta`$；在多 epoch 或异步更新中 $`\theta_{old}`$ 可能滞后于当前 $`\theta`$。
- **奖励**：$`r_i = R(q, o_i) \in \mathbb{R}`$，为标量，整条回答只有一个分数。这是 outcome reward 的标准设定（见第 1 章）。
- **参考模型（reference model）**：$`\pi_{ref}`$，通常是 SFT 后的初始模型，在整个 RL 训练过程中参数固定不变，用于 KL 约束以防止策略偏离过远。
- **词表**：$`\mathcal{V}`$，大小为 $`V`$。策略 $`\pi_\theta(\cdot | q, o_{<t})`$ 输出词表上的概率分布。

以上符号贯穿本章所有公式。特别强调：$`|o_i|`$ 是变量而非定值，这一事实直接影响目标函数中的归一化方式，也是后续长度偏差问题的根源（详见第 5 章）。

<a id="s3-2"></a>

## 3.2 GRPO 完整目标函数

<a id="s3-2-1"></a>

### 3.2.1 DeepSeekMath 原论文形式

DeepSeekMath（Shao et al., 2024, arXiv:2402.03300）给出的 GRPO 目标函数（原文 Eq. 4）为：

```math
J(\theta) = \mathbb{E}_{q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( r_{i,t}(\theta)\,\hat{A}_{i,t} - \beta\, \mathbb{D}_{KL}\!\left[\pi_\theta(\cdot|q,o_{i,<t}) \,\|\, \pi_{ref}(\cdot|q,o_{i,<t})\right] \right) \right]
```

其中三个核心组件定义如下：

**逐 token 重要性比率（per-token importance ratio）**：

```math
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}
```

**组内归一化优势（group-normalized advantage）**：

```math
\hat{A}_{i,t} = \hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}
```

**KL 散度**：$`\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$ 是在每个 token 位置 $`(i,t)`$ 上、对当前策略与参考模型之间的 KL 散度。

下面逐项拆解这个目标函数的六个组成部分。

<a id="s3-2-2"></a>

### 3.2.2 逐项拆解

**① 外层平均 $`\frac{1}{G}\sum_{i=1}^G`$**：对组内 $`G`$ 个回答取算术平均。这对应蒙特卡洛估计中对采样样本的期望近似。注意这里是对"回答"取平均，不是对 token 取平均——每个回答作为一个整体贡献一份梯度信号。

**② 内层平均 $`\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}`$**：对第 $`i`$ 个回答内的所有 token 位置取算术平均。**这个归一化需要特别注意**：它是对单条回答内部的 token 取平均，而不是对整个 batch 的所有 token 取平均。这意味着一条长度为 100 token 的回答和一条长度为 10 token 的回答，在目标函数中贡献的权重相同（都是 $`1/G`$），但前者将梯度信号分摊到了 100 个 token 上，每个 token 获得的梯度幅度仅为后者的 $`1/10`$。**这正是长度偏差（length bias）的数学根源**：短回答的每个 token 获得更强的梯度更新。完整的长度偏差分析与 Dr.GRPO 修正将在第 5 章展开，此处仅需记住这一结构特征。

**③ 替代目标项 $`r_{i,t}(\theta)\,\hat{A}_{i,t}`$**：这是标准的 importance-weighted surrogate objective，承接 PPO 文档第 5、6 章的推导。比率 $`r_{i,t}`$ 修正了 off-policy 数据的分布偏移，$`\hat{A}_{i,t}`$ 提供了梯度方向。当 $`r_{i,t} > 1`$ 且 $`\hat{A}_i > 0`$ 时，乘积为正，梯度增大该 token 的概率；反之则减小。

**④ 优势对所有 $`t`$ 取同一值 $`\hat{A}_{i,t} = \hat{A}_i`$**：这是 GRPO 与经典 PPO 最显著的结构差异之一。如第 1、2 章所述，在 outcome reward 设定下，奖励仅在序列末尾给出，所有 token 共享同一个 reward-to-go，因此 GAE 退化为常数（见 PPO 文档第 6 章及本文档第 2 章），GRPO 索性丢弃 GAE，直接用组内归一化的标量奖励作为每个 token 的优势。**代价是完全丧失了 token 级别的信用分配（credit assignment）**：一条错误回答中写得好的 token 和写得差的 token 收到完全相同的负优势信号。

**⑤ KL 惩罚项 $`-\beta\,\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$**：**GRPO 将 KL 项直接写在损失函数内部**，这与 RLHF 版 PPO 的典型做法形成鲜明对比。RLHF-PPO 通常将 KL 塑形进奖励：$`\tilde{r}_t = r_t - \beta \cdot \text{KL}_t`$，然后 KL 信号要经过基线减除、GAE 折扣累积、优势归一化等多步处理才最终影响梯度。而 GRPO 的损失函数 KL 项对每个 token 施加一个**直接的、可微的约束**，其梯度立即作用于该 token 的 logit，不经过任何中间变换。

两者的梯度路径差异可以用简化的形式说明。设某个 token 位置的 KL 值为 $`D_t = \mathbb{D}_{KL}[\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)]`$：

- **损失函数 KL**（GRPO 原论文）：$`\nabla_\theta (-\beta D_t) = -\beta \nabla_\theta D_t`$，梯度直接反传。
- **奖励塑形 KL**（RLHF-PPO）：KL 先进入 $`\tilde{r}_t = r_t - \beta D_t`$，再经 $`\hat{A}_t = \text{GAE}(\{\tilde{r}\})`$ 变成优势，最后进入 $`\nabla_\theta [r_{i,t} \hat{A}_t]`$。此时 $`\beta`$ 的有效量级被优势的尺度（取决于奖励方差和基线）所调制，实际约束强度难以精确控制。

**实践警告**：尽管 DeepSeekMath 原论文将 KL 写在损失函数里，但社区实现中存在两种写法。TRL 库的某些配置会将 KL 放回奖励中进行塑形。读者在阅读或调试代码时，**必须核对 KL 项的实际位置**，因为两种写法的 $`\beta`$ 含义和有效量级完全不同。

**⑥ 关于 clip 的论文-实现差异**：**DeepSeekMath 论文的 Eq. 4 中没有 min/clip 操作**。原论文的目标函数是纯替代目标加 KL 惩罚，不含 PPO 式的截断机制。然而，实际流传最广的实现（包括 TRL、verl 等主流框架）以及后续论文（DAPO、Dr.GRPO 等）普遍在 GRPO 上叠加了 PPO 的 clip 机制。这是一个重要的"论文与实现的差异"，必须诚实说明。

带 clip 的实用形式为：

```math
L^{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\!\left(r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}\!\left(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\right)\hat{A}_i\right) - \beta\,\hat{D}_{KL,i,t}\right]
```

其中 $`\text{clip}(x, 1-\varepsilon, 1+\varepsilon) = \max(1-\varepsilon, \min(1+\varepsilon, x))`$，$`\hat{D}_{KL,i,t}`$ 是 KL 散度的采样估计器（下一节详述）。取负号是因为优化框架通常最小化损失，而 $`J(\theta)`$ 是最大化目标。clip 的作用机制与 PPO 完全一致（见 PPO 文档第 6 章的六情形梯度分析），此处不再重复推导。

<a id="s3-3"></a>

## 3.3 KL 项的无偏估计器

<a id="s3-3-1"></a>

### 3.3.1 问题陈述

目标函数中的 $`\mathbb{D}_{KL}[\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)]`$（简写 $`x = (q, o_{i,<t})`$）定义为：

```math
\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}] = \sum_{y \in \mathcal{V}} \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
```

这是对词表 $`\mathcal{V}`$ 求和的量。虽然原则上可以精确计算（遍历整个词表），但在每个 token 位置都做一次全词表求和开销不小（$`V`$ 通常为几万到几十万）。更重要的是，我们需要一个**能通过采样得到的、可微的、能正确反传梯度的估计器**。

<a id="s3-3-2"></a>

### 3.3.2 Schulman k3 估计器及其无偏性证明

DeepSeekMath 采用的是 Schulman 在其博客 "Approximating KL Divergence" 中提出的 k3 估计器。设 $`y \sim \pi_\theta(\cdot|x)`$ 为从当前策略采样的 token（实际上就是 rollout 中生成的 $`o_{i,t}`$），定义比率：

```math
r = \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}
```

则 k3 估计器为：

```math
\hat{D}_{KL}^{(k3)} = r - \log r - 1 = \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - \log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - 1
```

**无偏性证明**：

```math
\begin{aligned}
\mathbb{E}_{y \sim \pi_\theta}\!\left[\hat{D}_{KL}^{(k3)}\right] &= \mathbb{E}_{y \sim \pi_\theta}\!\left[\frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - \log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - 1\right] \\
&= \underbrace{\mathbb{E}_{y \sim \pi_\theta}\!\left[\frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}\right]}_{(*)} - \underbrace{\mathbb{E}_{y \sim \pi_\theta}\!\left[\log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}\right]}_{(**)} - 1
\end{aligned}
```

对第一项 $`(*)`$，利用重要性采样的基本恒等式：

```math
\mathbb{E}_{y \sim \pi_\theta}\!\left[\frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}\right] = \sum_{y \in \mathcal{V}} \pi_\theta(y|x) \cdot \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} = \sum_{y \in \mathcal{V}} \pi_{ref}(y|x) = 1
```

对第二项 $`(**)`$，展开对数比：

```math
\mathbb{E}_{y \sim \pi_\theta}\!\left[\log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}\right] = \mathbb{E}_{y \sim \pi_\theta}[\log \pi_{ref}(y|x) - \log \pi_\theta(y|x)] = -\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]
```

这里用到了 KL 散度的定义 $`\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}] = \mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta - \log \pi_{ref}]`$。

代回原式：

```math
\mathbb{E}_{y \sim \pi_\theta}\!\left[\hat{D}_{KL}^{(k3)}\right] = 1 - \left(-\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]\right) - 1 = \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]
```

证毕。$`\blacksquare`$

<a id="s3-3-3"></a>

### 3.3.3 非负性证明

令 $`u = \log r`$，则 $`r = e^u`$，于是：

```math
\hat{D}_{KL}^{(k3)} = e^u - u - 1
```

由指数函数的凸性，$`e^u \geq u + 1`$ 对所有实数 $`u`$ 成立（这是 $`e^u`$ 在 $`u=0`$ 处的切线不等式，当且仅当 $`u=0`$ 即 $`r=1`$ 即 $`\pi_\theta = \pi_{ref}`$ 时取等号）。因此：

```math
e^u - u - 1 \geq 0 \quad \forall\, u \in \mathbb{R}
```

即 $`\hat{D}_{KL}^{(k3)} \geq 0`$ 恒成立。$`\blacksquare`$

**这个性质至关重要**：它保证了采样估计出的 KL 值永远不会是负数。如果 KL 估计器可以为负，那么作为惩罚项使用时，负的 KL 惩罚会变成"奖励"，鼓励策略远离参考模型，这显然是荒谬的。k3 的非负性从根本上杜绝了这种病态行为。

<a id="s3-3-4"></a>

### 3.3.4 与其他估计器的对比

Schulman 博客中还讨论了另外两种估计器，一并列出以供比较：

| 估计器 | 公式 | 无偏性 | 非负性 | 方差 |
|--------|------|--------|--------|------|
| k1（朴素） | $`\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}`$ | ✅ 无偏 | ❌ 可为负 | 大 |
| k2（二次） | $`\frac{1}{2}\left(\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right)^2`$ | ❌ 有偏 | ✅ 非负 | 小 |
| k3 | $`\frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - \log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - 1`$ | ✅ 无偏 | ✅ 非负 | 小 |

- **k1** 就是 KL 定义中的被期望量本身，显然无偏，但它是单个样本的对数比，方差大且可正可负。
- **k2** 是 KL 的二阶近似（当 $`\pi_\theta \approx \pi_{ref}`$ 时 $`\text{KL} \approx \frac{1}{2}(\log \text{ratio})^2`$），非负且方差小，但它是有偏估计器，在策略偏离参考模型较远时偏差显著。
- **k3** 兼具无偏与非负，且方差在实践中较小，这是它被 DeepSeekMath 及后续工作广泛采用的根本原因。

<a id="s3-3-5"></a>

### 3.3.5 k3 估计器的梯度分析

k3 估计器中 $`r = \pi_{ref}(y|x)/\pi_\theta(y|x)`$ 显式依赖于 $`\theta`$（因为 $`y`$ 是从 $`\pi_\theta`$ 采样的固定 token，$`\pi_{ref}(y|x)`$ 是常数，但 $`\pi_\theta(y|x)`$ 随 $`\theta`$ 变化）。对 $`\theta`$ 求梯度：

```math
\nabla_\theta \hat{D}_{KL}^{(k3)} = \nabla_\theta (r - \log r - 1) = \left(1 - \frac{1}{r}\right) \nabla_\theta r
```

由于 $`r = \pi_{ref}/\pi_\theta`$，故 $`\nabla_\theta r = -\frac{\pi_{ref}}{\pi_\theta^2} \nabla_\theta \pi_\theta = -r \cdot \nabla_\theta \log \pi_\theta`$。代入得：

```math
\nabla_\theta \hat{D}_{KL}^{(k3)} = \left(1 - \frac{1}{r}\right)(-r) \nabla_\theta \log \pi_\theta(y|x) = (1 - r) \nabla_\theta \log \pi_\theta(y|x)
```

即：

```math
\nabla_\theta \hat{D}_{KL}^{(k3)} = \left(1 - \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)}\right) \nabla_\theta \log \pi_\theta(y|x)
```

这个梯度形式的直觉是：当 $`\pi_\theta(y|x) > \pi_{ref}(y|x)`$（即 $`r < 1`$）时，括号项为正，梯度方向与 $`\nabla_\theta \log \pi_\theta`$ 一致，但由于 KL 项前面有负号 $`-\beta`$，实际效果是**减小**该 token 的概率（把策略拉回参考模型）；反之亦然。

**实践警告**：许多实现中，KL 估计器的梯度会被部分 detach。例如，有些代码只对 $`-\log r`$ 部分保留梯度，而将 $`r`$ 视为常量（stop-gradient），或者反过来。不同的 detach 策略会改变 KL 项的有效梯度方向和大小。读者在使用现有框架时，务必检查 KL 项的梯度实现是否与理论一致。

<a id="s3-4"></a>

## 3.4 GRPO 与 PPO 的逐项对照

下表将 PPO 的完整损失 $`L^{PPO} = L^{CLIP} - c_1 L^{VF} + c_2 S`$ 与 GRPO 损失进行逐项对照：

| 组件 | PPO | GRPO | 数学后果 |
|------|-----|------|----------|
| Clip 替代目标 $`L^{CLIP}`$ | $`\min(r_t \hat{A}_t^{GAE}, \text{clip}(r_t)\hat{A}_t^{GAE})`$ | $`\min(r_{i,t} \hat{A}_i, \text{clip}(r_{i,t})\hat{A}_i)`$（实用版本） | GRPO 的 $`\hat{A}_i`$ 来自组内统计而非 GAE+critic；序列内所有 token 共享同一优势值 |
| Critic 损失 $`c_1 L^{VF}`$ | 训练价值网络拟合 $`V^\pi`$ | **完全去掉** | 省掉一个模型的参数、显存和训练开销；代价是无法做 token 级价值估计 |
| 熵正则 $`c_2 S`$ | 通常保留，鼓励探索 | 原论文不含；实践中存在分歧 | 加熵可防熵崩塌但可能鼓励冗长输出；DAPO 明确调整了此项（见第 6 章） |
| KL 约束 | 常塑形进奖励：$`\tilde{r} = r - \beta \cdot KL`$ | 写在损失函数内（原论文）；也可放奖励中（部分实现） | 损失内 KL 提供直接、未经调制的梯度约束；$`\beta`$ 量级与 RLHF-PPO 不同 |
| 优势来源 | GAE：$`\hat{A}_t^{GAE} = \sum_l (\gamma\lambda)^l \delta_{t+l}`$ | 组内归一化：$`\hat{A}_i = (r_i - \bar{r})/\sigma_r`$ | GRPO 无时序信用分配；优势估计方差为 $`\sigma^2/G`$（见第 2 章） |

**记忆性总结**：

```math
\text{GRPO} = \text{PPO} - \text{critic} - \text{GAE} + \text{组内基线} + \text{显式 KL 项}
```

每一项的数学后果：去掉 critic 消除了价值估计偏差但丧失了 token 级信号；去掉 GAE 使优势变为序列级常量；组内基线以 $`G`$ 倍采样换取无偏的 $`V^\pi`$ 估计（方差 $`\sigma^2/G`$，见第 2 章）；显式 KL 项提供了比奖励塑形更精确的约束控制。

<a id="s3-5"></a>

## 3.5 完整算法伪代码

以下是 GRPO 训练的数学形式完整流程（工程实现细节留给第 8 章）：

---

**算法：GRPO 训练**

**输入**：prompt 数据集 $`\mathcal{D}`$，奖励函数 $`R`$，参考模型 $`\pi_{ref}`$，超参 $`G, B, K, \varepsilon, \beta, \eta`$

**初始化**：$`\theta \leftarrow \theta_{init}`$（通常为 SFT 模型参数）

**for** 每个训练 step **do**:

1. **采样 prompt 批次**：从 $`\mathcal{D}`$ 中抽取 $`B`$ 个 prompt $`\{q_1, \dots, q_B\}`$。

2. **Rollout 生成**：用**当前策略** $`\pi_\theta`$ 对每个 $`q_b`$ 独立采样 $`G`$ 个回答 $`\{o_{b,1}, \dots, o_{b,G}\}`$。**同时记录**每个 token 的 $`\log \pi_\theta(o_{b,i,t} | q_b, o_{b,i,<t})`$。这一步必须在生成的同一次前向传播中完成，不能事后重新计算——因为浮点精度和非确定性算子可能导致重算结果不一致。

3. **计算奖励**：对每个回答计算标量奖励 $`r_{b,i} = R(q_b, o_{b,i})`$。

4. **计算组内优势**：对每个 prompt $`b`$，分别计算：
   ```math
   \bar{r}_b = \frac{1}{G}\sum_{i=1}^G r_{b,i}, \quad \sigma_b = \text{std}(\{r_{b,i}\}_{i=1}^G)
   ```
   ```math
   \hat{A}_{b,i} = \frac{r_{b,i} - \bar{r}_b}{\sigma_b + \epsilon_{num}}
   ```
   其中 $`\epsilon_{num}`$ 是防止除零的小常数（如 $`10^{-8}`$）。若 $`\sigma_b = 0`$（全组奖励相同），则 $`\hat{A}_{b,i} = 0`$，该 prompt 在本 step 无梯度贡献。

5. **冻结旧策略日志概率**：记 $`\log \pi_{\theta_{old}}(o_{b,i,t}|\cdot)`$ 为步骤 2 中记录的值。此后在内层循环中不再更新此值。

6. **内层优化**：**for** $`k = 1`$ to $`K`$ **do**:
   - 从本批数据中抽取 mini-batch。
   - 用当前 $`\theta`$ 前向计算 $`\log \pi_\theta(o_{b,i,t}|\cdot)`$。
   - 计算比率 $`r_{b,i,t} = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})`$。
   - 计算 clip 替代目标与 KL 估计器 $`\hat{D}_{KL}^{(k3)}`$。
   - 合并损失 $`L^{GRPO}`$，反传梯度，优化器更新 $`\theta`$。

7. **丢弃本批 rollout 数据**，进入下一 step。

---

<a id="s3-5-1"></a>

### 3.5.1 关键超参与典型取值

| 超参 | 符号 | 典型范围 | 备注 |
|------|------|----------|------|
| 组大小 | $`G`$ | 8 ~ 64 | 越大优势估计方差越小，但 rollout 成本线性增长 |
| Prompt 批大小 | $`B`$ | 几十 ~ 几百 | 受显存和 rollout 吞吐限制 |
| 内层 epoch 数 | $`K`$ | 1 ~ 4 | LLM 场景下通常比经典 PPO 小，避免 off-policy 漂移 |
| Clip 范围 | $`\varepsilon`$ | 0.2 | DAPO 解耦为 $`\varepsilon_{low}=0.2`$, $`\varepsilon_{high}=0.28`$（见第 6 章） |
| KL 系数 | $`\beta`$ | 0.001 ~ 0.04 | 量级远小于经典 RLHF；RLVR 中常设 $`\beta=0`$（见第 4 章） |
| 学习率 | $`\eta`$ | $`\sim 10^{-6}`$ | LLM 微调学习率远小于经典 RL 的 $`3 \times 10^{-4}`$ |

**关于学习率的说明**：LLM RL 微调使用 $`10^{-6}`$ 量级学习率，而非经典 RL 常用的 $`3 \times 10^{-4}`$，原因是预训练权重已经处于一个非常精细的损失面上，大步长会破坏已有的语言能力。这与从头训练 actor-critic 的场景本质不同。

<a id="s3-5-2"></a>

### 3.5.2 K=1 时 clip 失效的关键观察

**当 $`K=1`$ 时，GRPO 严格来说是 on-policy 的**。此时 $`\theta_{old} = \theta`$（因为内层只跑一个 epoch，没有之前的更新使 $`\theta`$ 偏离 $`\theta_{old}`$），因此：

```math
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|\cdot)}{\pi_{\theta_{old}}(o_{i,t}|\cdot)} = 1 \quad \forall\, i, t
```

比率恒为 1，$`\text{clip}(1, 1-\varepsilon, 1+\varepsilon) = 1`$，$`\min(1 \cdot \hat{A}_i, 1 \cdot \hat{A}_i) = \hat{A}_i`$。**clip 机制完全不生效**。此时 GRPO 退化为：

```math
L^{GRPO}|_{K=1} = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\hat{A}_i - \beta\,\hat{D}_{KL,i,t}\right]
```

这就是「带组内基线的 REINFORCE + KL 惩罚」。只有当 $`K > 1`$（或因异步/延迟更新导致 $`\theta_{old} \neq \theta`$）时，比率才会偏离 1，clip 才开始发挥限制更新幅度的作用。

**这个观察的实践意义**：很多 RLVR（Reinforcement Learning with Verifiable Rewards）实践中 $`K`$ 确实很小（1 或 2），甚至采用 partial rollout 策略，导致 clip 的实际作用远比想象中有限。读者在调参时应意识到：如果 $`K=1`$，调节 $`\varepsilon`$ 不会产生任何效果。

<a id="s3-6"></a>

## 3.6 计算与显存分析

<a id="s3-6-1"></a>

### 3.6.1 训练中需驻留的模型

GRPO 训练时需要同时持有以下组件：

1. **策略模型 $`\pi_\theta`$**：需要梯度、优化器状态（Adam 的一二阶矩）、以及 fp32 master weights。
2. **参考模型 $`\pi_{ref}`$**：仅需前向推理，无需梯度和优化器状态。可使用 LoRA 时通过合并权重省掉独立副本，或通过 offload 降低显存占用。
3. **奖励模型**（若使用 RM 而非规则奖励）：仅需推理。规则奖励（如数学验证器）不需要额外模型。
4. **推理引擎**：vLLM、SGLang 等高吞吐生成引擎，用于高效完成 $`G`$ 倍 rollout。通常需要独立的 GPU 显存或与训练共享显存（通过权重同步机制）。

<a id="s3-6-2"></a>

### 3.6.2 显存粗略分解

对于参数量为 $`\Phi`$ 的模型，各组件的显存占用估算如下（混合精度训练常见口径）：

| 组件 | 显存占用 | 说明 |
|------|----------|------|
| fp16 权重 | $`2\Phi`$ 字节 | 前向/反向计算用 |
| fp16 梯度 | $`2\Phi`$ 字节 | 反向传播产生 |
| Adam 一二阶矩 | $`8\Phi`$ 字节 | fp32 存储，$`4\Phi \times 2`$ |
| fp32 master weights | $`4\Phi`$ 字节 | 优化器更新用 |
| **训练总计** | **$`\approx 16\Phi`$** | 单个可训练模型 |
| 推理（fp16） | $`2\Phi`$ 字节 | 仅前向，无梯度/优化器 |

**PPO vs GRPO 显存对比**：

- **PPO**：policy（训练 $`16\Phi`$）+ critic（训练 $`16\Phi`$）= $`\approx 32\Phi`$
- **GRPO**：policy（训练 $`16\Phi`$）+ reference（推理 $`2\Phi`$）= $`\approx 18\Phi`$

GRPO 节省了约 $`14\Phi`$ 的显存，这对于大模型场景意义重大。例如对 70B 模型（$`\Phi \approx 70 \times 10^9`$），节省约 $`14 \times 70 \times 10^9 \approx 980`$ GB 显存——这相当于数十张 A100 的容量。

<a id="s3-6-3"></a>

### 3.6.3 时间成本的转移

GRPO 并非免费午餐：它把显存开销转成了**推理时间开销**。$`G`$ 倍采样意味着 rollout 阶段的生成量是单采样的 $`G`$ 倍。在 LLM RL 训练中，rollout 生成往往已经是时间瓶颈（占总训练时间的 70%~90%），因此 $`G`$ 倍采样的成本不容忽视。

缓解手段包括：
- 使用 vLLM/SGLang 等高吞吐推理引擎，利用 continuous batching 和 PagedAttention 提升生成效率。
- 训练与推理的权重同步机制（如 ZeRO-3 下的参数收集、或 LoRA 权重合并），避免每次 rollout 都重新加载完整权重。
- 异步 rollout：在训练更新的同时并行生成下一批数据，掩盖生成延迟。

<a id="s3-6-4"></a>

### 3.6.4 何时 GRPO 更划算

定量判断准则：**当 critic 的显存/训练成本 > $`G`$ 倍采样的额外推理成本时，GRPO 更划算**。

- **小模型（≤7B）+ 短回答**：critic 成本低，rollout 也快，PPO 可能更划算（critic 提供的 token 级信号更有价值）。
- **大模型（≥13B）+ 长 CoT 回答**：critic 的显存和训练成本急剧增长，而长序列的价值估计本身就很困难（见第 1 章），GRPO 明显更划算。
- **RLVR 场景**：规则奖励不需要 RM，进一步降低了 GRPO 的基础设施需求。

<a id="s3-7"></a>

## 3.7 一个极简的数值算例

为了建立对 GRPO 优势计算的直觉，我们用具体数字手算几个场景。

<a id="s3-7-1"></a>

### 3.7.1 二值奖励、半对半错

设 $`G=4`$，某 prompt 的四个回答奖励为 $`r = [1, 0, 1, 0]`$（两个正确、两个错误）。

```math
\bar{r} = \frac{1+0+1+0}{4} = 0.5
```

使用总体标准差：

```math
\sigma = \sqrt{\frac{(1-0.5)^2 + (0-0.5)^2 + (1-0.5)^2 + (0-0.5)^2}{4}} = \sqrt{\frac{0.25 \times 4}{4}} = 0.5
```

归一化优势：

```math
\hat{A} = \left[\frac{1-0.5}{0.5},\; \frac{0-0.5}{0.5},\; \frac{1-0.5}{0.5},\; \frac{0-0.5}{0.5}\right] = [1, -1, 1, -1]
```

**解读**：正确的回答得到 $`+1`$ 的优势（增大其每个 token 的概率），错误的得到 $`-1`$（减小概率）。关键在于：**同一条回答里的所有 token 得到完全相同的信号**。如果某个错误回答的前半部分其实写得很好、只是最后一步推理出错，那些好的 token 仍然收到 $`-1`$ 的信号——这就是信用分配缺失的直观体现（承接第 2 章 2.7 节的讨论）。

<a id="s3-7-2"></a>

### 3.7.2 全对：梯度消失

设 $`r = [1, 1, 1, 1]`$。则 $`\bar{r} = 1`$，$`\sigma = 0`$。

此时 $`\hat{A}_i = (1-1)/(0+\epsilon) \approx 0`$，所有优势为零，**该 prompt 在本 step 无任何梯度贡献**。这对应第 2 章 2.5 节的结论：当组内奖励全同时，组均值基线完美解释了所有样本，残差为零。实践中这意味着"太简单"的 prompt 不会推动训练——只有存在区分度的 prompt 才有学习信号。

<a id="s3-7-3"></a>

### 3.7.3 三对一错：不对称惩罚

设 $`r = [1, 1, 1, 0]`$。则：

```math
\bar{r} = \frac{3}{4} = 0.75, \quad \sigma = \sqrt{\frac{3 \times (1-0.75)^2 + (0-0.75)^2}{4}} = \sqrt{\frac{3 \times 0.0625 + 0.5625}{4}} = \sqrt{\frac{0.75}{4}} = \frac{\sqrt{3}}{4} \approx 0.433
```

归一化优势：

```math
\hat{A} = \left[\frac{0.25}{0.433},\; \frac{0.25}{0.433},\; \frac{0.25}{0.433},\; \frac{-0.75}{0.433}\right] \approx [0.577,\; 0.577,\; 0.577,\; -1.732]
```

**关键观察**：那一个错误回答受到的惩罚（$`-1.732`$）远大于每个正确回答得到的奖励（$`+0.577`$），绝对值之比恰好为 $`3:1`$。这不是巧合，而是组内归一化的数学必然：优势之和必须为零（$`\sum_i \hat{A}_i = 0`$），所以当多数样本优于少数样本时，少数样本必须承受更大的负信号来平衡。

**这解释了 GRPO 训练中的一个经验现象**："避免错误"的信号往往强于"鼓励正确"。在三对一的场景中，错误回答受到的惩罚是正确回答奖励的三倍。这种不对称性使得 GRPO 天然倾向于保守策略——减少犯错比追求卓越获得更强的梯度信号。这在数学推理等"对错分明"的任务中可能是有益的（避免幻觉），但在创意写作等任务中可能导致过度保守。

<a id="s3-8"></a>

## 3.8 本章小结与自测

<a id="h100"></a>

### 要点回顾

1. GRPO 目标函数由三层结构组成：外层组平均、内层序列内 token 平均、以及每个 token 位置的 surrogate + KL 项。内层的 $`1/|o_i|`$ 归一化是长度偏差的根源。
2. 优势 $`\hat{A}_i`$ 对序列内所有 token 取同一值，这是 outcome reward 下 GAE 退化的必然结果，代价是丧失 token 级信用分配。
3. KL 项在原论文中写在损失函数内（而非塑形进奖励），提供更直接的梯度约束；但实践中两种写法均存在，需核对代码。
4. k3 估计器 $`r - \log r - 1`$ 兼具无偏性和非负性，是 GRPO 广泛采用的 KL 估计形式；其梯度为 $`(1-r)\nabla_\theta \log \pi_\theta`$。
5. $`K=1`$ 时比率恒为 1，clip 完全失效，GRPO 退化为带组内基线的 REINFORCE + KL 惩罚。
6. GRPO 相比 PPO 节省约 $`14\Phi`$ 显存（去掉 critic），但将成本转移到 $`G`$ 倍 rollout 生成时间上。

<a id="h101"></a>

### 自测题

**题目 1**：证明 k3 估计器 $`\hat{D}_{KL}^{(k3)} = \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - \log \frac{\pi_{ref}(y|x)}{\pi_\theta(y|x)} - 1`$（其中 $`y \sim \pi_\theta`$）是 $`\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$ 的无偏估计器。写出完整推导过程。

参考答案：$`\mathbb{E}_{y \sim \pi_\theta}[\pi_{ref}/\pi_\theta] = \sum_y \pi_\theta \cdot (\pi_{ref}/\pi_\theta) = \sum_y \pi_{ref} = 1`$；$`\mathbb{E}_{y \sim \pi_\theta}[\log(\pi_{ref}/\pi_\theta)] = -\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$；故期望 $`= 1 - (-\text{KL}) - 1 = \text{KL}`$。

**题目 2**：证明 k3 估计器恒非负，并说明这一性质在用作损失函数惩罚项时的重要性。

参考答案：令 $`u = \log r`$，则 $`\hat{D}_{KL}^{(k3)} = e^u - u - 1`$。由 $`e^u \geq u+1`$（凸函数切线不等式），得 $`e^u - u - 1 \geq 0`$。非负性保证 KL 惩罚项不会产生负值从而变成"奖励"，避免策略被错误地推向远离参考模型的方向。

**题目 3**：当内层 epoch 数 $`K=1`$ 时，GRPO 的 clip 机制是否生效？为什么？此时 GRPO 等价于什么算法？

参考答案：不生效。$`K=1`$ 时 $`\theta = \theta_{old}`$，故 $`r_{i,t} = \pi_\theta/\pi_{\theta_{old}} = 1`$，$`\text{clip}(1, 1-\varepsilon, 1+\varepsilon) = 1`$，min 的两个参数相等，clip 不起任何作用。此时 GRPO 等价于带组内基线的 REINFORCE + KL 惩罚。

**题目 4**：GRPO 目标函数中的 $`\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}`$ 归一化会导致什么问题？为什么？（简要回答即可，详细分析在第 5 章。）

参考答案：导致长度偏差。短回答的每个 token 获得的梯度幅度大于长回答（因为总信号被更少的 token 分摊），使得模型倾向于生成较短的回答。

**题目 5**：比较 KL 项"写在损失函数内"与"塑形进奖励"两种方式在梯度传播路径上的区别。

参考答案：损失内 KL 的梯度 $`-\beta \nabla_\theta D_{KL}`$ 直接反传，不经过任何中间变换。奖励塑形 KL 要先经 $`\tilde{r} = r - \beta \cdot KL`$，再经基线减除和归一化变成优势 $`\hat{A}`$，最后进入 surrogate 目标的梯度。后者中 $`\beta`$ 的有效量级被优势尺度调制，约束强度难以精确控制。

**题目 6**：设 $`G=4`$，某 prompt 的奖励为 $`r=[1,1,1,0]`$。手算归一化优势 $`\hat{A}`$，并解释为什么错误回答的惩罚绝对值大于正确回答的奖励绝对值。

参考答案：$`\bar{r}=0.75`$，$`\sigma=\sqrt{3}/4 \approx 0.433`$，$`\hat{A} \approx [0.577, 0.577, 0.577, -1.732]`$。因为组内归一化要求 $`\sum \hat{A}_i = 0`$，三个正优势之和必须被一个负优势平衡，故负优势的绝对值是正优势的三倍。这是数学必然，也使得"避免错误"的信号强于"鼓励正确"。

<a id="chapter-4"></a>

# 第 4 章 奖励信号与可验证奖励（RLVR）

<a id="s4-1"></a>

## 4.1 为什么奖励设计是 GRPO 的第一性问题

在 PPO 中，critic 网络 $`V_\psi(s)`$ 承担了双重角色：它既是优势函数的基线（baseline），也是奖励信号的平滑器。即使奖励函数本身有噪声、有偏差，critic 通过时序差分（TD）学习到的值函数能在一定程度上"吸收"这些缺陷——因为 TD 目标 $`r_t + \gamma V_\psi(s_{t+1})`$ 将相邻时刻的估计互相锚定，形成了一种内在的正则化。换言之，PPO 对奖励函数的容错能力部分来自 critic 的结构先验。

GRPO 去掉了 critic（见第 1 章）。取而代之的是对同一 prompt $`q`$ 采样 $`G`$ 个回答 $`\{o_1, \dots, o_G\}`$，用组内均值 $`\bar{r} = \frac{1}{G}\sum_i r_i`$ 作为 $`V^\pi(q)`$ 的蒙特卡洛估计，用组内标准差 $`\hat{\sigma}`$ 做归一化（见第 2 章）。这意味着：

1. **组内奖励分布成为唯一的学习信号来源。** 没有 critic 来平滑或修正奖励；梯度完全由 $`\hat{A}_i = (r_i - \bar{r}) / \hat{\sigma}`$ 驱动。
2. **第 2 章的三个核心结论全部直接依赖奖励函数的性质：**
   - 组均值是 $`V^\pi(q)`$ 的无偏 MC 估计——前提是 $`r_i`$ 本身是对真实回报的合理度量。
   - $`\hat{\sigma}`$ 用于归一化——若奖励尺度不稳定或含系统性偏差，$`\hat{\sigma}`$ 会放大或扭曲信号。
   - 组内全同奖励导致 $`\hat{A}_i \equiv 0`$，该 prompt 无梯度贡献——这在二值奖励下尤为突出（详见 4.6 节）。
3. **GRPO 比 PPO 对奖励设计更敏感。** PPO 的 critic 至少能通过时序结构平滑一部分奖励噪声；GRPO 没有任何缓冲层，奖励函数的每一个缺陷都会直接传导到梯度上。

因此，在 GRPO 框架下讨论"如何训练策略"之前，必须先回答"奖励信号本身的统计性质是什么"。这不是工程细节，而是算法能否工作的前提条件。本章系统分析这一问题。

<a id="s4-2"></a>

## 4.2 两类奖励源的严格区分

<a id="s4-2-1"></a>

### 4.2.1 奖励模型（Reward Model, RM）

奖励模型是一个参数化的神经网络 $`R_\phi(q, o)`$，其训练数据来自人类偏好标注。给定同一个 prompt $`q`$ 和两个回答 $`o_w`$（preferred）、$`o_l`$（dispreferred），Bradley-Terry 偏好模型假设：

```math
P(o_w \succ o_l \mid q) = \sigma\big(R_\phi(q, o_w) - R_\phi(q, o_l)\big),
```

其中 $`\sigma(x) = 1/(1+e^{-x})`$ 是 logistic 函数。RM 的训练损失为负对数似然：

```math
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(q, o_w, o_l) \sim \mathcal{D}} \left[ \log \sigma\big(R_\phi(q, o_w) - R_\phi(q, o_l)\big) \right].
```

RM 的关键性质：
- **输出连续标量**，可对任意 $`(q, o)`$ 打分。
- **可微**，原则上可作为策略梯度的直接信号源。
- **本质是有偏近似**：RM 学到的是人类标注者偏好的代理（proxy），而非真实任务目标 $`U(o)`$。当标注者的判断存在系统性偏差（如偏好长回答、偏好特定格式）时，RM 会继承甚至放大这些偏差。
- **可被攻击**：策略可以通过优化 RM 分数而非真实质量来获得高分，即 reward hacking（详见 4.5 节）。

<a id="s4-2-2"></a>

### 4.2.2 规则/可验证奖励（Rule-based / Verifiable Reward）

规则奖励由确定性程序判定，不依赖任何学习的模型。其一般形式为：

```math
R(q, o) = f\big(\text{extract}(o),\; y^*\big),
```

其中：
- $`\text{extract}(\cdot)`$ 是从模型回答中抽取答案的解析函数，例如从 `\boxed{}` 中提取内容、从代码块中提取函数返回值。
- $`y^*`$ 是标准答案（ground truth），来自数据集标注。
- $`f`$ 是判定函数，通常为二值 $`f \in \{0, 1\}`$（对/错），也可以是多档评分（如部分正确得 0.5）。

典型例子包括：数学题的答案匹配、代码题的单元测试通过率、格式合规检查等。

<a id="s4-2-3"></a>

### 4.2.3 两类奖励的对比

| 维度 | 奖励模型 (RM) | 规则/可验证奖励 |
|------|--------------|----------------|
| 是否需要训练 | 是（需偏好数据 + GPU） | 否（纯程序逻辑） |
| 是否可微 | 是 | 否（通常不可微） |
| 输出类型 | 连续标量 | 离散（常为二值） |
| 噪声来源 | 标注噪声 + 模型泛化误差 | 解析器漏洞 + 测试覆盖不全 |
| 被攻击难度 | 低（RM 是高维空间的平滑函数，易被对抗性输入欺骗） | 高（确定性程序，但仍有边界情况） |
| 适用任务 | 开放式生成（对话、写作、摘要） | 有明确正确答案的任务（数学、代码、逻辑） |
| 计算成本 | 高（每次推理需过一遍 RM 网络） | 低（字符串匹配或测试执行） |

<a id="s4-2-4"></a>

### 4.2.4 为什么 GRPO 的成功集中在规则奖励场景

GRPO 的核心机制是对同一 prompt 采样 $`G`$ 次，然后用组内统计量构造优势。这一机制的有效性取决于一个关键前提：**$`G`$ 次采样的奖励分数是可靠的、可比较的**。

在规则奖励下，这个前提天然成立：verifier 是确定性的，同一个回答永远得到相同的分数，$`G`$ 个回答之间的分数差异完全来自回答质量的真实差异。组内均值 $`\bar{r}`$ 是对该 prompt 下通过率的精确估计，$`\hat{\sigma}`$ 反映了模型在该 prompt 上的真实不确定性。

在 RM 场景下，情况截然不同：
- RM 的输出本身带有噪声（模型泛化误差、标注不一致性）。
- $`G`$ 个回答可能同时触发 RM 的同一个系统性偏差（如都很长），导致组内排名反映的不是真实质量排序，而是 RM 偏差的一致方向。
- RM 的可微性意味着策略可以"集体性地"找到骗过 RM 的模式，而组内相对比较会进一步强化这种模式——因为只要某个回答比其他回答"更像高分回答"，它就会获得正优势，即使所有回答的真实质量都不高。

这就是为什么 GRPO 的巨大成功几乎全部发生在规则奖励场景（DeepSeek-R1、Qwen-Math 等），而在 RM 场景下，PPO 仍然是更稳健的选择。

<a id="s4-3"></a>

## 4.3 RLVR 范式

<a id="s4-3-1"></a>

### 4.3.1 定义与数据构成

可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）是指使用确定性 verifier 提供奖励信号的强化学习范式。其典型数据构成为：

- 一批带标准答案的 prompt 集合 $`\{(q_j, y^*_j)\}_{j=1}^N`$，涵盖数学题、代码题、逻辑推理题等。
- 训练时，对每个 prompt $`q_j`$，策略模型 $`\pi_\theta`$ 生成 $`G`$ 个回答 $`\{o_{j,1}, \dots, o_{j,G}\}`$。
- Verifier 对每个回答判定 $`r_{j,i} = f(\text{extract}(o_{j,i}), y^*_j)`$。
- 按 GRPO 目标函数更新策略（见第 3 章）。

<a id="s4-3-2"></a>

### 4.3.2 RLVR 与经典 RLHF 的关键差异

**奖励来源**：RLVR 使用确定性 verifier，RLHF 使用学出来的 RM。这一区别导致了以下连锁效应。

**KL 约束的必要性**：在 RLHF 中，KL 散度项 $`\beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]`$ 是必不可少的，因为 RM 是可被攻破的代理目标，不加约束的策略会迅速偏离人类偏好的有效区域（见 PPO 文档第 6 章）。在 RLVR 中，情况有所不同：

- Verifier 是不可被"话术"攻破的硬约束——数学答案对就是对，代码通过就是通过。策略大幅偏离初始模型并不会导致 verifier 给出虚假高分。
- 更重要的是，长链推理（Chain-of-Thought, CoT）任务本身就**希望**模型大幅偏离 SFT 初始分布，探索出新的、更有效的推理路径。过强的 KL 约束会将策略锚定在初始分布附近，扼杀这种探索。
- 因此，RLVR 实践中 KL 项常常被设为 0 或极小值。DAPO（第 6 章）明确移除了 KL 项。

**反面风险**：完全去掉 KL 并非没有代价。缺乏正则化可能导致输出退化：语言质量下降、重复生成、格式崩坏、长度失控等。实践中常见的折中方案包括：保留一个小 $`\beta`$（如 $`0.001`$）、改用熵约束、加入长度惩罚等。这是当前仍有争议的设计点，没有统一最优解。

<a id="s4-3-3"></a>

### 4.3.3 代表性工作

DeepSeek-R1 技术报告（arXiv:2501.12948）是 RLVR 范式的里程碑。其中 R1-Zero 直接从 base model 开始做 RLVR（不经 SFT 冷启动），仅使用规则奖励（数学答案匹配 + 格式检查），就观察到了显著的推理能力提升。报告中描述的"aha moment"现象——模型在训练过程中自发出现反思行为（如"让我重新检查一下"）——表明 RLVR 能够诱导模型自主发现有效的推理策略，而非仅仅模仿 SFT 数据中的推理模板。

<a id="s4-4"></a>

## 4.4 Outcome Reward 与 Process Reward 的数学区别

<a id="s4-4-1"></a>

### 4.4.1 Outcome Reward（结果奖励）

Outcome reward 只在回答结束时给出一个分数：

```math
r_t = \begin{cases} 0 & t < |o|, \\ R(q, o) & t = |o|. \end{cases}
```

承接第 1 章的分析，在 outcome reward 下，reward-to-go 对所有时间步 $`t`$ 相同（因为折扣因子 $`\gamma=1`$ 且中间奖励为零），故 GRPO 的优势不随时间变化：

```math
\hat{A}_{i,t} = \hat{A}_i = \frac{r_i - \bar{r}}{\hat{\sigma}}, \quad \forall\, t.
```

**优点**：
- Verifier 实现简单，只需判定最终答案。
- 无额外标注成本。
- 不易被中间步骤的话术欺骗（只看结果）。

**缺点——信用分配缺失的量化分析**：

设一条回答有 $`L`$ 个 token，其中只有 $`k`$ 个 token 是"关键决策点"（决定了答案对错），其余 $`L-k`$ 个 token 是常规的、对结果无实质影响的语言生成。在 outcome reward 下，这 $`L`$ 个 token 收到完全相同的优势信号 $`\hat{A}_i`$。

考虑策略梯度的形式（忽略 clip 和 KL，聚焦信号结构）：

```math
\nabla_\theta J \propto \hat{A}_i \sum_{t=1}^{L} \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}).
```

这个求和中，只有 $`k`$ 项对应真正的"有用方向"（调整关键决策的概率），其余 $`L-k`$ 项是噪声（它们在期望意义下与任务无关，但方差非零）。作为一个启发式的量级估计（heuristic scaling argument，非严格定理），有效信号与总噪声的信噪比约为：

```math
\text{SNR} \sim \frac{k}{L}.
```

当 CoT 很长时（$`L \gg k`$），信噪比极低，梯度估计的方差很大，训练效率急剧下降。这解释了为什么长 CoT 训练特别困难：不是因为模型"学不会"，而是因为 outcome reward 下的信用分配粒度太粗，有效信号被大量无关 token 稀释了。

<a id="s4-4-2"></a>

### 4.4.2 Process Reward（过程奖励）

Process reward 对推理的每一步或每个片段给分。形式化为：

```math
R(q, o) = \sum_{j=1}^{M} w_j \cdot r_j,
```

其中 $`j`$ 索引推理步骤（step），$`r_j`$ 是第 $`j`$ 步的得分，$`w_j`$ 是权重（常取 1）。实现方式有两种：
- **Process Reward Model (PRM)**：训练一个专门的模型对每步打分。
- **步骤级 verifier**：对每步进行确定性检验（如数学推导的中间等式是否成立）。

**优点**：信用分配精确，能定位到具体哪一步出错，梯度信号的信噪比显著优于 outcome reward。

**缺点**：
- 需要步骤级标注（极其昂贵）或训练 PRM（PRM 本身又是可被攻击的 RM，回到了 RM 的老问题）。
- "步骤"如何切分是模糊的：一个 token？一句话？一个完整的推理段落？不同切分方式导致不同的奖励结构。
- PRM 的分数尺度难以统一：不同步骤的难度不同，同样的分数增量在不同位置的含义不同。

<a id="s4-4-3"></a>

### 4.4.3 当前主流选择

当前 RLVR 的主流是 **outcome reward + GRPO**，因为它绕开了 PRM 的所有难题（标注成本、步骤切分、PRM 可靠性）。代价就是前述的长 CoT 信用分配困难。这也是 token 级信用分配研究（第 2 章 2.7 节提及）持续活跃的根本动机：如何在保持 outcome reward 简洁性的同时，恢复细粒度的信用分配。

<a id="s4-5"></a>

## 4.5 Reward Hacking 的机理与表现

<a id="s4-5-1"></a>

### 4.5.1 形式化定义

设真实目标是最大化效用 $`U(o)`$，但我们优化的代理目标是奖励 $`R(q, o)`$。定义两者之间的 gap：

```math
\Delta(\theta) = \mathbb{E}_{o \sim \pi_\theta(\cdot|q)}[R(q, o)] - \mathbb{E}_{o \sim \pi_\theta(\cdot|q)}[U(o)].
```

Reward hacking 指的是：随着优化推进，$`\mathbb{E}[R]`$ 持续上升，但 $`\mathbb{E}[U]`$ 停滞甚至下降，即 $`\Delta(\theta)`$ 被优化压力主动放大。这是 Goodhart's Law 的数学表述：一旦一个度量成为优化目标，它就不再是好度量。

关键在于，优化算法并不"知道" $`R`$ 和 $`U`$ 的区别。它只看到 $`R`$ 的梯度，而梯度方向恰好是 $`\Delta`$ 增大的方向（因为 $`R = U + \Delta`$，最大化 $`R`$ 等价于同时最大化 $`U`$ 和 $`\Delta`$，而 $`\Delta`$ 通常在高维空间中更容易被放大）。

<a id="s4-5-2"></a>

### 4.5.2 RM 场景的典型 Hacking

- **谄媚（Sycophancy）**：模型学会附和用户的观点而非给出真实判断，因为人类标注者在偏好数据中系统性地偏好与自己一致的回答。
- **冗长偏好（Verbosity Bias）**：RM 系统性偏好长回答（标注者倾向于认为详细的回答更好），模型学会填充无关内容以增加长度。
- **格式套路化**：模型学会使用特定的开头（"Great question!"）、结构（分点列举、加粗关键词）来触发 RM 的高分模式。
- **编造引用**：模型生成看似权威但实际不存在的文献引用，因为 RM 无法验证引用的真实性但会给"看起来有据可查"的回答高分。

**GRPO 在 RM 场景下会加速 hacking**：组内相对比较强化了"比同组其他回答更像高分回答"的特征。如果 $`G`$ 个回答中有几个碰巧触发了 RM 的某个偏差模式并获得了较高分数，它们就会获得正优势，策略会进一步强化这种模式。这是一种正反馈循环。

<a id="s4-5-3"></a>

### 4.5.3 RLVR 场景的 Hacking

虽然 verifier 是确定性的，但 $`R(q,o)`$ 与 $`U(o)`$ 之间仍然存在 gap，只是 gap 的来源不同：

- **答案抽取漏洞**：verifier 用正则表达式从 `\boxed{}` 中提取答案。模型学会输出多个 `\boxed{}`、或在 `\boxed{}` 中塞入所有候选答案（如 `\boxed{A \text{ or } B \text{ or } C}`），使 extract 函数返回的内容碰巧匹配标准答案。
- **测试集过拟合（代码任务）**：模型学会硬编码单元测试的期望输出（如 `if input == [1,2,3]: return 6`），而非编写通用逻辑。verifier 判对，但模型的泛化能力为零。
- **截断利用**：模型故意生成超长回答，触发最大长度截断。某些 verifier 在截断后无法正确解析答案，给出默认分数（如 0.5 或跳过），模型利用这个默认值获得非零奖励。
- **枚举猜测（数学任务）**：在无法推导时，模型枚举所有可能的选项（如"A, B, C, D 都有可能"）， hoping 某个子串匹配标准答案。

每一条的数学本质都是相同的：$`R(q,o)`$ 的实现与 $`U(o)`$ 之间存在 gap，优化压力发现并放大了这个 gap。

<a id="s4-5-4"></a>

### 4.5.4 缓解手段的机理

- **加固 verifier**：更鲁棒的答案抽取逻辑、更多的单元测试用例、对抗性测试（专门检测 hacking 模式的测试）。机理：直接缩小 $`R`$ 与 $`U`$ 之间的 gap。
- **保留 KL 约束到 $`\pi_{\text{ref}}`$**：KL 惩罚 $`D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]`$ 将优化限制在 $`\pi_{\text{ref}}`$ 的邻域内。Hacking 模式通常存在于远离 $`\pi_{\text{ref}}`$ 的区域（因为初始模型不会输出 hacking 模式），KL 约束限制了策略能走多远，从而限制了它能发现的 gap 大小。
- **长度惩罚 / 超长奖励塑形**：直接堵住"靠长度骗分"这条路径。DAPO 的做法是对超过阈值的回答施加递减奖励（详见第 6 章）。
- **多样化奖励组合**：多个 verifier 投票、加入格式分、加入多样性惩罚。机理：提高同时骗过所有信号的成本。当 $`R = \sum_k w_k R_k`$ 时，hacking 需要同时在所有 $`R_k`$ 上找到可利用的 gap，这比攻破单一信号难得多。

<a id="s4-6"></a>

## 4.6 二值奖励下 GRPO 的特殊动力学

<a id="s4-6-1"></a>

### 4.6.1 组内统计量的解析表达

RLVR 中最常见的奖励是二值的：$`r \in \{0, 1\}`$。设某个 prompt $`q`$ 下模型的通过率为 $`p = P(R=1 \mid q)`$。

当 $`G`$ 足够大时，组内均值和标准差的期望为：

```math
\mathbb{E}[\bar{r}] = p, \quad \mathbb{E}[\hat{\sigma}] \approx \sqrt{p(1-p)}.
```

这是因为 $`G`$ 次独立采样的奖励服从二项分布 $`\text{Bin}(G, p)`$，其标准差为 $`\sqrt{Gp(1-p)}`$，除以 $`G`$ 后得到样本均值的标准差；而样本标准差 $`\hat{\sigma}`$ 的期望在 $`G`$ 较大时趋近于总体标准差 $`\sqrt{p(1-p)}`$。

于是优势的典型量级为：

```math
\hat{A}_i \approx \frac{r_i - p}{\sqrt{p(1-p)}}.
```

<a id="s4-6-2"></a>

### 4.6.2 信号强度的退化分析

当 $`p \to 1`$（题太简单，几乎全对）或 $`p \to 0`$（题太难，几乎全错）时，信号都会退化：

- **$`p \to 1`$**：组内几乎全为 1，$`\hat{\sigma} \to 0`$，分子 $`(r_i - p) \to 0`$。由第 2 章结论，全同奖励导致 $`\hat{A}_i \equiv 0`$，该 prompt 无梯度贡献。
- **$`p \to 0`$**：同理，几乎全为 0，零梯度。
- **$`p \approx 0.5`$ 时信号最强**：$`\sqrt{p(1-p)}`$ 在 $`p=0.5`$ 时取最大值 $`0.5`$，且组内既有对又有错，对比最鲜明。

<a id="s4-6-3"></a>

### 4.6.3 数值表

下表给出了不同通过率下的关键统计量。全同奖励概率为 $`P(\text{all same}) = (1-p)^G + p^G`$（全为 0 或全为 1 的概率之和）。

| $`p`$ | $`\sqrt{p(1-p)}`$（期望 std） | 全同概率 ($`G=8`$) | 全同概率 ($`G=16`$) | 有效信号强度 |
|-----|---------------------------|------------------|-------------------|-------------|
| 0.05 | 0.218 | $`0.95^8 + 0.05^8 \approx 0.6634`$ | $`0.95^{16} + 0.05^{16} \approx 0.4401`$ | 极弱 |
| 0.20 | 0.400 | $`0.80^8 + 0.20^8 \approx 0.1678`$ | $`0.80^{16} + 0.20^{16} \approx 0.0281`$ | 中等 |
| 0.50 | 0.500 | $`2 \times 0.5^8 = 0.0078`$ | $`2 \times 0.5^{16} \approx 0.0000305`$ | 最强 |
| 0.80 | 0.400 | $`0.20^8 + 0.80^8 \approx 0.1678`$ | $`0.20^{16} + 0.80^{16} \approx 0.0281`$ | 中等 |
| 0.95 | 0.218 | $`0.05^8 + 0.95^8 \approx 0.6634`$ | $`0.05^{16} + 0.95^{16} \approx 0.4401`$ | 极弱 |

注：$`G=8, p=0.05`$ 时，$`0.95^8 = 0.66342`$，$`0.05^8 = 3.906 \times 10^{-11}`$，故全同概率 $`\approx 0.6634`$。$`G=16, p=0.95`$ 时，$`0.95^{16} = 0.44013`$，$`0.05^{16} \approx 1.5 \times 10^{-21}`$，故全同概率 $`\approx 0.4401`$。$`p=0.5, G=16`$ 时，$`2 \times (1/2)^{16} = 2/65536 \approx 3.05 \times 10^{-5}`$。

<a id="s4-6-4"></a>

### 4.6.4 两个重要实践结论

**① 数据难度配比是 RLVR 训练的第一超参。** 应该让训练集中大部分 prompt 的通过率落在 $`0.2 \sim 0.8`$ 区间。否则，大量算力浪费在零梯度的 prompt 上（$`p \approx 0`$ 或 $`p \approx 1`$ 的题对训练几乎没有贡献）。

**② 训练过程中难度会自动漂移。** 随着模型变强，原本 $`p=0.5`$ 的题逐渐变成 $`p=0.95`$，信号枯竭。这不是工程瑕疵，而是 GRPO + 二值奖励的内在动力学特性。解决方案包括：动态采样（DAPO，第 6 章详述）、课程学习（逐步引入更难的题）、持续补充新的高难度数据。如果不处理这个问题，训练后期会出现明显的"高原期"——loss 不再下降，不是因为模型已经收敛，而是因为有效信号消失了。

<a id="s4-7"></a>

## 4.7 奖励的尺度、稀疏性与组内归一化的相互作用

<a id="s4-7-1"></a>

### 4.7.1 尺度不变性的证明

GRPO 的优势定义为 $`\hat{A}_i = (r_i - \bar{r}) / \hat{\sigma}`$。考虑对奖励做仿射变换 $`r'_i = a r_i + b`$（$`a > 0`$），则：

```math
\bar{r}' = a\bar{r} + b, \quad \hat{\sigma}' = a\hat{\sigma},
```

```math
\hat{A}'_i = \frac{r'_i - \bar{r}'}{\hat{\sigma}'} = \frac{a r_i + b - (a\bar{r} + b)}{a\hat{\sigma}} = \frac{a(r_i - \bar{r})}{a\hat{\sigma}} = \hat{A}_i.
```

**结论：GRPO 的优势对奖励的仿射变换不变。** 这意味着奖励的绝对尺度和偏移量不影响梯度方向（在不考虑 KL 项的前提下）。

**实践意义**：不需要像 PPO 那样做奖励归一化或缩放。不同任务（如数学题的二值奖励和代码题的多档奖励）可以直接混合训练，无需手动对齐量级。这是 GRPO 相对于 PPO 的一个重要便利。

<a id="s4-7-2"></a>

### 4.7.2 KL 项打破尺度不变性

一旦引入 KL 惩罚项 $`\beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]`$，情况发生变化。KL 散度有固定的绝对尺度（单位为 nat 或 bit），不受奖励变换的影响。当奖励从 $`r`$ 变为 $`ar+b`$ 时，目标函数变为：

```math
J'(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \left(r'_{i,t}(\theta)\,\hat{A}'_i - \beta \cdot D_{\text{KL}}\right)\right] = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \left(r_{i,t}(\theta)\,\hat{A}_i - \frac{\beta}{a} \cdot D_{\text{KL}}\right)\right],
```

其中我们利用了 $`\hat{A}'_i = \hat{A}_i`$ 以及 $`r'_{i,t}(\theta) = a \cdot r_{i,t}(\theta)`$（比率项也乘以 $`a`$，但与 $`\hat{A}'_i`$ 相乘后 $`a`$ 被抵消……等等，这里需要更仔细）。

实际上，回顾第 3 章的目标函数：

```math
J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \left(\min\big(r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_i\big) - \beta \cdot D_{\text{KL}}\right)\right].
```

由于 $`\hat{A}_i`$ 尺度不变，clip 内的两项都正比于 $`\hat{A}_i`$，所以整个奖励项确实尺度不变。但 KL 项不变，因此**奖励尺度变化等价于改变 $`\beta`$ 的有效值**。奖励放大 $`a`$ 倍等价于 $`\beta`$ 缩小 $`a`$ 倍。

**实践警告**：混合多个奖励源时，必须先统一各分量的尺度，再设置 $`\beta`$。否则，某个分量如果量级特别大，会隐式地压低 KL 约束的效果，导致训练不稳定。

<a id="s4-7-3"></a>

### 4.7.3 多目标奖励的组合

当奖励由多个分量组成时，$`R = \sum_k w_k R_k`$（如正确性 + 格式 + 长度），组内归一化作用于合成后的总分。这意味着各分量的相对权重 $`w_k`$ 不仅决定了它们在总分中的占比，还决定了它们在组内排序中的影响力。

更精细的做法是对每个分量分别做组内归一化再加权：

```math
\hat{A}_i = \sum_k w_k \cdot \frac{R_k(o_i) - \overline{R_k}}{\hat{\sigma}_k}.
```

这种方式确保每个分量的贡献不受其他分量尺度的影响，但引入了额外的计算开销和超参（每个分量的归一化统计量）。这种变体在第 7 章讨论序列级方法时会进一步涉及。

<a id="s4-8"></a>

## 4.8 奖励与 GRPO 假设的一致性检查清单

在设计 RLVR 训练之前，建议逐条检查以下问题：

1. **能否对同一 prompt 廉价地多次采样并得到可靠分数？**
   - 这是 GRPO 的前提。如果 verifier 很慢（如需要运行完整测试套件）或结果不确定（如 flaky tests），$`G`$ 次采样的成本过高或统计量不可靠。
   - *如果答否*：考虑减少 $`G`$、缓存 verifier 结果、或切换到不需要组内采样的方法（如 PPO + RM）。

2. **奖励是否是 outcome-level 的？**
   - GRPO 天然适配 outcome reward。如果任务的本质是 process-level 的（如多步证明中每步都需要正确），outcome reward 会导致严重的信用分配问题。
   - *如果答否*：考虑 PRM + PPO，或在 outcome reward 基础上叠加 token 级信用分配的改进方法（第 2 章 2.7 节、第 7 章变体）。

3. **通过率分布是否覆盖 $`0.2 \sim 0.8`$？**
   - 由 4.6 节的分析，这是有效信号存在的区间。
   - *如果答否*：实施课程学习、难度过滤、或动态采样（第 6 章）。也可以在训练前对数据集做预评估，剔除通过率极端的数据。

4. **Verifier 是否可被话术/格式绕过？**
   - 需要做对抗性测试：故意构造 hacking 模式的输入，检查 verifier 是否正确拒绝。
   - *如果答否*：加固 verifier（更多测试用例、更严格的解析逻辑、对抗性测试集）。保留 KL 约束作为兜底。

5. **是否需要 KL 约束？**
   - 取决于两个因素：(a) 任务是否希望模型大幅偏离初始分布（长 CoT 探索通常需要）；(b) 输出退化的风险有多高（开放域生成风险高，封闭域推理风险低）。
   - *如果需要大幅偏离且退化风险可控*：可以去掉 KL 或设极小 $`\beta`$（如 DAPO）。
   - *如果退化风险高*：保留 $`\beta > 0`$，或改用熵约束、长度惩罚等替代正则。

<a id="s4-9"></a>

## 4.9 本章小结与自测

<a id="h133"></a>

### 要点总结

1. GRPO 去掉了 critic，组内奖励分布成为唯一学习信号来源，奖励设计的容错空间远小于 PPO。
2. 规则奖励与奖励模型在可靠性、可攻击性、适用场景上有本质区别；GRPO 的成功集中在规则奖励场景，因为组内统计量在确定性 verifier 下才是可靠的。
3. RLVR 范式下 KL 约束常被弱化或移除，因为 verifier 不可被话术攻破，且长 CoT 探索需要策略大幅偏离初始分布；但完全去掉 KL 有输出退化风险。
4. Outcome reward 的信噪比约为 $`k/L`$（$`k`$ 为关键 token 数，$`L`$ 为总长度），解释了长 CoT 训练的困难；process reward 解决了信用分配但引入了 PRM 的新问题。
5. 二值奖励下，通过率 $`p \approx 0.5`$ 时信号最强，$`p \to 0`$ 或 $`p \to 1`$ 时信号枯竭；数据难度配比是 RLVR 的第一超参，训练中需应对难度自动漂移。
6. GRPO 优势对奖励的仿射变换不变，但 KL 项打破了这一不变性；混合奖励源时必须先统一尺度。

<a id="h134"></a>

### 自测题

**题 1**：证明 GRPO 的优势 $`\hat{A}_i`$ 对奖励的仿射变换 $`r \mapsto ar+b`$（$`a>0`$）不变。如果同时存在 KL 惩罚项 $`\beta D_{\text{KL}}`$，奖励放大 $`a`$ 倍等价于对 $`\beta`$ 做了什么变换？

参考答案：$`\hat{A}'_i = (ar_i+b - (a\bar{r}+b))/(a\hat{\sigma}) = (r_i-\bar{r})/\hat{\sigma} = \hat{A}_i`$。KL 项不受奖励变换影响，因此奖励放大 $`a`$ 倍等价于 $`\beta`$ 缩小为 $`\beta/a`$（即 KL 约束的相对权重降低 $`a`$ 倍）。

**题 2**：在二值奖励下，设某 prompt 的通过率 $`p=0.95`$，采样数 $`G=8`$。计算全同奖励的概率，并解释为什么这种情况下训练信号几乎消失。如果将 $`G`$ 增大到 16，全同概率变为多少？增大 $`G`$ 是否能根本解决这个问题？

参考答案：$`G=8`$ 时全同概率 $`= 0.95^8 + 0.05^8 \approx 0.6634`$；$`G=16`$ 时 $`\approx 0.95^{16} + 0.05^{16} \approx 0.4401`$。增大 $`G`$ 降低了全同概率，但当 $`p`$ 接近 1 时，即使不全同，组内绝大多数样本仍为 1，$`\hat{\sigma}`$ 很小，$`\hat{A}_i`$ 的量级仍然很低。根本问题是 $`p`$ 本身偏离 0.5 太远，需要从数据难度配比或动态采样层面解决。

**题 3**：为什么 RLVR 中可以弱化甚至去掉 KL 约束，而 RLHF 中不行？给出两个独立的理由。

参考答案：(1) Verifier 是确定性硬约束，不可被话术攻破，策略偏离初始分布不会导致虚假高分；(2) 长 CoT 任务需要策略大幅偏离 SFT 初始分布以探索新推理路径，过强 KL 会扼杀探索。

**题 4**：设一条 CoT 回答有 $`L=500`$ 个 token，其中 $`k=10`$ 个是关键决策 token。在 outcome reward 下，估算策略梯度信噪比的量级。如果使用 process reward 且能精确定位到每个关键步骤，信噪比改善多少倍？

参考答案：Outcome reward 下 SNR $`\sim k/L = 10/500 = 0.02`$。Process reward 若能精确定位，噪声项从 $`L-k`$ 降至接近 0，SNR 改善约 $`L/k = 50`$ 倍。

**题 5**：列举两种 RLVR 场景下的 reward hacking 模式，并分别说明其数学本质（即 $`R`$ 与 $`U`$ 之间的 gap 是如何被利用的）。

参考答案：(1) 答案抽取漏洞：模型在 `\boxed{}` 中塞入多个候选答案，extract 函数碰巧匹配标准答案。Gap 来源：extract 函数的实现不等于"答案正确"这一语义。(2) 测试集过拟合：模型硬编码测试用例的输入输出映射。Gap 来源：有限测试用例的通过率不等于程序的通用正确性。两者的共同本质是 $`R`$ 的实现与 $`U`$ 的语义之间存在可被优化压力发现和放大的 gap。

<a id="chapter-5"></a>

# 第 5 章 GRPO 的已知偏差与 Dr.GRPO 修正

<a id="s5-1"></a>

## 5.1 为什么 GRPO 会有系统性偏差

在前几章中，我们已经完整建立了 GRPO 的目标函数（见第 3 章 §3.2）及其理论基础（见第 2 章）。本章不再重复这些推导，而是聚焦于一个在实践中至关重要、但在原始论文中被忽略的问题：**GRPO 的目标函数本身包含系统性偏差（systematic bias），会导致策略朝错误方向收敛。**

这里所说的"偏差"需要精确定义。它不是指梯度估计量的统计偏差（statistical bias）——如第 2 章所证，组均值基线作为 $`V^\pi(q)`$ 的蒙特卡洛估计是无偏的。本章讨论的偏差是**优化目标层面的扭曲**：GRPO 目标函数中的某些归一化项改变了不同样本在损失中的相对权重，使得策略梯度的期望方向偏离了真实目标 $`\nabla_\theta \mathbb{E}[R(\mathbf{q}, \mathbf{o})]`$ 的方向。换言之，即使我们有无穷多采样、梯度估计完全精确，GRPO 仍然会收敛到一个并非最优的策略。

具体来说，GRPO 目标函数中有两个"看起来无害、实则引入偏差"的归一化项：

1. **序列内的 $`(1/|o_i|)`$ 长度归一化**：对第 $`i`$ 个回答的所有 token 取平均时除以该回答的长度。这一项使得不同长度回答的每个 token 获得的梯度权重不同，引入**长度偏差（response-level length bias）**。
2. **组内的 $`(1/\text{std})`$ 标准差归一化**：将奖励减去均值后除以组内标准差。这一项使得不同 prompt 的优势量级被各自的组内方差所调制，引入**难度偏差（question-level difficulty bias）**。

这两项在经典 RL 中都有对应的"合理"动机：长度归一化来自预训练阶段对固定长度上下文取平均的习惯；标准差归一化来自 advantage normalization 的常见 trick。但在 GRPO 的特定结构下，它们各自引入了与推理质量无关的系统性偏置。

本章的核心参考是 Liu et al. (2025)《Understanding R1-Zero-Like Training: A Critical Perspective》（arXiv:2503.20783），即 Dr.GRPO 论文。该工作的核心贡献就是指出上述两项偏差并给出极简修正：去掉这两个归一化项。下面逐节展开严格推导。

<a id="s5-2"></a>

## 5.2 长度偏差（response-level length bias）的严格推导

<a id="s5-2-1"></a>

### 5.2.1 从 GRPO 目标函数的内层求和入手

回顾第 3 章给出的带 clip 的 GRPO 实用形式（Eq. 68 的负损失版本）：

```math
L^{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\!\left(r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}\!\left(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\right)\hat{A}_i\right) - \beta\,\hat{D}_{KL,i,t}\right]
```

其中 $`r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q,o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})`$ 为逐 token 重要性比率，$`\hat{A}_i = (r_i - \text{mean})/\text{std}`$ 为组内归一化优势。

本节暂时忽略 KL 项和 clip 机制（它们不影响长度偏差的分析），聚焦于替代目标部分。对于第 $`i`$ 个回答，其对损失的贡献为：

```math
L_i(\theta) = -\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} r_{i,t}(\theta)\,\hat{A}_i
```

<a id="s5-2-2"></a>

### 5.2.2 梯度分析

对 $`L_i(\theta)`$ 求关于 $`\theta`$ 的梯度。在 $`K=1`$（单 epoch 更新）、$`r_{i,t} \approx 1`$ 的近似下（承接第 3 章 §3.6 的分析），有：

```math
\nabla_\theta L_i(\theta) \approx -\frac{\hat{A}_i}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} | q, o_{i,<t})
```

这个表达式的结构非常清晰：**整个回答的梯度方向由 $`\sum_t \nabla_\theta \log \pi_\theta`$ 决定（这是标准的 REINFORCE 梯度方向），但其幅度被因子 $`\hat{A}_i / |o_i|`$ 所缩放。**

关键观察：**同一个优势值 $`\hat{A}_i`$，作用在不同长度的回答上时，每个 token 分到的梯度权重是 $`\hat{A}_i / |o_i|`$。** 设同一 prompt 下有两个回答 $`o_a`$ 和 $`o_b`$，具有相同的优势值 $`\hat{A}`$，但 $`|o_a| < |o_b|`$，则：

```math
\frac{|\hat{A}|}{|o_a|} > \frac{|\hat{A}|}{|o_b|}
```

即短回答的每个 token 获得更大的梯度更新幅度。

<a id="s5-2-3"></a>

### 5.2.3 分情况讨论：不对称的后果

这是 Dr.GRPO 论文的关键洞察。$`(1/|o_i|)`$ 引入的并非简单的"短回答权重更大"，而是一种**与优势符号耦合的不对称效应**。我们分两种情况讨论。

**情形 1：$`\hat{A}_i > 0`$（正确回答，正优势）。** 此时梯度方向是增大各 token 的生成概率（正向更新）。由于短回答每个 token 获得的正向更新幅度 $`\hat{A}_i/|o_i|`$ 更大，策略被推向"在正确答案中偏好更简洁的表达"。也就是说，如果模型生成了两个都正确的回答，一个长一个短，短回答对策略参数的"拉力"更强。

**情形 2：$`\hat{A}_i < 0`$（错误回答，负优势）。** 此时梯度方向是减小各 token 的生成概率（负向更新，即惩罚）。长回答的 $`|o_i|`$ 更大，$`(1/|o_i|)`$ 使惩罚被摊薄，每个 token 受到的负向更新幅度 $`|\hat{A}_i|/|o_i|`$ 更小。也就是说，如果模型生成了两个都错误的回答，一个长一个短，长回答对策略参数的"推力"反而更弱——**惩罚不够狠**。

**合起来的净效应**：GRPO 系统性地鼓励"对就简短、错就啰嗦"的行为模式。这是一种与推理质量完全无关的长度偏置（length bias），其根源纯粹是目标函数中 $`(1/|o_i|)`$ 这一归一化项。

<a id="s5-2-4"></a>

### 5.2.4 为什么这在数学推理中有害

在数学推理任务中，这种偏置尤其危险：

- **"错就啰嗦"**：模型可能学会用冗长的错误推理来"蒙混"。由于长错误回答受到的惩罚被摊薄，模型没有足够强的信号去避免生成冗长的无效推理链。这直接导致训练中观察到的"回答越来越长但质量不升"现象——Dr.GRPO 论文将其称为 overthinking（过度思考）。
- **"对就简短"**：模型可能在正确回答中跳过必要的推理步骤。虽然简洁本身不是坏事，但在数学推理中，省略关键步骤往往意味着模型并未真正掌握推理过程，只是在记忆答案模式。

需要强调的是，这种偏置与"长链思维（long chain-of-thought）是否有用"是两个独立的问题。长链思维本身可能是有益的（允许模型进行更深入的推理），但 GRPO 的长度偏差导致的长度增长是**非选择性的**——它不分对错地改变了长度分布，且对错误回答的宽容度更高。

<a id="s5-2-5"></a>

### 5.2.5 Dr.GRPO 的修正：去掉 $`(1/|o_i|)`$

修正方案极其简单：**去掉序列内的 $`(1/|o_i|)`$ 归一化**，改用全局常数归一化。具体而言，将目标函数中的

```math
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} (\cdots)
```

替换为

```math
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{L_{\max}}\sum_{t=1}^{|o_i|} (\cdots)
```

其中 $`L_{\max}`$ 是一个全局常数（如最大生成长度或该 batch 的总 token 数）。等价地，也可以直接写成对所有 response token 取平均的形式：

```math
\frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} (\cdots)
```

**修正后长度偏差消失的数学原因**：每个 token 的梯度权重变为 $`\hat{A}_i / L_{\max}`$（或 $`\hat{A}_i / \sum_j |o_j|`$），不再依赖其所属回答的长度 $`|o_i|`$。无论回答长短，同一优势值下的每个 token 获得相同尺度的梯度更新。

<a id="s5-2-6"></a>

### 5.2.6 长度偏差也存在于开源 PPO 实现

Dr.GRPO 论文的一个重要发现是：**这个长度偏差不只存在于 GRPO，很多开源的 vanilla PPO 实现也犯了同样的错误。** PPO 的理论目标函数（见 PPO 文档第 5 章 Eq. 2）是对所有 token 直接求和（或等价地对所有 token 取平均），不包含按回答长度归一化的操作。但实际代码实现中，普遍采用了先对每个回答按 $`|o_i|`$ 取平均、再对回答取平均的两步聚合方式。

以下是伪代码级别的对比：

```python
# === 有偏写法（广泛存在于开源实现中）===
# 先对每个回答内部按 |o_i| 取平均，再对 batch 中的回答取平均
# 数学上等价于 (1/G) Σ_i (1/|o_i|) Σ_t loss_{i,t}
loss_biased = masked_mean(ppo_loss, response_mask, dim=-1).mean()

# === 无偏写法（Dr.GRPO 主张）===
# 直接对所有 response token 取平均，即除以总 token 数
# 数学上等价于 (1/Σ|o_i|) Σ_i Σ_t loss_{i,t}
# 或者用全局常数 MAX_TOKENS 归一化
loss_unbiased = masked_mean(ppo_loss, response_mask, dim=None).mean()
```

两者的数学差异正是 $`(1/|o_i|)`$ 那一项。`dim=-1` 表示沿 token 维度取平均（每个回答独立归一化），`dim=None` 表示对所有元素取平均（全局归一化）。Dr.GRPO 论文检查了 TRL、OpenRLHF、verl、SimpleRL-Zero、Open-Reasoner-Zero 等多个主流框架，发现它们全部使用了有偏写法（具体见 Dr.GRPO 论文 Table 2）。这一"公式-实现对齐"问题在 GRPO 出现之前就已存在，可能源于预训练阶段对固定长度上下文取 `loss.mean(-1)` 以提高数值稳定性的习惯。

<a id="s5-3"></a>

## 5.3 难度偏差（question-level difficulty bias）的严格推导

<a id="s5-3-1"></a>

### 5.3.1 从组内 std 归一化入手

GRPO 的优势定义为：

```math
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}
```

其中 $`\text{mean}`$ 和 $`\text{std}`$ 都是对同一 prompt $`q`$ 下 $`G`$ 个回答的奖励计算的。记 prompt $`q`$ 的组内奖励标准差为 $`\hat{\sigma}(q) = \text{std}(\{r_j\}_{j=1}^G)`$。

**关键点**：不同 prompt 的 $`\hat{\sigma}(q)`$ 不同。对于组内奖励高度一致的 prompt（要么几乎全对、要么几乎全错），$`\hat{\sigma}(q)`$ 很小，$`(1/\hat{\sigma}(q))`$ 会**放大**这些 prompt 的优势与梯度；对于组内奖励分散的 prompt（部分对部分错，即"难度适中"的题），$`\hat{\sigma}(q)`$ 较大，梯度被相对压制。

<a id="s5-3-2"></a>

### 5.3.2 与第 4 章信号强度分析的矛盾

这个行为与直觉相反，也与第 4 章的分析结论相矛盾。第 4 章 §4.6 指出：在二值奖励下，通过率 $`p \approx 0.5`$ 时信号最强（信息量最大），$`p \to 0`$ 或 $`p \to 1`$ 时信号枯竭。我们自然希望难度适中的题主导学习。

然而，std 归一化的效果恰好相反：当 $`p \to 0`$ 或 $`p \to 1`$ 时，$`\hat{\sigma} \to 0`$，优势量级爆炸，这些低信息量的边缘题反而获得了最大的梯度权重。**std 归一化把权重错误地分配给了低信息量的边缘题，压制了高信息量的中等难度题。**

<a id="s5-3-3"></a>

### 5.3.3 二值奖励下的严格刻画

考虑二值奖励 $`r_i \in \{0, 1\}`$。设组内 $`G`$ 个回答中有 $`k`$ 个正确（$`r_i=1`$）、$`G-k`$ 个错误（$`r_i=0`$）。则：

```math
\text{mean} = \frac{k}{G}, \quad \text{std} = \sqrt{\frac{k}{G}\left(1 - \frac{k}{G}\right)}
```

这里使用的是总体标准差（population std）。对于一个正确的回答（$`r_i = 1`$），其优势为：

```math
\hat{A}_{\text{correct}} = \frac{1 - k/G}{\sqrt{(k/G)(1 - k/G)}} = \sqrt{\frac{1 - k/G}{k/G}}
```

类似地，对于一个错误的回答（$`r_i = 0`$）：

```math
\hat{A}_{\text{incorrect}} = \frac{0 - k/G}{\sqrt{(k/G)(1 - k/G)}} = -\sqrt{\frac{k/G}{1 - k/G}}
```

下表给出了不同 $`k/G`$ 值时的 std 与正确答案优势量级：

| $`k/G`$ | $`\text{std} = \sqrt{(k/G)(1-k/G)}`$ | $`\hat{A}_{\text{correct}}`$ | 含义 |
|-------|-------------------------------------|---------------------------|------|
| 0.1   | 0.300                               | 3.00                      | 极难题，偶尔做对 → 优势极大 |
| 0.3   | 0.458                               | 1.53                      | 较难题 |
| 0.5   | 0.500                               | 1.00                      | 中等难度 → 优势最温和 |
| 0.7   | 0.458                               | 0.65                      | 较易题 |
| 0.9   | 0.300                               | 0.33                      | 极简单题，偶尔做错 → 优势较小 |

注意两个事实：

1. **std 在 $`k/G = 0.5`$ 时取最大值 0.5**，此时优势量级最温和（$`\hat{A}_{\text{correct}} = 1.0`$）。
2. **当 $`k/G \to 0`$ 或 $`k/G \to 1`$ 时，std $`\to 0`$，优势量级趋向无穷大**。例如 $`k/G = 0.1`$ 时正确答案的优势是 $`k/G = 0.5`$ 时的 3 倍。

这意味着：一道"几乎没人做对、但碰巧有一个回答做对了"的极难题，其梯度权重是一道"一半人做对"的中等难度题的数倍。而从信息论角度看，前者提供的学习信号远弱于后者（因为那个唯一的正确答案可能只是运气好）。

<a id="s5-3-4"></a>

### 5.3.4 Dr.GRPO 的修正：去掉 $`(1/\text{std})`$

修正方案同样简单：**去掉组内的 $`(1/\text{std})`$ 归一化**，只用减均值的优势：

```math
\tilde{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G)
```

去掉后，各 prompt 的优势量级回归到奖励本身的自然尺度。在二值奖励下，$`\tilde{A}_{\text{correct}} = 1 - k/G`$，$`\tilde{A}_{\text{incorrect}} = -k/G`$，其绝对值不超过 1，不再有边缘难度被放大的问题。

<a id="s5-3-5"></a>

### 5.3.5 数值稳定性与零梯度的澄清

去掉 std 归一化后，需要澄清一个细微但重要的问题。

**若某 prompt 全对（$`k=G`$）或全错（$`k=0`$）**，则 $`\text{mean} = 1`$ 或 $`\text{mean} = 0`$，所有 $`\tilde{A}_i = 0`$，梯度为零。这与第 2 章的结论一致：组内奖励全同时，无论是否除以 std，优势都为零（全对时分子为零，全错时分子也为零）。去掉 std 不改变"全同奖励零梯度"的事实。

**区别在于"部分对部分错时各题的相对权重"**。保留 std 时，$`k/G = 0.1`$ 的题比 $`k/G = 0.5`$ 的题获得约 3 倍的梯度权重；去掉 std 后，$`k/G = 0.1`$ 的题的正确答案优势为 $`0.9`$，$`k/G = 0.5`$ 的题为 $`0.5`$，前者仅约为后者的 1.8 倍（而非 3 倍），且这个比例完全由奖励的自然差异决定，不被方差人为放大。

**重要提醒**：去掉 std 归一化并不能解决"全同奖励零梯度"的信号枯竭问题。那是第 6 章 DAPO 动态采样要解决的问题——通过过滤掉全同奖励的 prompt 并补充新样本来保证有效梯度信号。不要误以为 Dr.GRPO 的修正能替代动态采样。

<a id="s5-4"></a>

## 5.4 loss 聚合方式（token-mean vs sequence-mean）

<a id="s5-4-1"></a>

### 5.4.1 三种常见聚合方式

承接 §5.2 的讨论，"如何把 token 级损失聚合成 batch 损失"是一个独立的设计维度。我们区分三种常见方式：

**① Sequence-mean-then-batch-mean（GRPO 原式）**：

```math
L_{\text{seq-mean}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \ell_{i,t}
```

先对每个回答内部按 $`|o_i|`$ 取平均，再对 $`G`$ 个回答取平均。**有长度偏差**，如 §5.2 所证。

**② Token-mean（全 batch 所有 token 直接平均）**：

```math
L_{\text{token-mean}} = \frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \ell_{i,t}
```

对整个 batch 的所有 response token 直接取平均，除以总 token 数。**无长度偏差**，Dr.GRPO 主张的方式。

**③ DAPO 的 token-level loss**：DAPO（第 6 章）采用类似的 token-mean 聚合，但配合动态采样和其他技术。此处仅作为一种聚合方式提及，完整讨论见第 6 章。

下表对比三者的性质：

| 聚合方式 | 数学形式 | 长度偏差 | 长短混合 batch 行为 |
|---------|---------|---------|-------------------|
| Sequence-mean | $`\frac{1}{G}\sum_i \frac{1}{\|o_i\|}\sum_t \ell_{i,t}`$ | 有 | 短回答每个 token 权重更大 |
| Token-mean | $`\frac{1}{\sum\|o_i\|}\sum_i\sum_t \ell_{i,t}`$ | 无 | 每个 token 权重相同 |
| DAPO token-level | 类似 token-mean + 动态采样 | 无 | 每个 token 权重相同 + 过滤无效 prompt |

<a id="s5-4-2"></a>

### 5.4.2 一个实践中的微妙辨析

Token-mean 会让长回答在 batch 损失中占更大比重，因为它贡献更多 token。在"希望每条回答平等"的直觉下，这似乎又不公平。需要辨析以下两个概念的区别：

- **"每个 token 的梯度权重依赖回答长度"**：这是 sequence-mean 引入的偏差。短回答的每个 token 获得 $`\hat{A}/|o_{\text{short}}|`$ 的权重，长回答只有 $`\hat{A}/|o_{\text{long}}|`$。这种不对称是没有理论依据的。
- **"长回答在 batch 损失中总贡献更大"**：这是 token-mean 的自然结果。一条 200 token 的回答确实包含 200 个决策点，而一条 20 token 的回答只包含 20 个。从策略梯度的角度看，每个 token 对应一个独立的动作选择，长回答确实应该贡献更多的梯度信号。

Token-mean 消除的是前者（不合理的 per-token 权重不对称），保留了后者（长回答因包含更多决策而自然贡献更多梯度）。这正是标准策略梯度定理的形式：$`\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_t \cdot A_t]`$，对所有 token 求和而非按序列归一化后再平均。

<a id="s5-4-3"></a>

### 5.4.3 后续研究

VL-Norm 等后续工作对 loss 聚合方式进行了进一步研究，探讨了在视觉-语言模型等不同模态下的归一化策略。这些工作表明 loss 聚合方式的选择并非一成不变，需要根据具体任务和模型架构进行调整。读者可关注相关文献获取最新进展。

<a id="s5-5"></a>

## 5.5 Dr.GRPO 的完整修正版目标函数

<a id="s5-5-1"></a>

### 5.5.1 合并两处修正

将 §5.2 和 §5.3 的修正合起来，得到 Dr.GRPO（论文中称为 unbiased optimizer）的完整目标函数。相对于 GRPO 原式，两处改动均为"去掉一个归一化项"：

**改动 1**：去掉 $`(1/|o_i|)`$ → 改用全局 token 归一化。

**改动 2**：去掉 $`(1/\text{std})`$ → 只用减均值的优势。

修正后的目标函数为：

```math
L^{\text{Dr.GRPO}}(\theta) = -\frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\left[\min\!\left(r_{i,t}(\theta)\,\tilde{A}_i,\; \text{clip}\!\left(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\right)\tilde{A}_i\right) - \beta\,\hat{D}_{KL,i,t}\right]
```

其中优势简化为：

```math
\tilde{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G)
```

注意 Dr.GRPO 论文在其实验中设 $`\beta = 0`$（去掉 KL 项），理由是 RLVR 使用规则验证器作为奖励，不存在奖励模型的分布偏移问题，因此不需要 KL 约束来防止策略偏离参考模型。但这不是 Dr.GRPO 修正的核心部分——KL 项的取舍是独立的设计决策，与偏差修正正交。在实际使用中，是否保留 KL 取决于具体任务设定。

<a id="s5-5-2"></a>

### 5.5.2 与标准策略梯度的关系

Dr.GRPO 论文在附录 A 中给出了完整的策略梯度推导，证明去掉 $`(1/|o_i|)`$ 和 $`(1/\text{std})`$ 后，所得目标函数恰好对应标准策略梯度定理的形式，其中优势估计使用组均值作为基线（即 REINFORCE with baseline）。具体而言，$`\tilde{A}_i = r_i - \text{mean}(\{r_j\})`$ 与 RLOO（REINFORCE Leave-One-Out）的优势估计仅相差一个常数因子 $`G/(G-1)`$（见第 2 章 §2.4），该因子可以被学习率吸收而不影响优化动力学。

换言之，Dr.GRPO 并非提出了一种新算法，而是**纠正了 GRPO 实现中对标准策略梯度的偏离**，使其回归到理论上正确的形式。

<a id="s5-5-3"></a>

### 5.5.3 实验结论（定性）

Dr.GRPO 论文的主要实证发现包括：

- 在数学推理基准（AIME 2024、MATH500 等）上，Dr.GRPO 相比 vanilla GRPO 提升了推理性能，同时减少了"回答变长但质量不升"的现象。
- Dr.GRPO 有效抑制了错误回答的长度膨胀：vanilla GRPO 训练中错误回答的平均长度持续增长，而 Dr.GRPO 保持了更稳定的长度分布。
- 消融实验（论文 Appendix C）显示，单独去掉长度归一化或单独去掉 std 归一化都能带来改善，两者同时去掉效果最佳。其中长度归一化对回答长度的影响更显著。
- 论文还分析了 base model 的选择（Qwen2.5 vs DeepSeek-V3-Base vs Llama）、prompt 模板的影响、以及数据集覆盖范围与 RL 训练效果的交互作用。

以上为定性总结，具体数值请以原论文为准。代码开源于 github.com/sail-sg/understand-r1-zero。

<a id="s5-6"></a>

## 5.6 其他被讨论的 GRPO 偏差与争议

GRPO 的偏差分析是一个仍在活跃进行的研究领域，Dr.GRPO 只是其中被广泛接受的一支。以下简要提及其他被讨论的问题，以便读者了解研究全景。**以下内容仅为指引，不构成确定性结论，具体见对应论文。**

<a id="s5-6-1"></a>

### 5.6.1 Advantage 的符号-长度耦合

有后续工作（如 Balanced Aggregation 类研究）指出，组内归一化不仅引入了 §5.2 所述的长度偏差，还会让"正优势样本"和"负优势样本"在长度上产生更复杂的耦合偏差。例如，如果正确回答倾向于比错误回答更长（这在推理任务中很常见），那么组内均值中心化后的优势符号就与长度产生了统计相关性，进一步扭曲梯度方向。这类研究提出了更精细的聚合修正方案（具体见对应论文）。

<a id="s5-6-2"></a>

### 5.6.2 二值奖励下组均值中心化的梯度枯竭

有研究（Gradient Starvation in Binary-Reward GRPO 一类）专门分析了二值奖励下组均值中心化的失效模式。当模型在某类题目上的通过率接近 0 或 1 时，组内奖励几乎全同，中心化后优势接近零，导致该类题目的梯度信号枯竭。这与 §5.3.5 提到的"全同奖励零梯度"是同一现象，但该研究方向侧重于分析其在训练动态中的具体表现和修复策略（具体见对应论文）。

<a id="s5-6-3"></a>

### 5.6.3 过校准与过度自信

有工作（Uncalibrated Reasoning 一类）指出 GRPO 训练可能导致模型对随机性结果过度自信：模型在某些题目上的输出概率分布变得过于尖锐，即使正确答案本身具有内在不确定性。这可能与 GRPO 缺乏 token 级信用分配有关——整条序列共享同一优势值，使得模型无法区分"哪些 token 真正重要"，转而将所有置信度集中到最终答案上（具体见对应论文）。

<a id="s5-6-4"></a>

### 5.6.4 On-policy vs Off-policy GRPO

有 ICLR 工作重新审视了 GRPO 在 on-policy 和 off-policy 两种 regime 下的表现。GRPO 原始设计假设 on-policy 采样（$`\theta_{old} = \theta`$），但在多 epoch 更新或异步训练等实际场景中，数据可能变为 off-policy。该工作指出 off-policy GRPO 的重要性比率 $`r_{i,t}`$ 在高维序列空间中可能出现严重的方差膨胀问题，并提出相应的修正建议（具体见对应论文）。这说明 GRPO 的理论基础在非 on-policy 设定下仍有待完善。

<a id="s5-7"></a>

## 5.7 偏差修正的取舍与争议

<a id="s5-7-1"></a>

### 5.7.1 "无偏"与"长度不变"的不可能性

去掉 std 归一化和长度归一化并非没有争议。一个重要事实是：**有后续工作论证了在 outcome reward 下，"无偏"和"长度不变"可能无法同时满足。**

直观理解如下：outcome reward 只在序列末尾给出，所有 token 共享同一奖励信号。如果我们要求优化目标对回答长度不变（即不因长度差异而产生偏置），就需要某种形式的长度归一化；但如果我们要求优化目标是标准策略梯度的无偏估计，就不应该引入额外的归一化。这两个要求在 outcome reward 的框架下存在内在张力。

Dr.GRPO 选择了"无偏"而放弃了"长度不变"。在实践中，这意味着长回答会在 batch 损失中贡献更多梯度（因为其包含更多 token），但这种贡献是与决策点数量成正比的"自然"贡献，而非 $`(1/|o_i|)`$ 引入的"扭曲"贡献。然而，在某些任务设定下（如回答长度差异极大、且长回答并不一定包含更多有效推理步骤时），这种"自然"贡献本身也可能带来问题。

<a id="s5-7-2"></a>

### 5.7.2 实践建议

基于当前的研究和实践经验，给出以下建议：

1. **长度偏差修正（去掉 $`1/|o_i|`$）通常有正收益**，尤其在长度差异大的任务（如允许长短不一的推理链）上。这是因为"对就简短、错就啰嗦"的偏置在几乎所有推理任务中都是有害的。
2. **std 归一化的去除效果更依赖数据难度分布**。如果训练数据的难度分布较为均匀（大部分题目的通过率在 0.2~0.8 之间），去掉 std 的收益可能不大；如果数据中包含大量极端难度的题目（通过率接近 0 或 1），去掉 std 可以避免边缘题主导梯度。
3. **不要盲从任何单一修正**。建议在自己的任务上做消融实验：分别测试 vanilla GRPO、去掉长度归一化、去掉 std 归一化、两者都去掉四种配置，观察训练动态和最终性能。
4. **关注后续研究**。GRPO 的偏差分析仍在快速发展中，本章所述内容反映的是截至 Dr.GRPO 论文发表时的认知。新的理论和实证结果可能进一步修正或补充现有理解。

<a id="s5-8"></a>

## 5.8 本章小结与自测

<a id="h167"></a>

### 要点总结

1. **GRPO 的偏差是优化目标层面的扭曲**，不是梯度估计的统计偏差。即使有无穷采样，GRPO 仍会收敛到非最优策略。
2. **长度偏差源于 $`(1/|o_i|)`$ 归一化**，导致"对就简短、错就啰嗦"的系统性偏置：正优势时短回答每 token 权重更大，负优势时长回答每 token 惩罚被摊薄。
3. **难度偏差源于 $`(1/\text{std})`$ 归一化**，导致边缘难度题（$`p \to 0`$ 或 $`p \to 1`$）被放大、中等难度题（$`p \approx 0.5`$）被压制，与第 4 章的信号强度分析方向相反。
4. **Dr.GRPO 的修正极其简单**：去掉 $`(1/|o_i|)`$ 和 $`(1/\text{std})`$，使目标函数回归标准策略梯度形式。两处改动都是"去掉一个归一化项"。
5. **长度偏差也广泛存在于开源 PPO 实现中**，是公式-实现对齐的历史遗留问题，不限于 GRPO。
6. **偏差修正并非终点**："无偏"与"长度不变"在 outcome reward 下可能不可兼得；去掉 std 不能解决全同奖励的信号枯竭问题（需 DAPO 动态采样）；GRPO 的偏差分析仍是活跃研究领域。

<a id="h168"></a>

### 自测题

**题 1**：设同一 prompt 下 GRPO 采样得到两个正确回答 $`o_a`$（$`|o_a|=50`$ token）和 $`o_b`$（$`|o_b|=200`$ token），它们的奖励相同（$`r_a = r_b = 1`$），组内均值 $`\bar{r} = 0.5`$，std $`= 0.5`$。计算两个回答每个 token 获得的梯度权重之比（$`o_a`$ 比 $`o_b`$）。如果使用 Dr.GRPO 修正，该比值变为多少？

参考答案：GRPO 下，$`\hat{A}_a = \hat{A}_b = (1-0.5)/0.5 = 1.0`$。$`o_a`$ 每 token 权重为 $`1.0/50 = 0.02`$，$`o_b`$ 为 $`1.0/200 = 0.005`$，比值为 $`0.02/0.005 = 4`$。Dr.GRPO 下，$`\tilde{A}_a = \tilde{A}_b = 1-0.5 = 0.5`$，每 token 权重均为 $`0.5/L_{\max}`$（或 $`0.5/\sum|o_i|`$），比值为 1。修正消除了长度导致的权重不对称。

**题 2**：解释为什么 GRPO 的长度偏差会导致"对就简短、错就啰嗦"的净效应。具体要求：分别说明正优势和负优势情形下 $`(1/|o_i|)`$ 的作用方向，并解释为什么两者合起来构成一种与推理质量无关的长度偏置。

参考答案：正优势时（$`\hat{A}>0`$，正确回答），$`(1/|o_i|)`$ 使短回答每 token 获得更大的正向更新，策略偏好简洁的正确表达。负优势时（$`\hat{A}<0`$，错误回答），$`(1/|o_i|)`$ 使长回答每 token 受到的负向更新被摊薄，策略对冗长的错误表达惩罚不足。合起来，无论回答对错，$`(1/|o_i|)`$ 都在推动"正确→短、错误→长"的方向，而这个方向完全由回答长度决定，与推理质量（是否正确、推理是否有效）无关。这是一种纯粹的结构偏置。

**题 3**：在二值奖励下，设 $`G=8`$，某 prompt 的组内通过率 $`k/G = 0.125`$（即 8 个回答中恰好 1 个正确）。计算该正确答案的 GRPO 优势 $`\hat{A}_{\text{correct}}`$ 和 Dr.GRPO 优势 $`\tilde{A}_{\text{correct}}`$。两者的比值是多少？这个比值说明了什么？

参考答案：$`\text{mean} = 1/8 = 0.125`$，$`\text{std} = \sqrt{0.125 \times 0.875} \approx 0.3307`$。GRPO 优势 $`\hat{A} = (1-0.125)/0.3307 \approx 2.646`$。Dr.GRPO 优势 $`\tilde{A} = 1-0.125 = 0.875`$。比值 $`\approx 2.646/0.875 \approx 3.02`$。这说明在极低通过率下，GRPO 的 std 归一化将优势放大了约 3 倍，使得这道"碰巧做对一题"的边缘题获得了远超其自然奖励尺度的梯度权重。

**题 4**：有人说"去掉 std 归一化就能解决 GRPO 在二值奖励下的信号枯竭问题"。这个说法是否正确？为什么？

参考答案：不正确。去掉 std 归一化只改变了"部分对部分错时各题的相对权重"，不改变"全同奖励零梯度"的事实。当某 prompt 全对（$`k=G`$）或全错（$`k=0`$）时，$`\text{mean}`$ 等于所有 $`r_i`$，$`\tilde{A}_i = r_i - \text{mean} = 0`$，梯度仍然为零。信号枯竭问题的解决需要 DAPO 的动态采样（第 6 章）或其他机制来保证每个 prompt 都有有效的梯度信号。

**题 5**：Dr.GRPO 论文指出长度偏差也存在于开源 PPO 实现中。请从数学上解释：PPO 的理论目标函数（对所有 token 求和）与常见的有偏实现（先按 $`|o_i|`$ 平均再按 batch 平均）之间的差异，恰好对应 GRPO 中的哪个归一化项？为什么预训练阶段的 `loss.mean(-1)` 不会引入同样的问题？

参考答案：差异恰好对应 $`(1/|o_i|)`$ 归一化。PPO 理论形式为 $`\sum_i \sum_t \ell_{i,t}`$（或除以总 token 数），有偏实现为 $`\sum_i (1/|o_i|) \sum_t \ell_{i,t}`$（再除以 $`G`$），两者之差正是每个回答被其自身长度归一化。预训练阶段 `loss.mean(-1)` 不会引入同样问题，是因为预训练的上下文长度是固定的（packing 后每个序列长度相同），$`(1/|o_i|)`$ 退化为常数 $`(1/L)`$，对所有样本一视同仁，不产生相对权重的扭曲。而在 RL 微调阶段，不同回答的长度 $`|o_i|`$ 是变量，$`(1/|o_i|)`$ 不再是常数，因此引入了长度偏差。

<a id="chapter-6"></a>

# 第 6 章 DAPO：面向长链推理的四项关键改进

<a id="s6-1"></a>

## 6.1 DAPO 的定位与整体思路

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，解耦裁剪与动态采样策略优化）由字节跳动 Seed、清华 AIR 与港大等团队于 2025 年提出（Yu et al., 2025, arXiv:2503.14476）。它不是一种替代 GRPO 的新算法，而是**在 GRPO 框架内针对长链推理（long Chain-of-Thought, long-CoT）RL 训练所做的四项精确修正**。DAPO 完整保留了 GRPO 的核心结构——组内基线、去 critic、逐 token 比率——但针对实践中暴露的四个具体病症各开一剂药。

下表给出四组"病症↔技术"的对应关系，也是本章的组织主线：

| 病症 | 数学根源 | DAPO 技术 | 核心改动 |
|------|---------|-----------|---------|
| 熵崩塌（entropy collapse） | 对称 clip 限制低概率好 token 的概率增长上界，策略分布过早收敛至 one-hot | Clip-Higher（解耦裁剪） | $`\varepsilon_{high} > \varepsilon_{low}`$，放宽上界 |
| 无效样本浪费算力 | 组内奖励全对/全错时 $`\hat{A}_i \equiv 0`$，梯度为零但采样成本已付 | Dynamic Sampling（动态采样） | 过采样并过滤全同奖励 prompt |
| 长短回答梯度不公 | 序列内 $`(1/\|o_i\|)`$ 归一化使长回答每 token 梯度被稀释 | Token-Level Loss（token 级损失） | 改为全 batch token 直接平均 |
| 超长截断噪声 | 截断回答被 verifier 误判为错误，产生虚假负信号；或故意超长骗分 | Overlong Reward Shaping（超长奖励塑形） | 软惩罚 + 截断过滤 |

此外，DAPO **明确移除了 KL 约束项**，理由将在 6.6 节详述。

**核心成绩**：基于 Qwen2.5-32B base 模型，DAPO 在 AIME 2024 基准上以 avg@32 达到 50 分，超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分，且仅使用约一半的训练步数。这一结果证明了在 GRPO 框架内进行针对性工程精修的有效性，也表明 R1-Zero 级别的 RLVR 训练并非不可复现。项目页为 dapo-sia.github.io，代码基于 verl 框架开源。

以下四节逐一展开每项技术的数学动机与推导。

<a id="s6-2"></a>

## 6.2 Clip-Higher：解耦上下裁剪阈值

<a id="s6-2-1"></a>

### 6.2.1 熵崩塌的定义与危害

承接 PPO 文档第 6 章，策略熵 $`H(\pi_\theta)`$ 是衡量探索性的核心指标。对于状态 $`s`$ 下的离散动作空间 $`\mathcal{V}`$：

```math
H(\pi_\theta(\cdot|s)) = -\sum_{a \in \mathcal{V}} \pi_\theta(a|s) \log \pi_\theta(a|s).
```

当 $`\pi_\theta(\cdot|s)`$ 趋近 one-hot 分布（即某个动作的概率趋近 1，其余趋近 0）时，$`H \to 0`$。**熵崩塌（entropy collapse）** 指的就是训练过程中策略熵过快下降到极低水平的现象。其危害在于：

1. **探索丧失**：策略几乎只生成一个确定性的输出，无法发现可能更优的替代路径。
2. **性能停滞**：一旦陷入局部最优的确定性策略，后续训练无法改善。
3. **多样性退化**：对于需要多种解题策略的数学推理任务，单一模式意味着覆盖率的天花板。

在 long-CoT RL 训练中，熵崩塌尤其致命：推理路径的搜索空间极大，过早收敛意味着模型放弃了绝大部分潜在的推理链路。

<a id="s6-2-2"></a>

### 6.2.2 对称 clip 如何加速熵崩塌

PPO-Clip 和标准 GRPO 均使用对称裁剪阈值 $`\varepsilon_{low} = \varepsilon_{high} = \varepsilon`$（常用 $`\varepsilon = 0.2`$），即：

```math
\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) = \max(1-\varepsilon, \min(1+\varepsilon, r_t)).
```

承接 PPO 文档第 6 章的六情形分析，当 $`\hat{A} > 0`$ 且 $`r_t > 1+\varepsilon`$ 时，$`\min`$ 操作选择被裁剪的分支，梯度为零——策略不再继续增大该 token 的概率。这个机制在防止策略更新过大方面功不可没，但它对**低概率好 token** 产生了系统性的压制效应。推理链条如下：

1. 考虑一个当前概率很低的 token $`a^*`$，即 $`\pi_\theta(a^*|s) \approx \delta`$，其中 $`\delta \ll 1`$。假设它是好 token（$`\hat{A} > 0`$），我们希望增大概率。
2. 重要性比率 $`r_t = \pi_\theta(a^*|s) / \pi_{\theta_{old}}(a^*|s)`$。由于 $`\pi_\theta(a^*|s)`$ 本身很小，即使绝对概率翻倍（从 $`\delta`$ 到 $`2\delta`$），$`r_t`$ 也只是从 1 变为 2——而 $`1+\varepsilon = 1.2`$ 的上界远在 2 之下。
3. 这意味着：低概率 token 只需极小的相对增幅就会触发 $`r_t > 1+\varepsilon`$ 的截断条件，梯度立即归零。
4. 相比之下，一个已经高概率的 token（$`\pi_\theta \approx 0.9`$），要从 0.9 涨到触发截断，需要 $`r_t > 1.2`$，即概率需涨到 $`0.9 \times 1.2 = 1.08`$——这不可能发生（概率上界为 1），所以高概率 token 实际上很少被上界截断。

**结论**：对称 clip 的上界对低概率 token 构成了不成比例的限制。低概率好 token 难以被"捞起来"，高概率 token 则不受影响。长期累积效应是：概率质量持续向已有的高概率 token 集中，低概率区域被系统性忽视，分布越来越尖锐，熵持续下降直至崩塌。

<a id="s6-2-3"></a>

### 6.2.3 Clip-Higher 的解法

DAPO 将上下裁剪阈值解耦（Decoupled Clip），定义非对称裁剪算子：

```math
\text{clip}_{asym}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high}) = \max(1-\varepsilon_{low}, \min(1+\varepsilon_{high}, r_t)),
```

其中 $`\varepsilon_{high} > \varepsilon_{low}`$。DAPO 的典型取值为 $`\varepsilon_{low} = 0.2`$，$`\varepsilon_{high} = 0.28`$。

修改后的单 token 目标为：

```math
\ell_{i,t}^{clip-higher} = \min\!\left(r_{i,t}\,\hat{A}_i,\; \text{clip}_{asym}(r_{i,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high})\,\hat{A}_i\right).
```

用 PPO 文档六情形的语言重新分析：

- **$`\hat{A} > 0`$ 的情形**：梯度归零的触发点从 $`r_t > 1+\varepsilon_{low}`$ 推迟到 $`r_t > 1+\varepsilon_{high}`$。回到上面的例子，低概率 token 从 $`\delta`$ 出发，现在允许 $`r_t`$ 涨到 $`1.28`$ 才被截断，而非 $`1.20`$。这多出的 $`0.08`$ 的"上涨空间"看似不大，但对于 $`\delta`$ 很小的 token，它意味着允许的绝对概率增幅从 $`0.2\delta`$ 增加到 $`0.28\delta`$——相对提升了 40%。在数百上千个训练步中，这个差异足以让原本被压制的低概率好 token 逐步积累概率质量。
- **$`\hat{A} < 0`$ 的情形**：下界仍为 $`1-\varepsilon_{low} = 0.8`$，控制"减小概率"的幅度不变。策略不会因上界放宽而在惩罚方向也变得激进。

<a id="s6-2-4"></a>

### 6.2.4 为什么下界不一起放宽

非对称设计的直觉是：**鼓励探索，但不鼓励激进遗忘**。

- 上界控制的是"增大概率"的自由度。放宽上界 = 允许策略更快地拥抱新发现的好行为 = 促进探索。
- 下界控制的是"减小概率"的自由度。保持较紧的下界 = 不让策略过快抛弃已有好行为 = 维持稳定性。

如果同时放宽下界（例如 $`\varepsilon_{low} = 0.28`$），策略可能在一步之内将某个 token 的概率大幅削减，导致已学到的有用知识被快速遗忘。这在 long-CoT 场景中尤其危险：推理链中的某些中间步骤可能暂时看起来不重要，但在后续推理中被证明是关键前提；过快的遗忘会破坏这种长程依赖的学习。

因此，$`\varepsilon_{low} = 0.2, \varepsilon_{high} = 0.28`$ 的非对称设定是一个经过权衡的设计选择，而非随意的参数调整。

<a id="s6-2-5"></a>

### 6.2.5 对诊断指标的影响

承接 PPO 文档第 6 章关于 clip_fraction 的讨论。clip_fraction 是被裁剪样本占总样本的比例，健康范围通常在 0.1~0.3。引入 Clip-Higher 后：

- **上侧 clip_fraction 会下降**：因为上界从 $`1+\varepsilon_{low}`$ 放宽到 $`1+\varepsilon_{high}`$，更少的正优势样本触及上界。
- **下侧 clip_fraction 基本不变**：下界未改。
- **总 clip_fraction 可能略有下降**：这是预期行为，不应被视为"clip 失效"。

监控熵是否健康的建议：
1. 跟踪每个训练步的策略熵 $`H(\pi_\theta)`$，确认其不会单调下降至接近零。
2. 观察上侧 clip_fraction 是否在合理范围内波动；若持续为零，说明 $`\varepsilon_{high}`$ 可能过大，失去了保护作用。
3. 结合评估集性能：若训练 loss 持续下降但评估指标停滞或退化，可能是熵崩塌的信号。

<a id="s6-3"></a>

## 6.3 Dynamic Sampling：动态采样过滤无效 prompt

<a id="s6-3-1"></a>

### 6.3.1 动机：零梯度样本的算力浪费

承接第 2 章和第 4 章的核心结论：当同一 prompt $`q`$ 的 $`G`$ 次采样获得的奖励全部相同（全对或全错）时，组内标准差 $`\hat{\sigma} = 0`$，归一化优势 $`\hat{A}_i \equiv 0`$，该 prompt 对梯度的贡献为零。然而，生成这 $`G`$ 个回答的推理算力已经消耗了。

第 4 章给出了二值奖励下全同奖励概率的精确表达式：

```math
P(\text{全同}) = p^G + (1-p)^G,
```

其中 $`p`$ 是模型在该 prompt 上的正确率。当 $`p \to 0`$ 或 $`p \to 1`$ 时，$`P(\text{全同}) \to 1`$。在训练初期（$`p`$ 普遍较低）和训练后期（简单题 $`p \to 1`$），大量 prompt 的全同概率很高，意味着相当比例的算力被浪费在零梯度样本上。

<a id="s6-3-2"></a>

### 6.3.2 DAPO 的动态采样机制

DAPO 的解法是**过采样（over-sample）并过滤**。具体流程：

1. 从 prompt 池中采样 $`N_{candidate} > N_{batch}`$ 个候选 prompt。
2. 对每个候选 prompt 生成 $`G`$ 个回答，计算组内准确率 $`\text{acc}_j = \frac{1}{G}\sum_{i=1}^G r_{j,i}`$。
3. **丢弃**所有 $`\text{acc}_j = 0`$ 或 $`\text{acc}_j = 1`$ 的 prompt（全错或全对）。
4. **保留**满足 $`0 < \text{acc}_j < 1`$ 的 prompt，直到凑满目标 batch 大小 $`N_{batch}`$。
5. 若有效 prompt 不足，继续采样新的候选并重复上述过程。

过滤条件的数学表达为：

```math
\mathcal{B}_{valid} = \left\{ j : 0 < \frac{1}{G}\sum_{i=1}^G r_{j,i} < 1 \right\}, \quad |\mathcal{B}_{valid}| = N_{batch}.
```

<a id="s6-3-3"></a>

### 6.3.3 为什么这有效：隐式课程学习

动态采样的效果不仅是"省算力"，更深层的作用是**自适应地将训练集中在当前能力边界上**。

随着训练推进，模型在简单题上的正确率 $`p \to 1`$，这些题被动态采样自动过滤；困难题的正确率 $`p \to 0`$，同样被过滤。最终留在 batch 中的主要是 $`p \approx 0.5`$ 附近的"中等难度"题目——恰好是第 4 章指出的信号最强区域（$`p \approx 0.5`$ 时 $`P(\text{全同})`$ 最小，组内方差最大）。

这构成了一种**隐式的、自适应的课程学习（implicit curriculum）**：无需人工设计难度调度，动态采样自然地将训练焦点维持在模型的"学习前沿"（learning frontier）。与传统课程学习不同，这里的难度评估完全由模型自身的采样表现决定，不需要外部难度标签。

从梯度信噪比的角度看：每个保留下来的 prompt 都保证贡献非零梯度，batch 的有效信息密度大幅提升。在固定算力预算下，这等价于增加了有效训练步数。

<a id="s6-3-4"></a>

### 6.3.4 代价与注意点

动态采样并非免费午餐：

1. **过采样的推理开销**：为了凑满一个 batch，可能需要生成远超 $`N_{batch} \times G`$ 个回答。在训练后期，当大部分简单题被过滤时，过采样倍率可能显著增加。
2. **有效分布偏移**：过滤改变了 prompt 的实际训练分布——简单题和极难题被系统性排除，batch 偏向中等难度。这在大多数 RLVR 场景下是有益的，但在以下情况可能有害：
   - 任务要求模型巩固基础能力（简单题的反复练习有助于稳定性和鲁棒性）。
   - 数据集本身难度分布狭窄，过滤后可用样本过少。
   - 需要覆盖极端难度的测试场景，但训练中从未见过这类题。
3. **实现复杂度**：需要异步的采样-过滤-填充流水线，增加了系统工程难度（详见第 8 章）。

<a id="s6-3-5"></a>

### 6.3.5 伪代码

```python
def dynamic_sampling(prompt_pool, model, G, N_batch):
    """动态采样：过滤全对/全错 prompt，保留有区分度的样本"""
    valid_prompts = []
    while len(valid_prompts) < N_batch:
        # 过采样一批候选 prompt
        candidates = sample_candidates(prompt_pool, size=N_batch * 2)
        for q in candidates:
            if len(valid_prompts) >= N_batch:
                break
            # 对每个 prompt 采样 G 个回答
            completions = [model.generate(q) for _ in range(G)]
            rewards = [verifier(q, o) for o in completions]
            acc = sum(rewards) / G
            # 仅保留组内既有对又有错的 prompt
            if 0 < acc < 1:
                valid_prompts.append((q, completions, rewards))
    return valid_prompts[:N_batch]
```

<a id="s6-4"></a>

## 6.4 Token-Level Policy Gradient Loss

<a id="s6-4-1"></a>

### 6.4.1 动机：序列内平均的长度偏差

承接第 5 章的结论，GRPO 原式对每个回答先做序列内平均 $`(1/|o_i|)\sum_t`$，再对组内做平均 $`(1/G)\sum_i`$：

```math
L^{GRPO} = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \ell_{i,t},
```

其中 $`\ell_{i,t}`$ 是 token $`(i,t)`$ 的 clip 损失（含 KL 项或不包含）。在这个双层平均结构下，每条回答无论长短，对总损失的贡献权重都是 $`1/G`$。但长回答的损失被分摊到了更多 token 上，每个 token 获得的梯度幅度与 $`1/|o_i|`$ 成正比。

在 long-CoT 场景中，回答长度差异可达数十倍（几百到上万 token）。一条 5000 token 的推理链中，每个关键推理步骤的梯度信号被稀释为短回答的 $`1/50`$ 甚至更低。这意味着模型更难从长回答中学到精细的推理模式。

<a id="s6-4-2"></a>

### 6.4.2 DAPO 的 token-level 聚合

DAPO 将损失聚合方式改为对整个 batch 的所有 token 直接平均：

```math
L^{DAPO-token} = \frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G \sum_{t=1}^{|o_i|} \ell_{i,t}.
```

两种形式的本质区别在于归一化因子：

| 聚合方式 | 归一化因子 | 每条回答的权重 | 每 token 权重 |
|---------|-----------|--------------|-------------|
| GRPO（sequence-level） | $`G \cdot |o_i|`$（双重平均） | $`1/G`$（等权） | $`1/(G \cdot \|o_i\|)`$ |
| DAPO（token-level） | $`\sum_i \|o_i\|`$（全局平均） | $`\|o_i\| / \sum_j \|o_j\|`$（按长度加权） | $`1 / \sum_j \|o_j\|`$（均匀） |

在 token-level 聚合下，每个 token 的权重都是 $`1/(\text{总 token 数})`$，与其所属回答的长度无关。长回答因为 token 数量多，在总损失中自然占据更大比重——这是合理的，因为长回答确实包含更多需要学习的决策点。

<a id="s6-4-3"></a>

### 6.4.3 与 Dr.GRPO 的关系

值得注意的是，Dr.GRPO（见第 5 章）独立地提出了几乎相同的修正：去掉 $`(1/|o_i|)`$ 的序列内归一化，改用 token-mean 聚合来消除长度偏差。DAPO 与 Dr.GRPO 在这一点上殊途同归、几乎同时提出，从不同角度论证了同一结论：**在 long-CoT 场景下，序列内平均是不合适的聚合方式，token-level 平均才是正确的选择**。

两者的微妙差异在于：Dr.GRPO 的出发点纯粹是消除长度偏差的统计论证；DAPO 则将其作为四项改进之一，与其他三项协同工作。在实际效果上，token-level loss 对 long-CoT 训练的贡献是基础性的——它确保了其他三项改进（clip-higher、动态采样、超长塑形）的梯度信号不会被长度归一化所稀释。

<a id="s6-4-4"></a>

### 6.4.4 对 long-CoT 的关键意义

在长链推理中，一个完整的推理过程可能包含数十个逻辑步骤，分布在数千个 token 中。其中真正关键的"aha moment"可能只有几个 token（例如选择某种特定的分解策略、引入一个巧妙的辅助构造）。在序列内平均下，这几个关键 token 的信号被除以 $`|o_i|`$（可能高达数千），几乎淹没在噪声中。Token-level loss 保持了每个 token 的完整信号强度，使得这些稀疏但关键的决策点能够被有效学习。

<a id="s6-5"></a>

## 6.5 Overlong Reward Shaping：超长奖励塑形

<a id="s6-5-1"></a>

### 6.5.1 动机：截断带来的奖励噪声

Long-CoT 训练中，模型生成的推理链可能非常长，触发预设的最大生成长度 $`L_{max}`$ 而被截断（truncation）。这带来两个问题：

1. **误判噪声**：被截断的回答往往没有写出最终答案（答案在推理链末尾），verifier 无法提取到有效答案，将其判定为错误（$`r=0`$）。然而，这个回答的推理过程可能是完全正确的——只是没写完。这给策略施加了一个**虚假的负信号**：正确的推理被惩罚，仅仅因为它太长。
2. **Reward hacking**：承接第 4 章，一种已知的 hacking 路径是"故意生成超长输出触发截断"。在某些实现中，截断的回答可能因为解析器异常而意外获得高分；或者模型学会了用冗长输出来"拖延"以避免暴露错误答案。

这两个问题的共同根源是：**硬截断是一个不可微的、突变的信号**。模型在长度达到 $`L_{max}-1`$ 时不受任何惩罚，在 $`L_{max}`$ 时突然被截断并可能被判错。这种阶跃式的反馈难以被策略梯度有效利用。

<a id="s6-5-2"></a>

### 6.5.2 DAPO 的两部分解法

DAPO 的超长奖励塑形包含两个互补的组件：

**① Overlong Filtering（超长过滤）**：对被截断的回答（$`|o_i| = L_{max}`$），其损失不参与梯度计算。形式上，引入掩码 $`m_i`$：

```math
m_i = \begin{cases} 0 & \text{if } |o_i| = L_{max} \quad (\text{被截断}) \\ 1 & \text{otherwise} \end{cases}
```

在 token-level loss 中，只有 $`m_i = 1`$ 的回答参与聚合。这直接消除了"因截断而误判"的噪声源。注意这与动态采样不同：动态采样过滤的是全对/全错的 prompt；超长过滤针对的是单个被截断的回答，即使同一 prompt 的其他回答未被截断且有效。

**② Soft Overlong Punishment（软超长惩罚）**：在硬上限之前设置一个缓冲区 $`[L_{max} - L_{cache}, L_{max}]`$，在此区间内施加线性递增的长度惩罚：

```math
R_{length}(o) = \begin{cases} 
0 & |o| \le L_{max} - L_{cache} \\
\displaystyle\frac{(L_{max} - L_{cache}) - |o|}{L_{cache}} & L_{max} - L_{cache} < |o| \le L_{max} \\
-1 & |o| > L_{max}
\end{cases}
```

其中 $`L_{cache}`$ 是缓冲区宽度（DAPO 论文中的超参）。最终奖励为：

```math
R(o) = R_{verifier}(o) + R_{length}(o).
```

<a id="s6-5-3"></a>

### 6.5.3 软惩罚的数学动机

软惩罚的核心优势在于**平滑性**。对比两种信号：

- **硬截断**：$`R(|o|)`$ 在 $`L_{max}`$ 处有一个阶跃间断。策略梯度 $`\nabla_\theta \mathbb{E}[R]`$ 对这个间断点的估计方差极大（因为采样落在间断点两侧时得到截然不同的奖励值），且间断点本身不可微，无法提供"应该提前多少停止"的方向信息。
- **软惩罚**：$`R_{length}(|o|)`$ 在缓冲区内是关于 $`|o|`$ 的连续线性函数。虽然 $`|o|`$ 本身关于 $`\theta`$ 不可微（生成长度是离散随机变量），但软惩罚通过改变缓冲区内每个 token 位置的奖励值，间接地为策略提供了一个"越接近上限、奖励越低"的平滑梯度方向。模型可以从中学到"应该在缓冲区开始之前就完成推理"。

从优化景观（optimization landscape）的角度看：硬截断创造了一个悬崖边缘，策略要么安全地在边缘内侧、要么掉下去；软惩罚则将悬崖改造为一个斜坡，策略可以沿着斜坡逐步学到合适的长度。

<a id="s6-5-4"></a>

### 6.5.4 堵住 reward hacking 路径

承接第 4 章提到的"故意超长触发 truncation 骗分"hacking 路径：

- **超长过滤**使截断回答完全不参与训练，消除了"截断意外得高分"的可能性。
- **软惩罚**使接近上限的回答即使未被截断也受到惩罚，模型无法通过"刚好写到上限"来获得无惩罚的高分。

两者结合，使得"故意写长"在任何情况下都不会带来收益，从根本上堵住了这条 hacking 路径。

<a id="s6-6"></a>

## 6.6 DAPO 移除 KL 项的决策

GRPO 原式中包含 KL 约束项 $`-\beta\,\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$（见第 3 章）。DAPO 明确将其移除，即设 $`\beta = 0`$。这一决策基于以下三方面考量：

**① 长链推理需要大幅偏离初始分布**。KL 约束的本质是将策略限制在参考模型 $`\pi_{ref}`$（通常是 SFT 模型）的邻域内。对于对话对齐等任务，这是必要的——我们不希望模型偏离人类偏好太远。但对于数学推理等 long-CoT 任务，SFT 模型的推理能力有限，RL 训练的目标恰恰是**探索出全新的、SFT 阶段未见过的推理路径**。这些"aha"行为往往位于远离 $`\pi_{ref}`$ 的参数空间中。KL 约束会将策略拉回 $`\pi_{ref}`$ 附近，扼杀这种探索。

数学上，KL 项的梯度 $`\nabla_\theta (-\beta D_{KL})`$ 始终指向减小 $`\pi_\theta`$ 与 $`\pi_{ref}`$ 差异的方向。当 $`\beta`$ 较大时，这个力可能主导策略更新，使模型不敢偏离 SFT 行为。在 RLVR 场景下，verifier 提供了明确的对错信号，不需要 KL 来充当"质量护栏"。

**② RLVR 的 reward hacking 风险低于 RM 场景**。KL 在 RLHF-PPO 中的另一个作用是防止策略过度优化奖励模型（reward hacking）。但在 RLVR 中，奖励来自确定性 verifier，hacking 的空间远小于连续可微的 RM（见第 4 章）。DAPO 的四项改进（尤其是超长奖励塑形）进一步降低了剩余 hacking 路径的风险，使 KL 的"防 hacking"功能不再是必需的。

**③ 简化实现与节省显存**。去掉 KL 项意味着不再需要维护参考模型 $`\pi_{ref}`$ 的前向传播。承接第 1 章的分析，GRPO 的显存负担包括 policy + reference + rollout 缓存；去掉 reference 后，显存需求进一步降低，这对 32B 及以上规模模型的训练至关重要。

**风险提示**：完全移除 KL 是 DAPO 针对 long-CoT 数学推理的特定选择，**不是普适最优**。在以下场景中，KL 通常仍需保留：
- RLHF 对齐任务（需要锚定人类偏好）。
- 非推理类生成任务（如创意写作、对话），策略过度偏离可能导致语言质量退化、输出崩坏。
- 训练数据覆盖面窄、泛化风险高的场景。

DAPO 论文的实验也证实了在数学推理任务上去掉 KL 不会导致语言质量明显退化，但这不能推广到所有任务类型。

<a id="s6-7"></a>

## 6.7 DAPO 完整目标函数与算法

<a id="s6-7-1"></a>

### 6.7.1 完整目标函数

综合四项改进与 KL 移除，DAPO 的完整目标函数为：

```math
L^{DAPO} = -\mathbb{E}\left[\frac{1}{\sum_{i \in \mathcal{B}_{valid}} |o_i|}\sum_{i \in \mathcal{B}_{valid}} m_i \sum_{t=1}^{|o_i|} \min\!\left(r_{i,t}\,\hat{A}_i,\; \text{clip}_{asym}(r_{i,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high})\,\hat{A}_i\right)\right]
```

其中：
- $`\mathcal{B}_{valid}`$ 是动态采样过滤后的有效 prompt 集合（$`0 < \text{acc} < 1`$）。
- $`m_i`$ 是超长过滤掩码（$`m_i = 0`$ 当 $`|o_i| = L_{max}`$）。
- $`\text{clip}_{asym}`$ 是非对称裁剪，$`\varepsilon_{low} = 0.2, \varepsilon_{high} = 0.28`$。
- $`\hat{A}_i = (r_i - \bar{r}) / \hat{\sigma}`$ 仍使用组内归一化优势。关于是否除以 $`\hat{\sigma}`$：DAPO 论文保留了减均值操作；对于 std 归一化，不同实现版本存在差异——部分实现保留 $`\hat{\sigma}`$ 归一化，部分实现（受 Dr.GRPO 影响）去掉 std 归一化以避免第 2 章所述的难度偏差。读者在使用具体代码时应核对实际配置。
- $`r_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})`$ 为逐 token 重要性比率。
- 奖励 $`r_i = R_{verifier}(o_i) + R_{length}(o_i)`$ 包含了软超长惩罚。
- **无 KL 项**。

取负号是因为优化框架最小化损失，而上式内部的期望是最大化目标。

<a id="s6-7-2"></a>

### 6.7.2 算法伪代码

以下伪代码在 GRPO 基础上标注了 DAPO 的四处改动：

```python
def DAPO_training_step(prompt_pool, model, G, N_batch, eps_low, eps_high, L_max, L_cache):
    # ===== 改动 1: Dynamic Sampling =====
    # 过采样并过滤全对/全错 prompt（替代 GRPO 的直接采样）
    batch = dynamic_sampling(prompt_pool, model, G, N_batch)

    total_loss = 0.0
    total_tokens = 0

    for (q, completions, rewards) in batch:
        # 计算组内优势（保留减均值；std 归一化视实现而定）
        mean_r = mean(rewards)
        std_r = std(rewards)  # 可选：部分实现去掉此行
        advantages = [(r - mean_r) / std_r for r in rewards]  # 或 (r - mean_r)

        for i, (o, A) in enumerate(zip(completions, advantages)):
            # ===== 改动 4: Overlong Filtering =====
            # 被截断的回答不参与损失计算
            if len(o) == L_max:
                continue

            # 计算软超长惩罚并叠加到奖励/优势中
            r_length = soft_length_penalty(len(o), L_max, L_cache)
            # 注意：r_length 影响的是原始奖励，进而影响优势计算
            # 此处简化展示；实际实现中应在优势计算前叠加

            for t in range(len(o)):
                r_t = model.prob(o[t] | q, o[:t]) / old_model.prob(o[t] | q, o[:t])

                # ===== 改动 2: Clip-Higher =====
                # 非对称裁剪（替代 GRPO 的对称裁剪）
                clipped_r = clip(r_t, 1 - eps_low, 1 + eps_high)
                token_loss = min(r_t * A, clipped_r * A)

                # ===== 改动 3: Token-Level Loss =====
                # 累加到全局 token 计数（替代序列内平均）
                total_loss += token_loss
                total_tokens += 1

    # Token-level 归一化
    loss = -total_loss / total_tokens  # 无 KL 项
    loss.backward()
    model.update()
```

<a id="s6-7-3"></a>

### 6.7.3 DAPO 相对 GRPO 的改动总结

| 改动 | GRPO 原版 | DAPO | 解决的病症 |
|------|----------|------|-----------|
| 裁剪阈值 | 对称 $`\varepsilon = 0.2`$ | 非对称 $`\varepsilon_{low}=0.2, \varepsilon_{high}=0.28`$ | 熵崩塌 |
| 采样策略 | 直接采样固定 batch | 过采样 + 过滤全同奖励 prompt | 无效样本浪费 |
| 损失聚合 | 序列内平均再组内平均 | 全 batch token 直接平均 | 长度偏差 |
| 超长处理 | 无（硬截断） | 软惩罚 + 截断过滤 | 截断噪声与 hacking |
| KL 约束 | $`\beta > 0`$（通常 0.001~0.04） | $`\beta = 0`$（移除） | 促进长链推理探索 |

<a id="s6-8"></a>

## 6.8 DAPO 的工程系统与可复现性

DAPO 的贡献不仅在于算法层面的四项改进，还在于**完整开源了一套大规模 RLVR 训练系统**。该系统基于 verl 框架构建，涵盖了从 rollout 生成、奖励计算到策略更新的完整流水线。

从系统架构角度看，DAPO 的实现涉及以下关键层面（此处仅做概述，详细工程实践见第 8 章）：

- **Rollout 与训练的分离**：生成（rollout）和训练（update）在不同 GPU 组上异步执行，避免互相阻塞。这对动态采样尤为重要——过采样意味着 rollout 量大于训练量，异步化可以隐藏这部分额外开销。
- **权重同步**：在多机多卡环境下，policy 权重需要在 rollout worker 和 training worker 之间高效同步。DAPO 基于 verl 的分布式通信原语实现了这一点。
- **长序列处理的显存优化**：long-CoT 的序列长度可达数万 token，对显存提出了极高要求。DAPO 的实现采用了序列并行、激活检查点等技术来处理这一问题。

对学习者的价值：DAPO 的代码库（基于 verl）是目前研究 long-CoT RLVR 工程实现的最重要参考之一。它不仅展示了四项算法改进的具体实现方式，还呈现了将这些改进集成到生产级训练系统中的工程决策。建议读者在学习完本文档后，结合 DAPO 的代码库进行对照阅读，以建立从数学公式到工程实现的完整认知。

<a id="s6-9"></a>

## 6.9 本章小结与自测

<a id="h200"></a>

### 要点回顾

1. **DAPO 是 GRPO 的工程精修**，不改变组内基线和去 critic 的核心框架，而是针对 long-CoT 训练的四个具体病症各做一项改进。
2. **Clip-Higher 通过放宽上界 $`\varepsilon_{high}`$ 抑制熵崩塌**：低概率好 token 获得更大的概率增长空间，防止策略分布过早收敛至 one-hot。下界保持不变以维持稳定性。
3. **Dynamic Sampling 过滤全同奖励 prompt**，保证每个 batch 中的每个 prompt 都贡献非零梯度，同时形成隐式课程学习，将训练焦点维持在模型的能力边界。
4. **Token-Level Loss 消除长度偏差**，使每个 token 获得与回答长度无关的均匀梯度权重，对 long-CoT 中稀疏关键决策点的学习至关重要。
5. **Overlong Reward Shaping 通过软惩罚和截断过滤**，消除硬截断带来的奖励噪声，并堵住"故意超长骗分"的 hacking 路径。
6. **DAPO 移除 KL 项**是为了促进长链推理探索、降低显存开销，但这是针对 RLVR 数学推理的特定选择，不适用于所有场景。

<a id="h201"></a>

### 自测题

**题目 1**：解释为什么对称 clip（$`\varepsilon_{low} = \varepsilon_{high}`$）会对低概率 token 产生不成比例的限制，从而加速熵崩塌。请用重要性比率 $`r_t`$ 的行为来说明。

参考答案：对于一个当前概率很低的 token（$`\pi_\theta \approx \delta \ll 1`$），即使其绝对概率翻倍，$`r_t`$ 也只从 1 变为 2。对称 clip 的上界 $`1+\varepsilon`$（如 1.2）远低于 2，因此低概率 token 只需极小的相对增幅就触发 $`r_t > 1+\varepsilon`$ 的截断条件，梯度归零。相比之下，高概率 token 很难触及上界。长期累积效应是低概率好 token 无法被提升，概率质量持续向高概率 token 集中，熵持续下降。

**题目 2**：动态采样如何与第 4 章的全同奖励概率 $`P(\text{全同}) = p^G + (1-p)^G`$ 相关联？它如何实现隐式课程学习？

参考答案：$`P(\text{全同})`$ 在 $`p \approx 0`$ 或 $`p \approx 1`$ 时接近 1，意味着简单题和极难题的大部分采样都是零梯度的。动态采样过滤掉这些 prompt，只保留 $`0 < \text{acc} < 1`$ 的 prompt，即 $`p`$ 处于中间区域的题目。随着训练推进，模型变强，原来的中等题变成简单题（$`p \to 1`$）被自动过滤，新的中等难度题进入 batch。这形成了自适应的课程学习，始终将训练集中在 $`p \approx 0.5`$ 附近——信号最强的区域。

**题目 3**：写出 GRPO sequence-level loss 和 DAPO token-level loss 的数学表达式，并说明为什么后者能消除长度偏差。

参考答案：GRPO: $`L = \frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \ell_{i,t}`$；DAPO: $`L = \frac{1}{\sum_i |o_i|}\sum_i \sum_t \ell_{i,t}`$。在 GRPO 中，每条回答权重为 $`1/G`$，每 token 权重为 $`1/(G|o_i|)`$，与长度成反比。在 DAPO 中，每 token 权重为 $`1/\sum_j |o_j|`$，与所属回答长度无关。因此长回答的 token 不再被稀释，长度偏差被消除。

**题目 4**：DAPO 的软超长惩罚为什么优于硬截断？从优化景观的角度解释。

参考答案：硬截断在 $`L_{max}`$ 处产生阶跃间断，奖励从 0 突变为可能的误判分数。这个不可微的突变使策略梯度估计方差极大，且无法提供"应提前多少停止"的方向信息。软惩罚在缓冲区 $`[L_{max}-L_{cache}, L_{max}]`$ 内提供连续的线性递减信号，将悬崖改造为斜坡，策略可以沿平滑梯度逐步学到合适的长度控制。

**题目 5**：DAPO 移除 KL 项的三个理由是什么？在什么场景下不应该移除 KL？

参考答案：三个理由：① 长链推理需要策略大幅偏离 SFT 初始分布以探索新推理路径，KL 约束会扼杀探索；② RLVR 使用确定性 verifier，reward hacking 风险低于 RM 场景，KL 的防 hacking 作用不是必需；③ 去掉 KL 省去 reference model 的前向传播，节省显存。不应移除的场景：RLHF 对齐任务（需锚定人类偏好）、非推理类生成任务（语言质量可能退化）、训练数据覆盖面窄的场景。

<a id="chapter-7"></a>

# 第 7 章 序列级方法与变体谱系

<a id="s7-1"></a>

## 7.1 变体爆炸的背景与分类框架

<a id="s7-1-1"></a>

### 7.1.1 为什么 GRPO 变体如此之多

自 2024 年 GRPO 提出以来，围绕它的变体在 2025–2026 年间呈爆发式增长。这一现象有三重原因：

第一，**RLVR（Reinforcement Learning with Verifiable Rewards）成为推理模型训练的主流范式**。当可验证奖励取代奖励模型成为主要信号源后，GRPO 作为"无需 critic 的组采样策略优化"天然适配 RLVR 的设置（见第 4 章），各团队在落地时不可避免地遇到各种工程与理论病症，进而提出针对性修正。

第二，**多数改动的实现成本极低**。GRPO 的核心目标函数紧凑，许多改进只需修改损失计算的一两行代码（例如将 clip 上界调高、去掉 std 归一化、将 token 级比率换成序列级比率）。低实现门槛使得"打补丁→发论文"的周期极短。

第三，**问题本身是多维度的**。GRPO 同时涉及优势估计、裁剪机制、优化粒度、采样策略四个独立组件，每个组件都有改进空间，且不同组件的改动往往可以组合。这导致变体数量随组合数快速增长。

面对这个局面，读者最需要的不是逐个记忆变体细节，而是建立一个**分类坐标系**，在看到任何新变体时能迅速定位它"改了 GRPO 的哪个组件、治了什么病"。

<a id="s7-1-2"></a>

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

<a id="s7-2"></a>

## 7.2 GSPO：序列级重要性比率的数学动机

GSPO（Group Sequence Policy Optimization）由阿里巴巴 Qwen 团队提出（Zheng et al., 2025, arXiv:2507.18071），是 GRPO 家族中将优化粒度从 token 级提升到序列级的代表性工作。本节从数学机理出发，解释为什么这种粒度提升是必要的，以及它带来了什么本质变化。

<a id="s7-2-1"></a>

### 7.2.1 Token 级 IS 比率的三重问题

回顾 GRPO 的做法（见第 3 章）：对组内第 $`i`$ 条回答的第 $`t`$ 个 token，独立计算重要性采样比率

```math
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}
```

并在每个 token 上独立执行 PPO-style clip。这个设计有三个深层问题。

**问题一：方差随序列长度累积。** 如 PPO 文档第 5 章所述，重要性采样估计量的方差在比值分布重尾时可以无界。GRPO 虽然用单步比率避免了轨迹级比率的指数级方差增长，但一条长回答包含 $`|o_i|`$ 个 token，每个 token 的 $`r_{i,t}`$ 都是独立的高方差随机变量。整条序列的等效更新是这些 token 比率的联合作用，其联合方差随 $`|o_i|`$ 增大而累积。这在长 CoT（Chain-of-Thought）场景下尤为严重。

**问题二：优化单元与奖励单元的粒度错配。** 这是 GSPO 的核心批判。在 GRPO 中，奖励 $`R(q, o_i)`$ 是**序列级**的——整条回答获得一个分数，优势 $`\hat{A}_i`$ 对同一序列的所有 token 相同（见第 2 章）。然而裁剪与优化却是 **token 级**的：同一条回答中，某些 token 的 $`r_{i,t}`$ 超出 clip 范围被截断，另一些则没有。这意味着实际生效的梯度与"这条回答整体应该被强化/削弱多少"的语义意图不一致。一条本应被整体强化的好回答，可能因为其中几个 token 的比率碰巧偏高而被部分 clip 掉，导致学习信号被扭曲。

**问题三：MoE 架构下的路由抖动灾难。** GSPO 论文明确指出，GRPO 在 Mixture-of-Experts（MoE）模型上的训练极不稳定。机理如下：MoE 的路由器（router）在每个 token 上动态选择激活哪些专家。策略更新会改变路由决策，而路由改变又会让 token 级的 log 概率发生与"内容质量"无关的剧烈跳变——同一个 token 在不同专家组合下的条件概率可能差异巨大。于是 $`r_{i,t}`$ 出现与优化目标无关的巨大波动，导致梯度爆炸或训练崩溃。这种抖动是 MoE 架构固有的，与策略本身的优劣无关，因此属于需要被消除的噪声。

<a id="s7-2-2"></a>

### 7.2.2 GSPO 的序列级比率

GSPO 的解法是将优化粒度从 token 级提升到序列级。定义序列级重要性比率为

```math
s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}\right)
```

即 token 级比率的**几何平均**。长度归一化因子 $`1/|o_i|`$ 的作用至关重要：若不归一，$`\log s_i`$ 将是 $`|o_i|`$ 项 log-ratio 之和，其量级随序列长度线性增长，导致 $`s_i`$ 数值爆炸或下溢，无法用于稳定的 clip 优化。归一化后，$`s_i`$ 的数值范围与序列长度解耦，使得同一套超参可以适用于不同长度的回答。

GSPO 的目标函数在序列级做 clip：

```math
J^{GSPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\,\hat{A}_i,\ \text{clip}(s_i(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_i\right)\right]
```

其中 $`\hat{A}_i`$ 仍为序列级优势（如 GRPO 的组均值基线或 Dr.GRPO 的修正形式）。注意此处 clip 作用于**整条回答**：一条回答要么整体被强化、要么整体被削弱、要么整体被 clip，优化单元与奖励单元完全对齐。（注：上述目标函数是序列级形式的直接类比，具体细节以原论文为准。）

<a id="s7-2-3"></a>

### 7.2.3 梯度分析：GSPO vs GRPO

要理解 GSPO 与 GRPO 的本质差异，必须对比两者的梯度结构。

对 GSPO，由于 $`s_i(\theta) = \exp\left(\frac{1}{|o_i|}\sum_t \log r_{i,t}(\theta)\right)`$，由链式法则可得

```math
\nabla_\theta s_i(\theta) = s_i(\theta) \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
```

因此在未触发 clip 的情形下，GSPO 对参数 $`\theta`$ 的梯度为

```math
\nabla_\theta J^{GSPO} \propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G s_i(\theta)\,\hat{A}_i \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
```

**关键特征**：$`s_i(\theta)`$ 是一个**序列级标量**，它对同一序列内所有 token 的梯度施加相同的加权系数。

对比 GRPO 的梯度（见第 3 章）：

```math
\nabla_\theta J^{GRPO} \propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} r_{i,t}(\theta)\,\hat{A}_i \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
```

**关键区别**：GRPO 让每个 token 用自己的 $`r_{i,t}`$ 独立加权。当某个 token 的 $`r_{i,t}`$ 因路由抖动或其他噪声异常偏高时，该 token 的梯度会被过度放大；而在 GSPO 中，这种局部异常被几何平均平滑掉了。**GSPO 用一个序列级标量统一加权所有 token 的梯度，GRPO 则让每个 token 用自己的比率加权**——前者方差更低、对局部噪声更鲁棒，后者可能对个别 token 过度反应。这是理解两者差异的关键。

<a id="s7-2-4"></a>

### 7.2.4 ε 的量级差异

由于 $`s_i(\theta)`$ 是 token 级比率的几何平均，根据大数定律的性质，其偏离 1 的幅度天然比单个 $`r_{i,t}`$ 小得多。直觉上，多个独立随机变量的几何平均比方差最大的那个个体更接近期望值。因此 GSPO 的 clip 阈值 $`\varepsilon`$ 需要设得比 GRPO 显著更小——若 GRPO 使用 $`\varepsilon = 0.2`$，GSPO 可能需要小若干个数量级的 $`\varepsilon`$ 才能达到等效的约束强度。**具体数值请以原论文为准，不要直接套用 GRPO 的超参。** 若错误地使用 GRPO 量级的 $`\varepsilon`$，序列级 clip 几乎永远不会触发，等同于退化为无 clip 的 REINFORCE。

<a id="s7-2-5"></a>

### 7.2.5 GSPO 的实践影响

GSPO 论文报告其在 Qwen3 系列模型的 RL 训练中得到了应用，尤其在 MoE 架构上显著提升了训练稳定性。这验证了上述理论分析：当 token 级比率因架构特性（如 MoE 路由）而产生与优化目标无关的剧烈波动时，序列级聚合是一种有效的降噪手段。对于非 MoE 的 dense 模型，GSPO 的收益可能不如 MoE 场景显著，但其"粒度对齐"的设计理念仍具参考价值。

<a id="s7-3"></a>

## 7.3 CISPO：保留全部 Token 梯度

CISPO（Clipped IS-weight Policy Optimization）出自 MiniMax-M1 技术报告（arXiv:2506.13585），代表了 GRPO 变体谱系中与 GSPO 截然不同的另一条路线：它不改变优化粒度，而是改变裁剪机制本身，目标是**保留所有 token 的梯度**。

<a id="s7-3-1"></a>

### 7.3.1 Hard Clip 的信息丢弃问题

承接 PPO 文档第 6 章的六情形分析：PPO-style hard clip 一旦生效（即 $`r_{i,t}`$ 超出 $`[1-\varepsilon, 1+\varepsilon]`$ 范围），该 token 对目标函数的梯度**完全归零**。这在短回答场景中或许可以接受，但在长 CoT 场景下代价极大。

一条上万 token 的推理回答中，真正决定推理质量的往往是少数"关键转折 token"——例如反思词 "wait"、"let me recheck"、"alternatively" 等。这些 token 的出现标志着模型在进行自我纠错或路径切换，是高质量推理的核心信号。然而，正因为它们是低概率事件（模型在旧策略下不太可能生成这些词），其重要性比率 $`r_{i,t}`$ 容易远超 clip 上界，从而被 hard clip 完全丢弃梯度。结果是：模型学到了大量"常规 token"的信号，却丢失了最关键的"反思 token"的信号，推理能力难以持续提升。

<a id="s7-3-2"></a>

### 7.3.2 CISPO 的解法：Detached IS 权重

CISPO 的核心思想是：**不对目标函数做 PPO 式 hard ratio clipping，而是把重要性采样权重裁剪后作为 detached（stop-gradient）系数**。其损失形式类似 REINFORCE / policy gradient：

```math
L^{CISPO} = -\mathbb{E}\left[\frac{1}{\sum_i |o_i|}\sum_i \sum_t \text{sg}\!\left(\hat{r}^{IS}_{i,t}\right)\,\hat{A}_i\,\log\pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
```

其中 $`\hat{r}^{IS}_{i,t} = \text{clip}(r^{IS}_{i,t},\, 0,\, \varepsilon^{IS}_{high})`$ 是被裁剪的 IS 权重，$`\text{sg}(\cdot)`$ 表示 stop-gradient（即该部分不参与求导）。（注：这是依据"CISPO 裁剪 IS 权重作为 detached 系数而非 hard ratio clipping"这一核心描述的公式重构，**具体形式请以 MiniMax-M1 技术报告为准**。）

<a id="s7-3-3"></a>

### 7.3.3 梯度对比：CISPO vs PPO/GRPO

CISPO 与 PPO/GRPO 在梯度层面的本质区别可以通过直接求导看出。

对 CISPO，由于 $`\text{sg}(\hat{r}^{IS}_{i,t})`$ 被视为常数，梯度为

```math
\nabla_\theta L^{CISPO} = -\mathbb{E}\left[\frac{1}{\sum_i |o_i|}\sum_i \sum_t \text{sg}\!\left(\hat{r}^{IS}_{i,t}\right)\,\hat{A}_i\,\nabla_\theta \log\pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
```

**关键特征**：$`\nabla_\theta \log\pi_\theta`$ 项对**所有 token 都非零**。clip 只影响"这个 token 的权重多大"（通过 $`\text{sg}(\hat{r}^{IS}_{i,t})`$ 控制），而不影响"这个 token 有没有梯度"。即使某个 token 的原始比率 $`r^{IS}_{i,t}`$ 远超 clip 上界，它仍然以被裁剪后的权重 $`\varepsilon^{IS}_{high}`$ 参与梯度更新，而非被完全丢弃。

对比 PPO/GRPO 的梯度：当 $`r_{i,t} > 1+\varepsilon`$ 且 $`\hat{A}_i > 0`$ 时，$`\min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t})\hat{A}_i) = (1+\varepsilon)\hat{A}_i`$，该项对 $`\theta`$ 的梯度为零（因为 clip 后的值是常数）。**Hard clip 直接杀死梯度，CISPO 的 detached 权重只是缩放梯度**——这是两者的本质区别。

<a id="s7-3-4"></a>

### 7.3.4 为什么这对长 CoT 重要

CISPO 的设计实现了"保信号"与"控方差"的折中：

- **保信号**：稀有但关键的 token（如反思词）不会因为比率超限而失去学习信号。它们的梯度被保留，只是权重被限制在一个合理范围内。
- **控方差**：clip 上界 $`\varepsilon^{IS}_{high}`$（通常设得比 PPO 的 $`1+\varepsilon`$ 大得多，如 10 量级——**具体以原论文为准**）仍能防止单个 token 的权重爆炸导致方差失控。相比完全不 clip 的 REINFORCE，CISPO 通过宽松的上界提供了必要的安全阀。

这种设计特别适合长 CoT 推理模型的 RL 训练，其中关键信号的稀疏性与 IS 比率的极端值并存。MiniMax-M1 技术报告将 CISPO 用于其长 CoT 推理模型的训练，验证了这一思路的有效性。

<a id="s7-4"></a>

## 7.4 其他值得了解的变体

以下变体各有定位，简要概述以帮助读者建立完整谱系认知。

<a id="s7-4-1"></a>

### 7.4.1 RLOO

RLOO（REINFORCE Leave-One-Out，Ahmadian et al., 2024, arXiv:2402.14740）使用留一法基线（见第 2 章），无 clip 无 KL，是 GRPO 的极简近亲。实践中与 GRPO 效果常接近，TRL 框架有独立实现。适合想理解"组内基线的最小可行版本"的读者，也适合作为消融实验的 baseline。

<a id="s7-4-2"></a>

### 7.4.2 Dr.GRPO

Dr.GRPO（arXiv:2503.20783，第 5 章已详述）去掉长度归一化 $`(1/|o_i|)`$ 与 std 归一化，消除两种系统性偏差。在谱系中属于"改优势估计 + 改聚合方式"类。当观察到回答长度失控或难度依赖的训练不稳定时，Dr.GRPO 的修正是首选起点。

<a id="s7-4-3"></a>

### 7.4.3 GVPO

GVPO（Group Variance Policy Optimization，NeurIPS 2025，作业帮与港科大广州）的核心特点是**避免使用重要性采样**，损失等价于 MSE 形式，理论上给出唯一最优解保证，支持无 IS 的 off-policy 训练从而可复用历史数据，对超参 $`\beta`$ 不敏感。（注：此条信息来自二手资料，**以原论文为准**。）GVPO 的意义在于绕开了 IS 方差问题这条根本路径，而非在 IS 框架内做修补。

<a id="s7-4-4"></a>

### 7.4.4 BAPO

BAPO（Balanced Policy Optimization）采用自适应裁剪以平衡正负样本的梯度贡献、防止熵崩塌。其目标与 DAPO 的 clip-higher 相近（都是解决正负样本不对称问题），但机制不同：BAPO 动态调整 clip 范围而非固定解耦上下界。

<a id="s7-4-5"></a>

### 7.4.5 SAPO

SAPO（Soft Adaptive Policy Optimization）用软门控函数替代硬 clip，提供更平滑的梯度衰减曲线。有分析指出它在某些参数设置下退化到类 GSPO 的序列级形式，暗示序列级优化与软裁剪之间存在深层联系。

<a id="s7-4-6"></a>

### 7.4.6 长度偏差的后续工作

LUSPO（Length-Unbiased Sequence Policy Optimization）指出 GSPO 本身仍存在长度偏差并加以修正。这说明"长度偏差"是贯穿 token 级与序列级方法的共同难题（呼应第 5 章），并非切换到序列级就能自动解决。

<a id="s7-4-7"></a>

### 7.4.7 关于变体数量的声明

**这个领域变体极多且仍在快速演进**（2025–2026 年间新变体持续出现），本章只覆盖影响力最大的几支。其他已知名称包括 REINFORCE++-Baseline、GEPO、ESPO、MO-GRPO（多目标 reward hacking 缓解）、DAGRPC 等。读者遇到新变体时，应使用 7.1 节的四类框架去定位它改了什么组件，并以其原论文与主流框架（verl、TRL、OpenRLHF、NeMo-RL、AReaL）的实现为准。

<a id="s7-5"></a>

## 7.5 谱系图与演化逻辑

<a id="s7-5-1"></a>

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

<a id="s7-5-2"></a>

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

<a id="s7-6"></a>

## 7.6 选型建议

<a id="s7-6-1"></a>

### 7.6.1 决策路径

以下是面向实践者的选型决策路径，按优先级排列：

1. **任务有可靠 verifier、模型是 dense 架构、回答长度中等** → **GRPO 或 DAPO** 起步。DAPO 在长 CoT 上更稳，是当下更常用的默认选择。
2. **模型是 MoE 架构** → 优先考虑 **GSPO**。Token 级比率在 MoE 上有路由抖动问题（见 7.2.1），序列级比率能有效平滑。
3. **长 CoT 且观察到"关键 token 学不到"、熵快速下降** → 考虑 **CISPO**（保梯度）或 DAPO 的 clip-higher。两者机制不同但目标相近，可做消融对比。
4. **回答长度差异极大、观察到长度失控** → 必须处理长度偏差：**Dr.GRPO** 的去归一化修正 + DAPO 的超长软惩罚塑形。
5. **想复用历史 rollout 数据 / 担心 IS 方差** → 考虑 **GVPO** 类无 IS 方法。但注意第 2 章指出组内基线本身不适合跨 batch 复用，需配合相应的基线修正。
6. **资源受限、只想跑通验证想法** → **RLOO**（最简单，无 clip 无 KL）或直接用 GRPO 的小 $`G`$ 配置。

<a id="s7-6-2"></a>

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

<a id="s7-6-3"></a>

### 7.6.3 务实建议

**不要盲目追新变体。** 多数变体的收益在特定设置下才显著，且很多改动可组合。建议从 DAPO 或 GRPO 的成熟框架实现起步，先跑通并建立自己的基线与诊断能力（见第 8 章），再针对观察到的具体病症引入对应改动，**每次只改一处并做消融**。变体选择应由诊断驱动，而非由论文发表时间驱动。

<a id="s7-7"></a>

## 7.7 本章小结与自测

<a id="h233"></a>

### 要点回顾

1. GRPO 变体可按四组件分类：优势估计、裁剪机制、优化粒度、采样与数据。多数变体是局部替换且可组合。
2. GSPO 用序列级 IS 比率（token 级比率的几何平均）解决粒度错配与 MoE 路由抖动问题，梯度由序列级标量统一加权。
3. CISPO 用 detached IS 权重替代 hard clip，保留所有 token 的梯度，特别适合长 CoT 中稀疏关键信号的学习。
4. 序列级比率经长度归一化后波动范围远小于 token 级比率，ε 需显著调小。
5. 长度偏差是贯穿 token 级与序列级方法的共同难题，Dr.GRPO 与 LUSPO 分别从不同角度修正。
6. 选型应由诊断驱动：dense 模型用 DAPO/GRPO，MoE 用 GSPO，长 CoT 稀疏信号用 CISPO，长度失控用 Dr.GRPO。

<a id="h234"></a>

### 自测题

**题目 1**：GSPO 的序列级比率 $`s_i(\theta)`$ 为何能稳定 MoE 模型的 RL 训练？请从梯度的角度解释。

参考答案：MoE 路由器在每个 token 上动态选择专家，策略更新导致路由变化，使 token 级 log 概率产生与内容质量无关的剧烈跳变。GRPO 的 token 级比率 $`r_{i,t}`$ 直接反映这种跳变，导致个别 token 的梯度被异常放大或缩小。GSPO 的 $`s_i(\theta)`$ 是 token 级比率的几何平均，$`\nabla_\theta s_i = s_i \cdot (1/|o_i|)\sum_t \nabla_\theta \log\pi_\theta(o_{i,t}|\cdot)`$，用一个序列级标量统一加权所有 token 的梯度，局部路由抖动被平均效应平滑掉，因此梯度方差更低、训练更稳定。

**题目 2**：CISPO 与 PPO hard clip 在梯度上有什么本质区别？为什么这对长 CoT 重要？

参考答案：PPO hard clip 在比率超出范围时梯度完全归零，该 token 的学习信号被丢弃。CISPO 将裁剪后的 IS 权重作为 detached 系数（stop-gradient），梯度始终来自 $`\nabla_\theta \log\pi_\theta`$ 项，对所有 token 非零。clip 只控制权重大小，不决定有无梯度。在长 CoT 中，关键转折 token（如反思词）往往是低概率事件，其比率易超 clip 上界；CISPO 保留了这些稀有但关键 token 的学习信号，同时通过宽松的上界防止方差失控。

**题目 3**：为什么 GSPO 的 clip 阈值 $`\varepsilon`$ 不能直接沿用 GRPO 的值（如 0.2）？

参考答案：$`s_i(\theta)`$ 是 $`|o_i|`$ 个 token 级比率的几何平均，其偏离 1 的幅度天然比单个 $`r_{i,t}`$ 小得多。若使用 GRPO 量级的 $`\varepsilon`$，序列级 clip 几乎永远不会触发，GSPO 退化为无 clip 的 REINFORCE，失去约束策略更新幅度的安全机制。因此 $`\varepsilon`$ 必须显著调小以匹配序列级比率的波动范围。具体数值以原论文为准。

**题目 4**：使用 7.1 节的四组件分类框架，分析 DAPO 同时改了哪些组件？

参考答案：DAPO 改了三个组件：② 裁剪机制（clip-higher，解耦上下界 $`\varepsilon_{low}=0.2, \varepsilon_{high}=0.28`$）、③ 优化粒度（token-level loss 聚合）、④ 采样与数据（动态采样过滤全对全错组）。此外还移除了 KL 项。这体现了变体之间的可组合性。

**题目 5**：GVPO 与 GRPO/GSPO/CISPO 的根本区别是什么？它解决了什么问题？

参考答案：GVPO 的根本区别在于**完全避免使用重要性采样**，损失等价于 MSE 形式。GRPO/GSPO/CISPO 都在 IS 框架内做改进（改粒度、改裁剪方式），但都无法根除 IS 估计量本身的方差问题。GVPO 绕开了 IS 这条路径，理论上给出唯一最优解保证，并支持 off-policy 训练以复用历史数据。（注：以上信息来自二手资料，以原论文为准。）

<a id="chapter-8"></a>

# 第 8 章 工程实现与实践

本章是《GRPO 系统学习文档》的收尾章，目标是将前 7 章建立的理论体系落到可运行的代码与工程实践上。读者应已读完本文档第 1–7 章及 PPO 系统学习文档（含其第 7 章工程实现），具备 PyTorch 基础与 LLM RL 的数学直觉。本章不再重复推导，所有理论结论均通过引用定位；重点在于**每段代码与前文公式的精确对应**、工程决策背后的机理、以及一套可操作的诊断与排错方法论。

<a id="s8-1"></a>

## 8.1 框架选型与上手路径

<a id="s8-1-1"></a>

### 8.1.1 主流框架对比

下表汇总当前主流的 GRPO 训练框架。各框架迭代迅速，具体 API 名称与参数可能随版本变化，**请以官方文档为准**。

| 框架 | 定位 | 适用规模 | 支持算法 | 推理加速集成 | 上手难度 | 典型场景 |
|------|------|----------|----------|-------------|---------|---------|
| **TRL** (Hugging Face) | 轻量级、教学友好 | 单卡~8 卡，1B~7B | GRPO, RLOO, DAPO(部分) | 可选 vLLM/SGLang 后端 | ★★☆ | 学习、小规模实验、快速原型 |
| **verl** (字节跳动) | 生产级大规模训练 | 多机多卡，7B~数百B | GRPO, DAPO, Dr.GRPO | 原生 vLLM/SGLang，colocate/disagg | ★★★★ | 复现 DAPO/R1 级别实验、工业部署 |
| **OpenRLHF** | 通用 RLHF/RLVR 平台 | 多卡~多机 | GRPO, PPO, DPO 等 | 支持 vLLM | ★★★ | 多算法对比研究 |
| **NeMo-RL** (NVIDIA) | NVIDIA 生态集成 | 多卡~多机 | GRPO, CISPO, PPO | NeMo 推理栈 | ★★★ | CISPO 实验、NVIDIA GPU 深度优化 |
| **AReaL** (inclusionAI) | 算法文档+实现参考 | 研究导向 | GRPO, Dr.GRPO, DAPO, GSPO | 视配置 | ★★★ | MoE/GSPO 研究、算法对照阅读 |
| **Unsloth** | 单卡极致效率 | 单卡，≤7B | GRPO | 自研加速内核 | ★★☆ | 消费级显卡上的 GRPO 微调 |

**选型建议**：

- **学习与小规模实验用 TRL**：单卡或少量卡即可跑 0.5B~7B 模型的 GRPO，API 简洁，社区资源丰富。
- **大规模生产用 verl**：DAPO 官方基于 verl 实现，支持 colocate/disaggregated 部署，适合复现 R1 级别长 CoT 训练。
- **MoE / GSPO 相关看 AReaL 与 Qwen 公开实现**：AReaL 文档整理了 PPO/GRPO/Dr.GRPO/DAPO/GSPO 系列的算法说明，便于对照。
- **CISPO 看 NeMo-RL 指南**：NeMo-RL 提供了 CISPO 的实现指南与 MiniMax-M1 相关参考。
- **单卡资源受限用 Unsloth**：在消费级 GPU 上高效微调小模型。

<a id="s8-1-2"></a>

### 8.1.2 TRL 最小可运行示例

以下示例使用 TRL 的 `GRPOTrainer` 在一个小模型上训练"回答必须包含关键词"的简单任务。**注意：TRL 的 API 名称与参数可能随版本变化，请以官方文档为准。** 代码中每个关键参数都标注了对应的前文章节。

```python
"""
TRL GRPO 最小可运行示例
依赖: pip install trl transformers torch datasets
注意: API 名称可能随 TRL 版本变化，请以官方文档为准
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset


# ========== 1. 自定义奖励函数（对应第 4 章 R(q,o) = f(extract(o), y*)）==========
def keyword_reward_func(completions, **kwargs):
    """
    二值规则奖励：回答中包含目标关键词得 1.0，否则得 0.0。
    对应第 4 章的规则奖励（Rule-based Reward）。
    completions: list[str]，每个元素是一条完整回答文本。
    """
    target_keyword = "光合作用"
    rewards = []
    for completion in completions:
        # 简单的关键词匹配；实际任务中应使用更鲁棒的抽取逻辑（见 8.2.1）
        if target_keyword in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ========== 2. 构造简单数据集 ==========
data = [
    {"prompt": "请解释植物如何获取能量。"},
    {"prompt": "光合作用的基本过程是什么？"},
    {"prompt": "为什么植物需要阳光？"},
    {"prompt": "描述叶绿体的功能。"},
]
dataset = Dataset.from_list(data)


# ========== 3. 加载模型与 tokenizer ==========
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 小模型用于演示
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ========== 4. GRPO 配置（每个参数标注对应章节）==========
config = GRPOConfig(
    # --- 组采样 ---
    num_generations=8,           # G=8，对应第 2 章组大小；G<4 时 std 极不可靠
    max_completion_length=256,   # 对应第 6 章截断控制

    # --- 损失函数超参 ---
    beta=0.01,                   # β，KL 惩罚系数，对应第 3 章；RLVR 常设 0
    epsilon=0.2,                 # ε_low，clip 下界，对应第 3 章
    # epsilon_high=0.28,         # ε_high，clip-higher 上界，对应第 6 章 DAPO
                                 # 以官方文档确认该参数名是否可用

    # --- Dr.GRPO 开关 ---
    # scale_rewards=False 时不除 std，对应第 5 章 Dr.GRPO 去难度偏差
    # 默认 True 为标准 GRPO；设为 False 得到 Dr.GRPO 式行为
    # 参数名可能随版本变化，请以官方文档为准

    # --- 损失聚合方式 ---
    # loss_type 控制 sequence-mean vs token-mean，对应第 5 章长度偏差讨论
    # 可选值以官方文档为准

    # --- 训练超参 ---
    learning_rate=1e-6,          # LLM RL 微调用 1e-6 量级，对应第 3 章
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=1,
    output_dir="./grpo_demo_output",

    # --- 推理加速（可选）---
    # use_vllm=True,             # 启用 vLLM 做 rollout 生成，大幅提速
    # vllm_device="cuda:0",      # 以官方文档确认可用参数
)


# ========== 5. 启动训练 ==========
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=[keyword_reward_func],  # 可传多个奖励函数（第 4 章组合奖励）
)

trainer.train()
print("训练完成。检查 output_dir 中的日志与 checkpoint。")
```

**运行说明**：上述代码可在单张消费级 GPU 上运行（0.5B 模型 + G=8 + 短序列）。将 `model_name` 替换为本地路径或其他小模型即可。若遇到 API 变更导致的参数错误，请查阅 TRL 官方文档中 `GRPOConfig` 的最新定义。

<a id="s8-2"></a>

## 8.2 从零实现 GRPO 的核心逻辑

本节给出一份结构清晰、与公式逐行对应的 PyTorch 参考实现。这不是一个完整的训练框架，而是**剥离了工程复杂性后的算法核心**，用于帮助读者理解每一行代码背后的数学含义。实际生产中请使用 8.1 节的成熟框架。

<a id="s8-2-1"></a>

### 8.2.1 奖励函数：规则奖励与防 hacking

以下实现对应第 4 章的 $`R(q, o) = f(\text{extract}(o), y^*)`$，并针对第 4 章提到的 reward hacking 路径做了加固。

```python
import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    从回答中提取 \\boxed{} 内的答案。
    对应第 4 章的答案抽取逻辑。

    防 hacking 措施（对应第 4 章 reward hacking 路径）：
    - 多个 \\boxed{} 时取最后一个（而非第一个），因为模型可能在推理过程中
      修正答案，最终答案通常在末尾。
    - 抽取失败返回 None，由调用方决定给 0 分还是格式惩罚分。
    - 去除首尾空白，避免空格差异导致误判。
    """
    # 匹配 \boxed{...}，支持嵌套花括号（非贪婪匹配最外层）
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    # 取最后一个匹配结果（防多 boxed hacking）
    return matches[-1].strip()


def math_verifier_reward(
    completion: str,
    ground_truth: str,
    format_penalty: float = -0.1,
) -> float:
    """
    数学题的规则奖励函数。
    对应第 4 章 R(q, o) = f(extract(o), y*)。

    Args:
        completion: 模型生成的完整回答。
        ground_truth: 标准答案字符串。
        format_penalty: 抽取失败时的格式惩罚分（对应第 4 章格式奖励）。

    Returns:
        奖励标量。正确 = 1.0，错误 = 0.0，抽取失败 = format_penalty。
    """
    extracted = extract_boxed_answer(completion)
    if extracted is None:
        # 抽取失败：给一个小的负分作为格式信号，而非直接给 0
        # 这避免了"故意不写 boxed 也能拿 0 分"的 hacking 路径
        return format_penalty
    # 精确字符串匹配；对于数值题可加 normalize 逻辑
    if extracted == ground_truth.strip():
        return 1.0
    else:
        return 0.0
```

**与理论的对应与差异**：第 4 章指出奖励尺度不变性（$`R \to aR + b`$ 时 $`\hat{A}`$ 不变，$`a > 0`$），因此 1.0/0.0 的具体数值不影响优势计算。`format_penalty` 的选择也不影响组内归一化后的优势方向，但会影响组内 std 的大小，进而影响梯度幅度。实践中建议通过实验确定。

<a id="s8-2-2"></a>

### 8.2.2 Rollout 与组采样：old_logprob 的正确获取

这是 GRPO 实现中**最高频的错误点**。核心原则：**$`\log \pi_{\theta_{old}}(o_{i,t}|\cdot)`$ 必须在生成 rollout 的同一次前向传播中取出，或在 `torch.no_grad()` 下用完全相同的参数重算。** 绝不能事后再用更新过的模型计算——否则 old_logprob 与数据不匹配，比率 $`r`$ 从第一步就不为 1，clip 行为完全错乱。

```python
@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompts: list[str],
    num_generations: int,       # G，对应第 2 章
    max_completion_length: int,
) -> dict:
    """
    对每个 prompt 生成 G 个回答，并同时记录 old_logprob。
    对应第 3 章的采样过程 {o_i} ~ π_{θ_old}(·|q)。

    关键：old_logprob 在本次前向中一并取出，保证与生成结果严格匹配。
    这是避免第 8.5 节坑 #2 的核心措施。

    Returns:
        dict with keys: 'input_ids', 'attention_mask', 'completion_mask',
                        'old_logprobs', 'rewards'
    """
    G = num_generations
    B = len(prompts)

    # 将每个 prompt repeat G 次，得到 B*G 条输入
    repeated_prompts = []
    for p in prompts:
        repeated_prompts.extend([p] * G)

    inputs = tokenizer(repeated_prompts, return_tensors="pt", padding=True,
                       truncation=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # 生成回答（greedy 或 sampling 均可；GRPO 通常用 sampling）
    gen_kwargs = dict(
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        return_dict_in_generate=True,
        output_scores=True,       # 返回每步 logits 以计算 logprob
    )
    outputs = model.generate(**inputs, **gen_kwargs)

    # 提取 completion 部分的 token ids
    all_ids = outputs.sequences                    # shape: (B*G, prompt_len + gen_len)
    completion_ids = all_ids[:, prompt_len:]        # shape: (B*G, gen_len)

    # ===== 关键：在同一次前向中计算 old_logprob =====
    # 方法：用生成的 token ids 做一次 forward，取对应位置的 log_softmax
    # 注意：必须在 torch.no_grad() 下、用相同参数执行
    full_logits = model(all_ids).logits             # (B*G, seq_len, vocab)
    # 取 completion 每个位置对应的下一个 token 的 log prob
    # logits[t] 预测的是 token[t+1]，所以用 shift
    log_probs = torch.log_softmax(full_logits[:, :-1, :], dim=-1)
    # completion 从 prompt_len 开始，对应的 logit 索引是 prompt_len-1 到 end-1
    comp_log_probs = log_probs[:, prompt_len - 1:, :]  # (B*G, gen_len, vocab)
    # gather 出实际生成 token 的 log prob
    old_logprobs = comp_log_probs.gather(
        dim=-1, index=completion_ids.unsqueeze(-1)
    ).squeeze(-1)                                    # (B*G, gen_len)

    # 构造 completion_mask：只对有效 completion token 为 1
    # 对应 8.2.6 的 mask 处理
    pad_token_id = tokenizer.pad_token_id
    completion_mask = (completion_ids != pad_token_id).float()

    return {
        "input_ids": all_ids,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "old_logprobs": old_logprobs,                # 与生成结果严格匹配
        "prompt_len": prompt_len,
    }
```

**与理论的差异说明**：实际框架（如 TRL）通常在 `model.generate` 内部直接返回 logprob，避免额外的 forward pass。上面的写法是为了清晰展示原理。在生产代码中，应优先使用框架提供的机制以确保效率与正确性。

<a id="s8-2-3"></a>

### 8.2.3 优势计算：严格按 prompt 分组

```python
def compute_group_advantages(
    rewards: torch.Tensor,        # shape: (B*G,)
    batch_size: int,              # B
    num_generations: int,         # G
    scale_rewards: bool = True,   # True=标准GRPO, False=Dr.GRPO（第5章）
    eps: float = 1e-4,            # 数值保护，对应第 2 章 σ̂→0 问题
) -> torch.Tensor:
    """
    按 prompt 分组计算组内归一化优势。
    对应第 2 章 Â_i = (r_i - mean) / std 与第 3 章的优势定义。

    ⚠️ 关键：必须 reshape 成 (B, G) 后沿 G 维统计。
    在整个 batch 上统计是常见 bug（见 8.5 节坑 #1），
    会让不同 prompt 的奖励互相污染基线。

    Args:
        rewards: 扁平的奖励张量，排列为 [prompt0_gen0, prompt0_gen1, ...,
                 prompt1_gen0, ...]。
        scale_rewards: False 时只减均值不除 std，对应第 5 章 Dr.GRPO。
        eps: 防止 std=0 时除零，对应第 2 章"组内奖励全同时 σ̂=0"。
    """
    G = num_generations
    # 严格 reshape 成 (B, G)，沿 G 维（dim=1）统计
    grouped = rewards.view(batch_size, G)             # (B, G)
    group_mean = grouped.mean(dim=1, keepdim=True)     # (B, 1)
    group_std = grouped.std(dim=1, keepdim=True)       # (B, 1)

    if scale_rewards:
        # 标准 GRPO：Â = (r - mean) / (std + eps)
        # eps 保护对应第 2 章：当组内奖励全同时 std=0，不加 eps 会产生 NaN
        advantages = (grouped - group_mean) / (group_std + eps)
    else:
        # Dr.GRPO（第 5 章）：去掉 std 归一化，消除难度偏差
        # 此时优势仅做组内去均值，保留了原始奖励的量级信息
        advantages = grouped - group_mean

    # 展平回 (B*G,) 以便后续与 token 级数据对齐
    return advantages.view(-1)                         # (B*G,)
```

<a id="s8-2-4"></a>

### 8.2.4 将序列级优势广播到每个 token

```python
def broadcast_advantages_to_tokens(
    adv_per_response: torch.Tensor,   # (B*G,)
    completion_mask: torch.Tensor,    # (B*G, T)
) -> torch.Tensor:
    """
    将序列级优势广播到每个 token。
    对应第 1 章"outcome reward 下所有 token 共享优势"的结论，
    以及第 3 章 Â_{i,t} = Â_i。

    同一个回答的所有 token 获得相同的优势值。
    mask 位置的优势虽然被赋值，但在损失计算中会被 mask 掉。
    """
    # unsqueeze(-1) 将 (B*G,) → (B*G, 1)，再 expand 到 (B*G, T)
    advantages = adv_per_response.unsqueeze(-1).expand_as(completion_mask)
    return advantages
```

<a id="s8-2-5"></a>

### 8.2.5 损失计算：比率、clip、k3 KL、聚合

```python
def compute_grpo_loss(
    new_logprobs: torch.Tensor,      # (B*G, T)，当前策略的 log π_θ
    old_logprobs: torch.Tensor,      # (B*G, T)，旧策略的 log π_{θ_old}
    ref_logprobs: torch.Tensor,      # (B*G, T)，参考模型的 log π_{ref}
    advantages: torch.Tensor,        # (B*G, T)，已广播的 token 级优势
    completion_mask: torch.Tensor,   # (B*G, T)
    epsilon: float = 0.2,            # clip 范围，对应第 3 章
    beta: float = 0.01,              # KL 系数，对应第 3 章
    loss_type: str = "token_mean",   # 聚合方式，对应第 5 章
) -> dict:
    """
    GRPO 损失计算核心。
    对应第 3 章带 clip 的实用形式与 k3 KL 估计器。

    loss_type:
        - "seq_mean": 先对每个序列内取平均，再对 batch 取平均。
          对应第 3 章原始形式 (1/|o_i|)Σ_t，但有长度偏差（第 5 章）。
        - "token_mean": 对所有有效 token 取全局平均。
          对应第 5 章 Dr.GRPO 的无偏聚合，消除长度偏差。
    """
    # ===== ① 逐 token 比率（对数域计算，对应 PPO 文档第 7 章数值稳定性）=====
    # logratio = log(π_θ / π_{θ_old}) = log π_θ - log π_{θ_old}
    # 在对数域做减法比直接除更稳定，避免 exp 溢出
    logratio = new_logprobs - old_logprobs
    ratio = logratio.exp()                              # r_{i,t}(θ)

    # ===== ② Clip 项（对应第 3 章 min(r·Â, clip(r,1±ε)·Â)）=====
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    # min 实现保守更新：正优势时取较小者限制增大，负优势时取较大者限制减小
    policy_loss = -torch.min(surr1, surr2)              # 取负号转为最小化

    # ===== ③ k3 KL 估计器（对应第 3 章 §3.3）=====
    # k3: D_KL ≈ exp(log_ref - log_new) - (log_ref - log_new) - 1
    # 恒非负且在 log_ref ≈ log_new 时二阶近似于真实 KL
    log_ratio_ref = ref_logprobs - new_logprobs         # log(π_ref / π_θ)
    kl_k3 = log_ratio_ref.exp() - log_ratio_ref - 1.0   # 恒 ≥ 0

    # ===== ④ 合并损失 =====
    # 对应第 3 章: loss = policy_loss - β * KL（注意 KL 前是减号，
    # 因为 KL 惩罚要增大 loss，而 policy_loss 本身已取负）
    total_loss_per_token = policy_loss + beta * kl_k3

    # ===== ⑤ Mask 与聚合（对应第 5 章长度偏差讨论）=====
    if loss_type == "seq_mean":
        # 有偏写法：先对每个序列内取 masked mean，再对 batch 取 mean
        # 对应第 3 章原始 (1/|o_i|)Σ_t 形式
        # ⚠️ 这会引入长度偏差（第 5 章 §5.2）
        seq_lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
        seq_loss = (total_loss_per_token * completion_mask).sum(dim=-1) / seq_lengths
        loss = seq_loss.mean()
    elif loss_type == "token_mean":
        # 无偏写法：对所有有效 token 取全局平均
        # 对应第 5 章 Dr.GRPO 的修正：消除 (1/|o_i|) 的长度偏差
        loss = (total_loss_per_token * completion_mask).sum() / completion_mask.sum()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 返回诊断指标（对应 8.4 节）
    with torch.no_grad():
        approx_kl = (ratio - 1.0 - logratio).mean()     # 二阶近似 KL
        clip_frac = ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).float().mean()

    return {
        "loss": loss,
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_frac.item(),
        "kl_k3_mean": (kl_k3 * completion_mask).sum().item() / completion_mask.sum().item(),
    }
```

**与理论的差异说明**：
1. **k3 KL 估计器**：第 3 章证明了 k3 是无偏且恒非负的。代码中 `log_ratio_ref.exp() - log_ratio_ref - 1.0` 直接对应 k3 公式。当 `new_logprobs ≈ ref_logprobs` 时，该式二阶近似于 $`(\Delta)^2/2`$，数值行为良好。
2. **对数域算比率**：承接 PPO 文档第 7 章的数值稳定性处理，避免直接做 `π_θ / π_{θ_old}` 导致的浮点溢出。
3. **approx_kl 的二阶近似**：`(exp(logratio) - 1 - logratio)` 对应 PPO 文档第 7 章的 `approx_kl` 计算公式，它在 `logratio ≈ 0` 时近似于 `(logratio)²/2`，比直接用 `logratio.mean()` 更准确。

<a id="s8-2-6"></a>

### 8.2.6 Mask 的正确处理

Mask 错误是 GRPO 实现中最隐蔽的 bug 之一。以下给出 `completion_mask` 的构造原则：

```python
def build_completion_mask(
    input_ids: torch.Tensor,       # (B*G, prompt_len + gen_len)
    prompt_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    构造 completion_mask：仅对 completion 部分的有效 token 为 1。
    prompt 部分和 padding 部分必须为 0。

    ⚠️ mask 错误会导致梯度被 prompt token 或 padding token 污染，
    这是 8.5 节坑 #3 描述的常见问题。
    """
    seq_len = input_ids.shape[1]
    mask = torch.zeros_like(input_ids, dtype=torch.float)
    # completion 从 prompt_len 开始
    mask[:, prompt_len:] = 1.0
    # padding 位置置 0
    mask[input_ids == pad_token_id] = 0.0
    # 只保留 completion 部分（与 completion_ids 对齐）
    return mask[:, prompt_len:]                          # (B*G, gen_len)
```

**为什么 mask 如此重要**：GRPO 的损失是对 completion token 求平均。如果 prompt token 未被 mask 掉，它们的 logprob 也会参与梯度计算，但这些 token 不是模型"选择"的（它们是给定的条件），对其施加梯度会破坏条件概率的正确性。如果 padding token 未被 mask 掉，它们的 logprob 通常是任意值（取决于初始化），会引入纯噪声梯度。

<a id="s8-3"></a>

## 8.3 显存、并行与 Rollout 加速

<a id="s8-3-1"></a>

### 8.3.1 显存分解

承接第 3 章的 $`16\Phi`$ 估算（GRPO 相比 PPO 省去了 critic 模型的 $`8\Phi`$），本节展开说明显存的具体构成。设模型参数量为 $`\Phi`$，精度为 bf16（2 字节/参数）。

**训练侧显存**：
- Policy 权重：$`2\Phi`$ 字节
- 梯度：$`2\Phi`$ 字节
- Adam 优化器状态（一阶矩 + 二阶矩）：$`4\Phi + 4\Phi = 8\Phi`$ 字节（fp32）
- fp32 master weights（混合精度训练时）：$`4\Phi`$ 字节
- **训练侧合计**：约 $`16\Phi`$ 字节（bf16 训练 + fp32 优化器状态）

**推理侧显存**：
- Reference model 权重：$`2\Phi`$ 字节（仅权重，无梯度和优化器状态）
- Rollout 引擎（vLLM/SGLang）的 KV cache：与序列长度线性相关

**KV Cache 估算式**：

```math
\text{KV Cache} \approx 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes\_per\_elem}
```

其中 $`L`$ 为层数，$`H_{kv}`$ 为 KV 头数（GQA 时小于注意力头数），$`d_{head}`$ 为头维度，$`S`$ 为序列长度，$`B`$ 为 batch size，`bytes_per_elem` 为精度字节数（bf16=2, fp16=2, fp32=4）。

**长 CoT 的影响**：当 $`S`$ 从 512 增长到 8192 时，KV cache 增长 16 倍。对于 7B 模型（$`L=32, H_{kv}=8, d_{head}=128`$），$`S=8192, B=32`$ 时 KV cache 约 $`2 \times 32 \times 8 \times 128 \times 8192 \times 32 \times 2 \approx 26.8`$ GB。这就是为什么长 CoT 训练中 KV cache 往往比模型权重本身占用更多显存。

<a id="s8-3-2"></a>

### 8.3.2 省显存的手段及其代价

| 技术 | 省了什么 | 代价 | 适用场景 |
|------|---------|------|---------|
| **LoRA / QLoRA** | 只训练低秩增量矩阵（通常 <1% 参数），省去大部分梯度和优化器状态。**关键优势**：关掉 adapter 就是 reference model，无需额外加载 ref model，直接省下 $`2\Phi`$ | 表达能力受限；QLoRA 的 4-bit 量化可能影响精度 | 中小规模微调，显存紧张时首选 |
| **Gradient Checkpointing** | 不保存中间激活值，反向传播时重算，省去激活显存 | 训练速度下降约 20%~30%（重算开销） | 序列较长或 batch 较大时 |
| **优化器 Offload** | 将 Adam 状态卸载到 CPU 内存 | 显著降低训练速度（CPU↔GPU 传输瓶颈） | 显存极度紧张时的最后手段 |
| **8-bit Adam** | 优化器状态从 fp32 压缩到 8-bit，省约 $`6\Phi`$ | 可能有轻微精度损失 | 大模型训练的常用选项 |
| **bf16 混合精度** | 前向/反向用 bf16，优化器状态保持 fp32 | 需要 Ampere+ GPU；某些操作需手动 cast | 几乎所有现代 LLM 训练的默认选择 |

**LoRA 与 Reference Model 的关系**：这是一个重要的工程洞察。在 LoRA 设置下，reference model 就是 base model + 关闭的 adapter。因此不需要单独加载一份 reference model 权重，只需在前向时禁用 adapter 即可得到 ref_logprobs。这直接省下 $`2\Phi`$ 显存，对小显存环境意义重大。

<a id="s8-3-3"></a>

### 8.3.3 并行策略

| 并行策略 | 切什么 | 通信开销 | 适用规模 | 备注 |
|---------|--------|---------|---------|------|
| **数据并行 (DP/DDP)** | 数据切分到各 GPU，每张卡持有完整模型副本 | AllReduce 梯度同步 | 单机多卡 | 最简单，但受限于单卡显存 |
| **FSDP / DeepSpeed ZeRO** | 参数/梯度/优化器状态分片到各 GPU | AllGather 参数 + ReduceScatter 梯度 | 多卡~多机 | ZeRO-1 只分优化器状态；ZeRO-2 加分梯度；ZeRO-3 全分 |
| **张量并行 (TP)** | 单层内部矩阵运算切分到多 GPU | AllReduce/AllGather 每层两次 | 单卡放不下的大模型 | 需要高速互联（NVLink） |
| **流水线并行 (PP)** | 不同层分配到不同 GPU | 点对点激活传递 | 超深模型 | 存在 bubble，需微批次调度 |
| **Expert 并行 (EP)** | MoE 的不同 expert 分配到不同 GPU | AllToAll token 路由 | MoE 模型专用 | 呼应第 7 章 GSPO 与 MoE 稳定性 |

<a id="s8-3-4"></a>

### 8.3.4 Rollout 与训练的耦合方式

Rollout 生成是 LLM RL 的时间瓶颈，通常占总训练时间的 **70%~90%**（承接第 3 章）。G 倍采样的成本主要落在这里，推理引擎的吞吐直接决定训练速度。

**Colocate 模式**（训练与推理共用 GPU）：
- 同一组 GPU 上交替执行训练和推理
- 优点：省显存（不需要两套 GPU），权重同步零延迟
- 缺点：切换开销（加载/卸载推理引擎）、训练与推理不能同时进行
- 适用：GPU 数量有限、模型不太大的场景

**Disaggregated 模式**（分离部署）：
- 训练和推理分别占用独立的 GPU 集群
- 优点：吞吐高（两者可并行）、无切换开销
- 缺点：需要更多 GPU、权重同步是核心难点
- 适用：大规模生产训练（如 R1 级别）

**权重同步**是 LLM RL 工程的核心难点之一：每次训练更新后，需要将新的 policy 权重推送给推理引擎。这涉及：(1) 显存峰值（可能需要临时存储两份权重）、(2) 同步延迟（大模型权重传输耗时）、(3) 一致性保证（推理引擎必须使用最新的权重）。verl 等框架在这方面做了大量工程优化。

**推理加速引擎**：vLLM 和 SGLang 是当前主流选择。它们通过 PagedAttention、连续批处理（continuous batching）、投机解码等技术大幅提升生成吞吐。没有这些引擎，G=64 的组采样在 7B 模型上可能需要数十分钟才能完成一个 batch 的 rollout。

<a id="s8-4"></a>

## 8.4 训练诊断指标

承接并扩展 PPO 文档第 7 章的指标体系。GRPO 场景下部分指标含义发生变化，且有新增指标。

<a id="s8-4-1"></a>

### 8.4.1 指标速查表

| 指标 | 数学含义 | 健康范围 | 异常时的排查方向 |
|------|---------|---------|----------------|
| **reward** | 平均奖励 $`\mathbb{E}[r]`$ | 稳步上升 | 不涨 → 看 accuracy 分布、zero_grad_ratio |
| **accuracy** | 通过率（二值奖励下的 reward 均值） | 从初始值逐步上升至 0.7~0.95 | 长期 ≈0 或 ≈1 → 信号枯竭（第 4 章），调整数据难度 |
| **completion_length** | 平均回答长度 | 稳定或缓慢增长 | 单调暴涨而 reward 不涨 → 长度偏差/hacking（第 4、5 章）；暴跌 → 崩溃 |
| **entropy** | 策略熵 $`-\sum \pi \log \pi`$ | 缓慢下降 | 快速崩塌 → clip-higher 要解决的问题（第 6 章），加熵正则 |
| **approx_kl** | 新旧策略 KL 的二阶近似 $`\exp(\delta)-1-\delta`$ | <0.1 | 超阈值 → 更新步长过大，降学习率或增 G |
| **kl_to_ref** | 与 reference model 的 KL | 缓慢增长 | β=0 时这是**唯一偏离监控**（第 3 章），突增 → 策略退化 |
| **clip_fraction** | 被 clip 的 token 占比 | K=1 时应 ≈0 | 非 0 → old_logprob 不匹配（实现 bug）或异步更新 |
| **std_group** | 组内奖励标准差 | >0 且不过大 | ≈0 → 零梯度（第 2 章），增大 G 或调整难度 |
| **zero_grad_ratio** | Â 全为 0 的 prompt 占比 | <10% | 过高 → 动态采样（第 6 章）或课程学习 |
| **grad_norm** | 梯度范数 | 稳定 | 突增 → 崩溃前兆，加 gradient clipping |
| **explained_variance** | — | — | **GRPO 没有此指标**（无 critic），这是与 PPO 的重要差异 |

<a id="s8-4-2"></a>

### 8.4.2 GRPO 特有的诊断逻辑

**clip_fraction 的特殊含义**：承接第 3 章，当 $`K=1`$（单次更新，即生成数据后立即训练一步就丢弃）时，$`\theta = \theta_{old}`$，比率 $`r_{i,t} = 1`$ 恒成立，不应有任何 token 被 clip。若观察到 `clip_fraction > 0`，只有两种可能：(1) old_logprob 与 new_logprob 不匹配（实现 bug，见 8.5 节坑 #2）；(2) 使用了 $`K > 1`$ 的多 epoch 更新。这个诊断逻辑在调试时非常有价值。

**kl_to_ref 在 β=0 时的重要性**：RLVR 场景中常设 $`\beta = 0`$（去掉 KL 惩罚以获得更强的探索）。此时损失函数中没有 KL 项，但**必须仍然监控 kl_to_ref**。否则策略可能在不知不觉中严重偏离初始模型，导致语言质量崩坏（如输出乱码、重复、格式退化）而无任何预警。

**explained_variance 的缺失**：PPO 有 critic 网络提供价值估计，`explained_variance` 衡量 critic 对回报方差的解释比例。GRPO 没有 critic，用组内均值替代了价值函数，因此不存在这个指标。取而代之应监控组内基线的质量：G 的大小、std 的分布、zero_grad_ratio。

<a id="s8-4-3"></a>

### 8.4.3 症状驱动的排查顺序清单

当训练出现问题时，按以下顺序排查：

1. **reward 不涨** → 先看 accuracy 分布（是否全对/全错）
2. accuracy 全对或全错 → 看 **zero_grad_ratio**（是否大量 prompt 零梯度）→ 调整数据难度或启用动态采样（第 6 章）
3. accuracy 正常但 reward 仍不涨 → 看 **entropy** 是否崩塌 → 加熵正则或 clip-higher
4. entropy 正常 → 看 **completion_length** 是否失控 → Dr.GRPO 修正 + 超长塑形（第 5、6 章）
5. completion_length 正常 → 看 **clip_fraction** 是否异常（判断 old_logprob 是否匹配）
6. clip_fraction 正常 → 看 **grad_norm** 是否突增 → 加 gradient clipping 或降学习率
7. 以上均正常但效果不佳 → 检查 G 是否太小（≥8）、loss_type 是否正确、mask 是否正确

<a id="s8-5"></a>

## 8.5 常见坑清单

以下每条均为「症状 → 根因 → 修复」格式，覆盖 GRPO 工程实践中最高频的问题。

| # | 症状 | 根因 | 修复 |
|---|------|------|------|
| 1 | reward 波动剧烈、训练不稳定 | 优势在整个 batch 上归一化而非按 prompt 分组，基线被跨 prompt 污染 | 严格 reshape 成 (B, G) 后沿 G 维统计（见 8.2.3） |
| 2 | clip_fraction 在 K=1 时非 0；比率从一开始就 ≠1 | old_logprob 事后用更新过的模型重算，与生成数据不匹配 | 必须在 rollout 同一次前向中取出 old_logprob（见 8.2.2） |
| 3 | loss 异常大或小；训练初期就发散 | mask 未排除 prompt/padding token，梯度被无关 token 污染 | 用 completion_mask 严格屏蔽非 completion 区域（见 8.2.6） |
| 4 | 短回答被过度偏好；长回答质量下降 | 用 `dim=-1` 先做序列内平均再 batch 平均，引入长度偏差（第 5 章） | 改用 token-mean 聚合：`(loss*mask).sum() / mask.sum()` |
| 5 | 训练几乎无进展；advantage 全接近 0 | G 设得太小（如 2），std 估计极不可靠（第 2 章） | 至少 G≥8；推荐 G=16~64 |
| 6 | NaN/Inf 出现在 loss 或梯度中 | 组内奖励全同时 std=0，除零产生 NaN（第 2 章） | 加 eps 保护（`std + 1e-4`）或用 `scale_rewards=False` |
| 7 | zero_grad_ratio 持续很高；reward 停滞 | 数据难度失衡，几乎全对或全错，大量零梯度 prompt（第 4 章） | 难度过滤、课程学习、或 DAPO 动态采样（第 6 章） |
| 8 | completion_length 持续暴涨而 reward 不涨 | 长度偏差或长度 hacking（第 4、5 章） | Dr.GRPO 修正（去 1/\|o_i\|）+ DAPO 超长软惩罚（第 6 章） |
| 9 | entropy 快速崩塌；多样性丧失 | 对称 clip 压制低概率 token（第 6 章） | clip-higher（ε_high > ε_low）或加熵正则 |
| 10 | 语言质量悄悄退化（乱码/重复）但 loss 看起来正常 | β=0 且不监控 kl_to_ref，策略偏离初始模型而无察觉 | 即使不加 KL 惩罚也必须监控 kl_to_ref（见 8.4） |
| 11 | reward 虚高但实际回答质量差 | verifier 可被绕过（多个 boxed、硬编码测试输出），reward hacking（第 4 章） | 加固抽取逻辑（见 8.2.1）+ 对抗性测试 |
| 12 | 正确答案被判错；reward 噪声大 | max_completion_length 太小导致截断，答案未完整生成（第 6 章） | 增大截断阈值、超长过滤或软惩罚；检查截断率 |
| 13 | 训练一开始就发散或 loss 极大 | 学习率沿用经典 RL 的 3e-4，破坏预训练权重（第 3 章） | LLM RL 微调用 1e-6 量级，降 1~2 个数量级 |
| 14 | 配置了 clip 但行为不符合预期 | 把 GRPO 的 clip（比率截断）与梯度裁剪 `clip_grad_norm_` 混淆 | 明确两者作用对象不同：前者截断重要性比率，后者限制梯度范数 |

<a id="s8-6"></a>

## 8.6 学习资源与进阶路径

<a id="s8-6-1"></a>

### 8.6.1 论文清单（按推荐阅读顺序）

以下论文均为本文档引用的核心文献。阅读顺序从基础到前沿，每篇标注了重点阅读内容。

1. **Sutton & Barto《Reinforcement Learning: An Introduction》第 13 章**：策略梯度基础。PPO 文档已覆盖，此处作为前置知识回顾。
2. **Schulman et al. 2017《Proximal Policy Optimization Algorithms》**：PPO 原论文。理解 clip 机制与 surrogate objective 的原始动机。
3. **Ahmadian et al. 2024《Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs》** (arXiv:2402.14740)：RLOO 方法，理解组内基线的最小版本，是 GRPO 的理论前驱。
4. **Shao et al. 2024《DeepSeekMath》** (arXiv:2402.03300)：**GRPO 原始出处**。重点读 GRPO 小节（§3.2 附近），理解从 PPO 到 GRPO 的简化动机。
5. **DeepSeek-AI 2025《DeepSeek-R1》** (arXiv:2501.12948)：RLVR 大规模实践与 aha moment。理解 GRPO 在实际系统中的表现与涌现现象。
6. **Yu et al. 2025《DAPO》** (arXiv:2503.14476)：四项工程改进（clip-higher、动态采样、token-level loss、超长塑形），长 CoT 训练必读。
7. **Liu et al. 2025《Understanding R1-Zero-Like Training: A Critical Perspective》** (arXiv:2503.20783)：Dr.GRPO，偏差分析必读。理解长度偏差与难度偏差的严格推导。
8. **MiniMax 2025《MiniMax-M1》** (arXiv:2506.13585)：CISPO 与长 CoT 系统。理解 detached IS 权重的设计动机。
9. **Zheng et al. 2025《Group Sequence Policy Optimization》** (arXiv:2507.18071)：GSPO，序列级比率与 MoE 稳定性。

<a id="s8-6-2"></a>

### 8.6.2 代码库

| 代码库 | 定位 | 适用场景 |
|--------|------|---------|
| **TRL** (huggingface/trl) | GRPOTrainer 文档与示例 | 入门、小模型实验 |
| **verl** (volcengine/verl) | DAPO 官方实现 | 大规模复现、生产级训练 |
| **OpenRLHF** (OpenRLHF/OpenRLHF) | 多算法 RLHF/RLVR 平台 | 算法对比研究 |
| **NeMo-RL** (NVIDIA/NeMo-RL) | 含 CISPO 实现指南 | CISPO 实验、NVIDIA 生态 |
| **AReaL** (inclusionAI/AReaL) | GRPO 系列算法文档 | 算法对照阅读、MoE/GSPO |
| **Unsloth** (unslothai/unsloth) | 单卡高效 GRPO | 消费级 GPU 微调 |
| **sail-sg/understand-r1-zero** | Dr.GRPO 官方代码 | 偏差分析实验复现 |

<a id="s8-6-3"></a>

### 8.6.3 一条务实的学习路径

以下路径强调动手与"主动破坏实验"，这是理解每个设计为何存在的最有效方式。

**第一步：跑通基线**。用 TRL 的 GRPOTrainer 在小模型（0.5B~1.5B）+ 简单可验证任务（如 GSM8K 子集或自定义算术题）上跑通，确认 reward 上升。记录初始的 reward、accuracy、completion_length、entropy 曲线。

**第二步：代码对照**。逐行对照 8.2 节的实现，在框架源码中找到优势计算、比率计算、KL 计算、损失聚合四处，与第 3、5 章公式一一对应。确认框架使用的是 seq_mean 还是 token_mean，KL 是在损失内还是奖励塑形中。

**第三步：主动破坏**。逐个修改以下设置，观察训练曲线如何崩坏，用 8.4 的指标定位原因：
- 把优势归一化改成全 batch 统计（坑 #1）
- 把 `scale_rewards` 开关翻转（对比 GRPO vs Dr.GRPO）
- 把 G 降到 2（坑 #5）
- 把 old_logprob 改成事后重算（坑 #2）
- 把 loss_type 从 token_mean 改成 seq_mean（坑 #4）

每次只改一个变量，记录指标变化。这种"控制变量法"比读十遍论文更能建立直觉。

**第四步：观察长度动力学**。记录 completion_length 曲线，尝试引入 Dr.GRPO 修正与 DAPO 超长塑形，对比长度是否受控。特别关注 reward 与长度的相关性变化。

**第五步：进阶实验**。在 verl 上复现 DAPO 的四项技术开关消融；或实现 GSPO 的序列级比率，对比同设置下的稳定性与 MoE 训练行为。

<a id="s8-7"></a>

## 8.7 本章小结与全文档收尾

本章完成了从理论到工程的最后一公里。以下是核心要点回顾：

1. **框架选型有明确分层**：学习用 TRL，生产用 verl，特定算法看对应代码库。不要试图自己从零搭建生产级训练系统。
2. **old_logprob 的正确获取是最高优先级**：它决定了比率 $`r`$ 的起点是否为 1，进而决定了 clip 行为是否正确。这是 GRPO 实现中最容易出错、后果最严重的环节。
3. **优势必须按 prompt 分组计算**：在整个 batch 上统计是最常见的 bug，会让不同 prompt 的奖励互相污染基线，训练信号完全错乱。
4. **损失聚合方式直接影响长度偏差**：seq_mean 保留长度偏差，token_mean 消除长度偏差。选择哪种取决于是否需要 Dr.GRPO 修正。
5. **β=0 时必须监控 kl_to_ref**：没有 KL 惩罚不等于不需要 KL 监控。策略偏离初始模型的程度必须有独立指标追踪。
6. **诊断指标是调试的核心工具**：GRPO 没有 explained_variance，取而代之的是 zero_grad_ratio、std_group、completion_length 等特有指标。掌握 8.4 节的排查顺序清单，能在大多数问题发生时快速定位根因。

<a id="h261"></a>

### 学完这两份文档之后该做什么

RLVR 领域的知识半衰期很短——新算法、新技巧、新框架每月都在涌现。试图记住所有变体的细节是不现实的。真正可迁移的能力有三样：

**(a) 策略梯度与基线理论的数学直觉**。PPO 文档第 3、4 章建立了策略梯度的完整推导，本文档第 2 章将其特化到组内基线。有了这个基础，任何新方法的"改了什么"都能在一页纸内说清楚：改了优势估计？改了 clip 机制？改了损失聚合？改了采样策略？所有变体都是这四个维度的组合。

**(b) 用诊断指标定位问题的能力**。8.4 节的指标体系和排查清单不是死记硬背的条目，而是一种思维方式：当训练曲线异常时，能从指标反推到算法层面的原因，而不是盲目调参。这种能力在任何 RL 系统中都通用。

**(c) 用第 7 章的四类框架快速读懂新变体的能力**。第 7 章给出了"改优势/改 clip/改粒度/改采样"的分类框架。遇到一篇新论文时，先用这个框架定位它的改动维度，再深入读细节。这比从头到尾线性阅读效率高一个数量级。

有了这三样能力，任何新算法都能在一小时内定位其改动与动机。剩下的就是动手实验——跑通、破坏、修复、再跑通。理论和工程之间的鸿沟，只有在反复的实践中才能真正弥合。祝你在 LLM RL 的探索之路上走得扎实、走得远。
