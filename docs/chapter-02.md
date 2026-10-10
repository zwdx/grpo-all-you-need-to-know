[上一章](chapter-01.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-03.md)

---

<a name="chapter-2"></a>

# 第 2 章 组内基线的理论基础

**本章目录**

- [2.1 从 PPO 文档的基线理论出发](#s2-1)
- [2.2 蒙特卡洛基线：$V^\pi(q)$ 的采样估计](#s2-2)
- [2.3 留一法基线（Leave-One-Out）与 RLOO](#s2-3)
- [2.4 GRPO 用的是哪一种基线](#s2-4)
- [2.5 除以标准差的含义与代价](#s2-5)
- [2.6 与 Critic 基线的系统性对比](#s2-6)
- [2.7 GAE 在 GRPO 中的命运](#s2-7)
- [2.8 采样数 $G$ 的选取：数学分析](#s2-8)
- [2.9 本章小结与自测](#s2-9)

<a name="s2-1"></a>

## 2.1 从 PPO 文档的基线理论出发

在 PPO 文档中（见 PPO 文档第 3 章），我们已经建立了关于策略梯度基线（baseline）的两个基本事实。为避免重复推导，此处仅做简要回顾，随后提出本章要回答的核心问题。

<a name="s2-1-1"></a>

### 2.1.1 基线的两个基本事实

**事实 (a)：基线的无偏性。** 对于任意只依赖状态 $s$ 而不依赖动作 $a$ 的函数 $b(s)$，从策略梯度估计量中减去它不改变梯度的期望值。即：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s)\right] = b(s) \cdot \mathbf{0} = \mathbf{0}
$$

其中最后一步使用了似然比技巧（likelihood ratio trick）的核心结论 $\mathbb{E}[\nabla_\theta \log \pi_\theta] = \mathbf{0}$（见 PPO 文档第 1 章）。因此，$\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (G - b(s))] = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G]$，减去基线不影响梯度期望的无偏性。

**事实 (b)：最优基线的形式。** 使梯度估计方差最小的标量基线为：

$$
b^*(s) = \frac{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 \cdot G\right]}{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\right]}
$$

这是被梯度范数平方加权的回报期望，而非简单的 $\mathbb{E}[G]$。实践中，由于 $b^*(s)$ 难以精确计算，通常用状态价值函数 $V^\pi(s) = \mathbb{E}[G \mid s]$ 作为近似替代。PPO 正是通过训练一个 critic 网络来拟合 $V^\pi(s)$，进而构造广义优势估计（GAE）。

<a name="s2-1-2"></a>

### 2.1.2 本章要回答的问题

在 LLM 强化学习场景中，状态 $s$ 对应 prompt $q$（加上已生成的前缀），而一条完整回答 $o$ 就是一条完整轨迹。如第 1 章所述，outcome reward 设定下所有 token 共享同一个 reward-to-go，这意味着 $V^\pi(q)$——从 prompt $q$ 出发的期望回报——恰好就是"该 prompt 下模型生成回答的平均质量"。

这自然引出一个关键问题：**$V^\pi(q)$ 这个量能不能不靠训练一个神经网络来得到，而靠采样直接估计？如果能，代价是什么？**

具体而言，我们想知道：

1. 用采样均值代替训练的 critic，梯度估计的偏差和方差分别如何变化？
2. 采样估计引入的自相关（self-inclusion）问题有多严重？能否消除？
3. 这种替代在什么条件下是合理的，在什么条件下会失效？

本章将逐一回答这些问题。

<a name="s2-2"></a>

## 2.2 蒙特卡洛基线：$V^\pi(q)$ 的采样估计

<a name="s2-2-1"></a>

### 2.2.1 LLM RL 中的价值函数与采样估计

在 LLM RL 的 MDP 框架中（见 PPO 文档第 1 章及本文档第 1 章），prompt $q$ 充当初始状态 $s_0$，语言模型的生成过程构成一条完整轨迹 $o = (y_1, y_2, \ldots, y_T)$，奖励 $R(q, o)$ 通常在序列末尾给出（outcome reward）。因此，状态价值函数的定义简化为：

$$
V^\pi(q) = \mathbb{E}_{o \sim \pi(\cdot|q)}[R(q, o)]
$$

这里的关键观察是：因为整条回答 $o$ 是从策略 $\pi(\cdot|q)$ 一次性采样得到的完整对象，$V^\pi(q)$ 就是对"该 prompt 下回答质量的期望"的一个简洁表达。不需要像传统 RL 那样对每个中间状态分别估计价值。

给定当前策略 $\pi_\theta$，我们可以从 $\pi_\theta(\cdot|q)$ 中独立采样 $G$ 个回答 $\{o_1, o_2, \ldots, o_G\}$，并构造 $V^\pi(q)$ 的蒙特卡洛估计量：

$$
\hat{V}_G(q) = \frac{1}{G} \sum_{i=1}^{G} R(q, o_i)
$$

记 $r_i = R(q, o_i)$ 为第 $i$ 个样本的奖励，则 $\hat{V}_G(q) = \frac{1}{G}\sum_{i=1}^G r_i$。

<a name="s2-2-2"></a>

### 2.2.2 无偏性证明

**命题 2.1**（$\hat{V}_G$ 的无偏性）。$\mathbb{E}[\hat{V}_G(q)] = V^\pi(q)$。

**证明：**

$$
\begin{aligned}
\mathbb{E}[\hat{V}_G(q)] &= \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} r_i\right] \\
&= \frac{1}{G} \sum_{i=1}^{G} \mathbb{E}[r_i] \quad \text{（期望的线性性）} \\
&= \frac{1}{G} \sum_{i=1}^{G} \mathbb{E}_{o_i \sim \pi(\cdot|q)}[R(q, o_i)] \quad \text{（每个 $o_i$ 独立同分布于 $\pi(\cdot|q)$）} \\
&= \frac{1}{G} \sum_{i=1}^{G} V^\pi(q) \\
&= V^\pi(q)
\end{aligned}
$$

证毕。$\square$

<a name="s2-2-3"></a>

### 2.2.3 方差分析

由于 $o_1, \ldots, o_G$ 相互独立，$\hat{V}_G$ 的方差为：

$$
\text{Var}[\hat{V}_G(q)] = \text{Var}\left[\frac{1}{G}\sum_{i=1}^G r_i\right] = \frac{1}{G^2}\sum_{i=1}^G \text{Var}[r_i] = \frac{\sigma^2(q)}{G}
$$

其中我们定义了核心量：

$$
\sigma^2(q) \triangleq \text{Var}_{o \sim \pi(\cdot|q)}[R(q, o)]
$$

$\sigma^2(q)$ 是**同一 prompt 下回答质量的内在方差**。它是本章后续所有分析的基石。直觉上：

- 当任务对模型来说很"简单"（几乎所有回答都正确）或很"难"（几乎所有回答都错误）时，$\sigma^2(q)$ 很小；
- 当任务恰好在模型能力边界附近（部分回答对、部分错），$\sigma^2(q)$ 很大；
- 对于二值奖励（$r \in \{0, 1\}$），若模型在该 prompt 上的正确率为 $p$，则 $\sigma^2(q) = p(1-p)$，在 $p = 0.5$ 时取最大值 $0.25$。

方差以 $1/G$ 速率衰减意味着：要将基线估计的标准差减半，需要将采样数 $G$ 增至 4 倍。这是一个经典的蒙特卡洛收敛速率。

<a name="s2-2-4"></a>

### 2.2.4 基线误差向梯度的传导

现在考虑用 $\hat{V}_G(q)$ 替代 $V^\pi(q)$ 后，梯度估计量的性质。定义优势估计量：

$$
\hat{A}_i = r_i - \hat{V}_G(q) = r_i - \frac{1}{G}\sum_{j=1}^G r_j
$$

对应的梯度估计量为：

$$
\hat{g} = \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i
$$

虽然 $\hat{V}_G$ 本身是 $V^\pi$ 的无偏估计，但 **$\hat{A}_i$ 作为整体用于梯度估计时是有偏的**。原因在于 $\hat{V}_G$ 中包含了 $r_i$ 自身——基线与被评估样本之间存在自相关（self-inclusion）。

为了看清这一点，将 $\hat{A}_i$ 展开：

$$
\hat{A}_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j = r_i - \frac{r_i}{G} - \frac{1}{G}\sum_{j \neq i} r_j = \frac{G-1}{G} r_i - \frac{1}{G}\sum_{j \neq i} r_j
$$

代入梯度估计量并取期望：

$$
\begin{aligned}
\mathbb{E}[\hat{g}] &= \frac{1}{G}\sum_{i=1}^G \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \left(\frac{G-1}{G} r_i - \frac{1}{G}\sum_{j \neq i} r_j\right)\right] \\
&= \frac{1}{G}\sum_{i=1}^G \left(\frac{G-1}{G} \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i] - \frac{1}{G}\sum_{j \neq i} \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q)] \cdot \mathbb{E}[r_j]\right)
\end{aligned}
$$

注意第二项中，由于 $o_i$ 与 $o_j$（$j \neq i$）独立，期望可以分解。又因 $\mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q)] = \mathbf{0}$，第二项整体为零。因此：

$$
\mathbb{E}[\hat{g}] = \frac{G-1}{G} \cdot \frac{1}{G}\sum_{i=1}^G \mathbb{E}[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i] = \frac{G-1}{G} \cdot \mathbb{E}[\nabla_\theta \log \pi_\theta(o|q) \cdot R(q,o)]
$$

而真实的策略梯度为 $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(o|q) \cdot R(q,o)]$。因此：

$$
\mathbb{E}[\hat{g}] = \frac{G-1}{G} \cdot \nabla_\theta J(\theta)
$$

**这表明全组均值基线导致的梯度估计存在一个 $(G-1)/G$ 的缩放偏差。** 我们将在 2.4 节详细讨论这个结果的深层含义。目前只需记住：自相关确实引入了偏差，但其形式是一个纯粹的常数缩放因子。

<a name="s2-3"></a>

## 2.3 留一法基线（Leave-One-Out）与 RLOO

<a name="s2-3-1"></a>

### 2.3.1 动机：消除自相关偏差

上一节的分析揭示了问题的根源：$\hat{V}_G$ 中包含 $r_i$ 自身，导致基线与被评估样本不独立。解决方案很自然——**在评估第 $i$ 个样本时，将其从基线计算中剔除**。这就是留一法（Leave-One-Out, LOO）基线的核心思想。

<a name="s2-3-2"></a>

### 2.3.2 留一法基线的定义

定义第 $i$ 个样本的留一法基线为：

$$
b_i^{\text{LOO}} = \frac{1}{G-1} \sum_{j \neq i} r_j
$$

对应的优势估计量为：

$$
\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}} = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j
$$

<a name="s2-3-3"></a>

### 2.3.3 无偏性的严格证明

**命题 2.2**（留一法梯度估计的无偏性）。定义梯度估计量 $\hat{g}^{\text{LOO}} = \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}$，则 $\mathbb{E}[\hat{g}^{\text{LOO}}] = \nabla_\theta J(\theta)$。

**证明：** 由期望的线性性，只需证明对每个 $i$：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot R(q, o_i)\right]
$$

将 $\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}}$ 代入左边：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot (r_i - b_i^{\text{LOO}})\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i\right] - \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot b_i^{\text{LOO}}\right]
$$

关键在于第二项。注意 $b_i^{\text{LOO}} = \frac{1}{G-1}\sum_{j \neq i} r_j$ **只依赖于 $\{o_j\}_{j \neq i}$**，而 $o_i$ 与 $\{o_j\}_{j \neq i}$ 相互独立（因为它们是从 $\pi(\cdot|q)$ 中独立采样的）。因此：

$$
\begin{aligned}
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot b_i^{\text{LOO}}\right] &= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbb{E}_{o_i}\left[\nabla_\theta \log \pi_\theta(o_i|q) \mid \{o_j\}_{j \neq i}\right]\right] \\
&= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbb{E}_{o_i}\left[\nabla_\theta \log \pi_\theta(o_i|q)\right]\right] \quad \text{（独立性）} \\
&= \mathbb{E}_{\{o_j\}_{j \neq i}}\left[b_i^{\text{LOO}} \cdot \mathbf{0}\right] \quad \text{（似然比技巧）} \\
&= \mathbf{0}
\end{aligned}
$$

因此：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot \hat{A}_i^{\text{LOO}}\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(o_i|q) \cdot r_i\right]
$$

对所有 $i$ 求平均即得 $\mathbb{E}[\hat{g}^{\text{LOO}}] = \nabla_\theta J(\theta)$。证毕。$\square$

**关键洞察：** 上述证明的核心条件是 $b_i^{\text{LOO}}$ 与 $o_i$ 的**独立性**。这不是充分条件的弱化版本，而是充要条件：任何与 $o_i$ 独立的基线都不会引入偏差；任何包含 $o_i$ 信息的基线（如全组均值）都会引入偏差。留一法恰好是"使用尽可能多的样本信息同时保持独立性"的最优选择。

<a name="s2-3-4"></a>

### 2.3.4 留一法基线的方差

$b_i^{\text{LOO}}$ 基于 $G-1$ 个独立样本的均值，因此其方差为：

$$
\text{Var}[b_i^{\text{LOO}}] = \frac{\sigma^2(q)}{G-1}
$$

与之对比，全组均值基线的方差为 $\sigma^2(q)/G$。留一法基线的方差略大，比值为 $G/(G-1)$。当 $G=8$ 时，比值为 $8/7 \approx 1.14$；当 $G=64$ 时，比值为 $64/63 \approx 1.016$。这是为无偏性付出的代价——在实践中几乎可以忽略。

更完整地看，留一法梯度估计量的方差涉及 $\hat{A}_i^{\text{LOO}} = r_i - b_i^{\text{LOO}}$ 的方差。由于 $r_i$ 与 $b_i^{\text{LOO}}$ 独立：

$$
\text{Var}[\hat{A}_i^{\text{LOO}}] = \text{Var}[r_i] + \text{Var}[b_i^{\text{LOO}}] = \sigma^2(q) + \frac{\sigma^2(q)}{G-1} = \sigma^2(q) \cdot \frac{G}{G-1}
$$

相比之下，全组均值优势 $\hat{A}_i = r_i - \hat{V}_G$ 的方差为（利用 $r_i$ 与 $\hat{V}_G$ 的协方差 $\text{Cov}(r_i, \hat{V}_G) = \sigma^2/G$）：

$$
\text{Var}[\hat{A}_i] = \text{Var}[r_i] + \text{Var}[\hat{V}_G] - 2\text{Cov}(r_i, \hat{V}_G) = \sigma^2 + \frac{\sigma^2}{G} - \frac{2\sigma^2}{G} = \sigma^2 \cdot \frac{G-1}{G}
$$

有趣的是，全组均值优势的方差反而**更小**（因子 $(G-1)/G < G/(G-1)$），但这正是以偏差为代价换来的。这构成了一个经典的偏差-方差权衡（bias-variance tradeoff）。

<a name="s2-3-5"></a>

### 2.3.5 RLOO：REINFORCE Leave-One-Out

RLOO（REINFORCE Leave-One-Out）由 Ahmadian et al. (2024) 在论文《Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs》(arXiv:2402.14740) 中提出。它的定义极为简洁：

$$
\nabla_\theta J_{\text{RLOO}}(\theta) \approx \frac{1}{G}\sum_{i=1}^G \nabla_\theta \log \pi_\theta(o_i|q) \cdot \left(r_i - \frac{1}{G-1}\sum_{j \neq i} r_j\right)
$$

RLOO 本质上就是 **REINFORCE + 留一法基线**：没有重要性比率（importance ratio）、没有 clip 操作、没有 KL 散度正则项。它是 GRPO 的"极简近亲"。

**RLOO 与 GRPO 的关系可以用一个公式概括：**

$$
\text{GRPO} \approx \text{RLOO} + \text{PPO 式 clip} + \text{KL 正则项} + \text{除以组内 std}
$$

具体而言：
- RLOO 提供无偏的优势估计基础；
- GRPO 在此基础上加入了 PPO 风格的 clip 机制以限制策略更新幅度；
- GRPO 加入了 KL 散度项以防止策略偏离参考模型过远；
- GRPO 额外除以了组内标准差进行归一化（这一点的理论后果将在 2.5 节分析）。

在实践中，RLOO 在 TRL 库中有独立实现，且在多个基准上与 GRPO 效果接近。这说明 PPO 式的 clip 和 KL 项虽然有用，但并非 GRPO 成功的唯一原因——**留一法基线本身就提供了足够好的梯度信号**。

<a name="s2-4"></a>

## 2.4 GRPO 用的是哪一种基线

<a name="s2-4-1"></a>

### 2.4.1 原论文的明确选择

这是一个很多资料混淆的关键点。**DeepSeekMath 原论文（Shao et al., 2024）中定义的 GRPO 使用的是全组均值基线，而非留一法基线。** 原论文的优势定义为：

$$
\hat{A}_i^{\text{GRPO}} = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

其中 $\text{mean}$ 和 $\text{std}$ 均基于全部 $G$ 个样本计算。因此，严格来说，原始 GRPO 的优势估计是**有偏的**，偏差量级为 $O(1/G)$（如 2.2.4 节所示）。

<a name="s2-4-2"></a>

### 2.4.2 偏差的实践影响

这个偏差在实践中影响有限，原因有二：

1. **$G$ 通常足够大。** 实践中 $G$ 取值在 8 到 64 之间，$(G-1)/G$ 的范围是 $0.875$ 到 $0.984$，相对于梯度估计本身的随机噪声而言微不足道。
2. **clip 机制本身也引入偏差。** PPO 的 clip 操作是一种有偏的梯度修正，GRPO 继承了这一特性。在这种背景下，基线的 $O(1/G)$ 偏差并不构成额外的理论负担。

<a name="s2-4-3"></a>

### 2.4.3 论文与实现的差异

需要注意的是，部分现代实现提供了不同的基线选项。例如，TRL 库的某些配置支持 `scale_rewards` 参数和不同的 baseline 类型选择，有些实现确实采用了留一法变体。读者在阅读代码时应注意核对具体实现与原论文的差异。

<a name="s2-4-4"></a>

### 2.4.4 偏差的定量表达：$(G-1)/G$ 缩放定理

现在我们给出 2.2.4 节结论的一个更清晰的表述。

**命题 2.3**（全组均值优势与留一法优势的精确关系）。在不除以 std 的前提下，全组均值基线得到的优势恰好是留一法优势的 $(G-1)/G$ 倍：

$$
r_i - \text{mean}(\{r_1, \ldots, r_G\}) = \frac{G-1}{G} \left(r_i - b_i^{\text{LOO}}\right)
$$

**证明：** 首先写出全组均值与留一法均值的关系：

$$
\text{mean}(\{r_1, \ldots, r_G\}) = \frac{1}{G}\sum_{j=1}^G r_j = \frac{1}{G}\left(r_i + \sum_{j \neq i} r_j\right) = \frac{r_i}{G} + \frac{G-1}{G} \cdot \frac{1}{G-1}\sum_{j \neq i} r_j = \frac{r_i}{G} + \frac{G-1}{G} \cdot b_i^{\text{LOO}}
$$

因此：

$$
\begin{aligned}
r_i - \text{mean} &= r_i - \frac{r_i}{G} - \frac{G-1}{G} \cdot b_i^{\text{LOO}} \\
&= \frac{G-1}{G} \cdot r_i - \frac{G-1}{G} \cdot b_i^{\text{LOO}} \\
&= \frac{G-1}{G} \left(r_i - b_i^{\text{LOO}}\right)
\end{aligned}
$$

证毕。$\square$

**这个结论的含义非常深刻：** 全组均值基线相对于留一法基线的偏差，表现为一个纯粹的常数缩放因子 $(G-1)/G$。在梯度层面，这等价于将学习率乘以 $(G-1)/G$，而**完全不改变梯度的方向**。换句话说，「GRPO 的基线偏差」实际上只是一个微小的等效学习率偏移，而非方向性错误。

**重要限制条件：** 上述等价性仅在以下条件下严格成立：
1. **不除以 std**：一旦引入 $\text{std}(\{r_1, \ldots, r_G\})$ 作为分母，由于 std 本身也依赖于 $r_i$，变换变为非线性，常数缩放的论证不再适用。
2. **不做 clip**：clip 操作是非线性的，$(G-1)/G$ 的缩放可能在 clip 的边界处产生不同的截断行为。

因此，完整的 GRPO（含 std 归一化和 clip）的偏差分析比命题 2.3 更复杂。但命题 2.3 仍然提供了一个有价值的基准参照：在最简情形下，偏差的本质是良性的。

<a name="s2-5"></a>

## 2.5 除以标准差的含义与代价

<a name="s2-5-1"></a>

### 2.5.1 数学含义：逐 prompt 的自适应尺度归一化

GRPO 的优势定义为：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
$$

从统计学角度看，这是对组内奖励进行**标准化（standardization）**：减去均值再除以标准差，使得不同 prompt 的优势分布近似统一到零均值、单位方差的尺度上。如果组内奖励近似正态分布，标准化后的优势约服从 $\mathcal{N}(0, 1)$。

<a name="s2-5-2"></a>

### 2.5.2 好处

这种归一化带来两个实际好处：

1. **跨 prompt 梯度尺度一致。** 不同 prompt 的奖励方差可能差异巨大：一个简单的算术题可能所有回答都得 1 分（$\sigma \approx 0$），而一道竞赛题的回答质量参差不齐（$\sigma$ 较大）。不归一化时，前者的梯度贡献极小，后者主导训练。归一化后，每个 prompt 的梯度贡献量级相当。
2. **隐式梯度归一化。** 在小 batch 设置下，除以 std 相当于对每个 prompt 的梯度做了自适应缩放，有助于稳定训练动态，减少对全局学习率的敏感性。

<a name="s2-5-3"></a>

### 2.5.3 代价：难度偏差与非线性破坏

除以 std 的代价同样显著：

**第一，破坏了常数缩放论证。** 如 2.4.4 节所述，在不除 std 时，全组均值基线与留一法基线只差一个常数因子。但 std 的计算也包含 $r_i$：

$$
\hat{\sigma} = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}
$$

这使得 $\hat{A}_i = (r_i - \bar{r})/\hat{\sigma}$ 成为 $r_i$ 的非线性函数。命题 2.3 的简洁结论不再成立，偏差分析变得更加复杂。

**第二，引入难度偏差（difficulty bias）。** 考虑 $\hat{A}_i$ 的量级与 $\hat{\sigma}$ 的关系：

$$
|\hat{A}_i| \propto \frac{|r_i - \bar{r}|}{\hat{\sigma}(q)}
$$

当 $\hat{\sigma}(q)$ 很小时（即该 prompt 对模型来说太简单或太难，组内奖励几乎全同），$\hat{A}_i$ 会被放大；当 $\hat{\sigma}(q)$ 很大时（模型在该 prompt 上表现不稳定），$\hat{A}_i$ 会被压缩。这意味着：

- **"极端难度"的 prompt 获得过大权重**：这些 prompt 本应提供的训练信号最弱（因为模型要么已经掌握、要么完全不会），却被赋予了最大的梯度贡献。
- **"适中难度"的 prompt 被压制**：这些 prompt 恰恰是最有训练价值的（模型处于学习前沿），却被归一化缩小了贡献。

这个偏差的完整分析与修正方案将在第 5 章（Dr.GRPO）中详细讨论。此处只需理解其数学机理。

**第三，数值不稳定性。** 当 $\hat{\sigma}(q) \to 0$ 时，$\hat{A}_i$ 趋向无穷大。实践中必须加入保护项：

$$
\hat{A}_i = \frac{r_i - \bar{r}}{\hat{\sigma} + \epsilon}
$$

其中 $\epsilon$ 通常取 $10^{-8}$ 量级。但当 $\hat{\sigma}$ 本身就在 $\epsilon$ 量级时，这个保护并不能真正解决问题——它只是把爆炸的上限从无穷大变成了一个很大的有限值。

<a name="s2-5-4"></a>

### 2.5.4 全同奖励下的零梯度现象

一个特别重要的特殊情况是：**当组内所有奖励完全相同时**（例如二值奖励下全部答对或全部答错），$\hat{\sigma} = 0$ 且 $r_i - \bar{r} = 0$，此时所有 $\hat{A}_i = 0$（即使不加 $\epsilon$ 保护，$0/0$ 在实践中也被处理为 0）。

**这意味着该 prompt 对梯度的贡献完全为零。**

这个现象的训练动力学后果是深远的：随着模型逐渐变强，越来越多的 prompt 变成"全对"状态，这些 prompt 不再提供任何训练信号。梯度信号的有效来源逐渐枯竭，训练效率下降。这正是第 6 章 DAPO 引入"动态采样"（dynamic sampling）要解决的核心问题——通过过滤掉全同奖励的 prompt 并补充新的有区分度的样本，维持训练信号的持续供给。此处我们只需记住这个数学事实及其直觉含义。

<a name="s2-6"></a>

## 2.6 与 Critic 基线的系统性对比

本节是本章的核心结论章节。我们从多个维度系统对比「组内采样基线」与「训练的 critic 基线」。

| 维度 | 训练的 Critic 基线 | 组内采样基线（GRPO/RLOO） |
|------|-------------------|------------------------|
| **偏差** | 因函数近似而有偏；偏差大小取决于 critic 网络的容量、训练数据和收敛程度，难以量化控制 | LOO 形式下严格无偏；全组均值形式下有 $O(1/G)$ 的缩放偏差（命题 2.3），可控且良性 |
| **方差** | 来自 critic 自身的估计噪声，随训练过程动态变化，难以预测 | $\sigma^2(q)/G$（LOO 为 $\sigma^2(q)/(G-1)$），形式确定，可通过增大 $G$ 精确控制 |
| **自适应性** | **状态相关**：能对每个 token 位置给出不同的 $V(s_t)$，理论上支持精细的信用分配 | **序列级**：整条回答共享同一个基线值，无法区分序列内部不同位置的贡献 |
| **计算成本** | 需要额外的 critic 模型：训练开销（反向传播）、推理开销（前向传播）、显存占用（模型参数+优化器状态） | 仅需 $G$ 倍采样推理，无额外模型、无训练开销；代价是 $G$ 倍的生成算力 |
| **泛化能力** | 作为函数逼近器，能对未见过的状态给出价值估计，具有外推能力 | 完全依赖当前 batch 的采样，无任何泛化能力；每个新 batch 都必须重新采样，无法复用历史数据 |
| **收敛性与稳定性** | Actor-critic 耦合系统：critic 训练不稳定会通过优势估计传导至 actor，可能导致训练崩溃 | 无反馈回路：基线估计独立于任何可学习参数，不存在耦合不稳定性；但信号更粗糙 |
| **数据效率** | 可利用 off-policy 数据更新 critic，跨 epoch 复用采样数据（PPO 的多 epoch 训练） | 严格的 on-policy：每次策略更新后必须重新采样，旧数据的基线估计失效 |

<a name="s2-6-1"></a>

### 2.6.1 自适应性的深入讨论

表格中"自适应性"一行是**最本质的区别**，需要展开讨论。

Critic 的核心优势在于它是状态相关的：对于序列中的每个前缀 $s_t = (q, y_1, \ldots, y_{t-1})$，critic 能给出不同的价值估计 $V(s_t)$。这使得 GAE 能够进行精细的信用分配——区分"哪个 token 做得好、哪个做得差"。

然而，如第 1 章所述，在 outcome reward 设定下，奖励只在序列末尾给出，所有中间步的即时奖励为零。此时 reward-to-go 对所有 token 都是相同的（等于整个序列的奖励 $R(q, o)$）。在这种特殊结构下，真实的状态价值函数 $V^\pi(s_t)$ 对所有前缀 $s_t$ 近似相等（都约等于 $V^\pi(q)$），因为无论从哪个前缀出发，最终的期望回报都由完整回答的质量决定。

**这意味着：在 outcome reward 设定下，critic 的状态相关性优势被大幅削弱。** Critic 实际上退化为一个近似常数函数，其提供的信息与序列级基线相差无几。这正是 GRPO 能在 LLM RL 中成功的深层理论原因——它放弃了一个在当前设定下本就价值有限的能力，换来了巨大的计算和工程简化。

**但这个论证是有条件的。** 如果奖励是 process-level 的（即每个 token 或每步都有独立的奖励信号，详见第 4 章），那么不同前缀的 reward-to-go 确实不同，critic 的状态相关性就有了实质价值。在这种情况下，GRPO 的序列级基线假设不再成立，使用 critic 可能是必要的。这个条件依赖关系是理解 GRPO 适用范围的关键。

<a name="s2-6-2"></a>

### 2.6.2 结论性判断

综合以上分析，我们给出以下判断：

> **在 outcome reward + 短到中等长度 + 有可靠验证器（verifier）的 LLM 推理任务上，组内采样基线优于训练的 critic 基线。** 这是因为在此设定下 critic 的状态相关性优势被削弱，而其计算成本和训练不稳定性代价却依然存在。
>
> **在 process reward、奖励计算昂贵、或无法多次采样的场景下，critic 仍是必要的。** 此时序列级基线的假设不成立，或者 $G$ 倍采样的成本不可承受。

<a name="s2-7"></a>

## 2.7 GAE 在 GRPO 中的命运

<a name="s2-7-1"></a>

### 2.7.1 GAE 的回顾

在 PPO 文档第 4 章中，广义优势估计（Generalized Advantage Estimation, GAE）定义为：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，$\gamma$ 是折扣因子，$\lambda \in [0, 1]$ 控制偏差-方差权衡。

<a name="s2-7-2"></a>

### 2.7.2 GRPO 设定下的退化

在 GRPO 的典型设定中：

1. **$\gamma = 1$**：LLM 生成任务是有限步的，不需要折扣。
2. **奖励只在末尾给出**：$r_t = 0$ 对所有 $t < T$，$r_T = R(q, o)$。
3. **没有训练的 critic**：GRPO 不使用 $V(s_t)$。

在这些条件下考察 TD 残差。对于中间步 $t < T$：

$$
\delta_t = 0 + V(s_{t+1}) - V(s_t)
$$

如果 $V$ 近似为常数（如 2.6.1 节论证的，在 outcome reward 下这是合理的），则 $\delta_t \approx 0$。所有信息集中在最后一步：

$$
\delta_T = R(q, o) - V(s_T)
$$

因此 GAE 退化为：$\hat{A}_t^{\text{GAE}} \approx (\gamma\lambda)^{T-t} \cdot \delta_T$。当 $\gamma = 1$ 且 $\lambda = 1$ 时，所有时间步的优势都等于 $\delta_T = R(q, o) - V(s_T)$，即一个与 $t$ 无关的常数。

**这正是 GRPO 的做法：直接定义序列级常量优势 $\hat{A}_{i,t} = \hat{A}_i$ 对所有 $t$。** GRPO 丢弃 GAE 不是一个随意的简化，而是在 outcome reward 设定下 GAE 自然退化的结果。

<a name="s2-7-3"></a>

### 2.7.3 代价：信用分配的丧失

这个简化的理论代价是放弃了**信用分配（credit assignment）** 能力。GRPO 无法区分一条回答中"哪个 token 做得好、哪个做得差"，只能对整条回答统一奖惩。

在短回答（几十到几百 token）的场景下，这个问题不太严重。但在长链推理（long Chain-of-Thought, long-CoT）中，一条回答可能包含上万 token，其中只有少数关键步骤决定了最终答案的正确性。GRPO 对所有 token 赋予相同的优势信号，相当于把关键步骤的信号稀释到了大量无关 token 上，降低了学习效率。

后续的研究方向——包括过程奖励模型（Process Reward Model, PRM）、token 级信用分配方法——正是在弥补这个理论缺口。这些内容将在第 4 章和第 7 章中进一步讨论。

<a name="s2-8"></a>

## 2.8 采样数 $G$ 的选取：数学分析

<a name="s2-8-1"></a>

### 2.8.1 边际收益递减

从 $\text{Var}[\hat{V}_G] = \sigma^2/G$ 出发，$G$ 的边际收益遵循经典的 $1/G$ 规律：

- $G$ 从 4 增到 8：方差减半，标准差降至原来的 $1/\sqrt{2} \approx 0.71$；
- $G$ 从 8 增到 16：方差再减半，但绝对改善量小于上一档；
- $G$ 从 64 增到 128：方差同样减半，但计算成本翻倍，性价比显著降低。

这意味着 $G$ 的选择存在一个实用饱和点，超过该点后增大 $G$ 的收益不足以抵消成本。

<a name="s2-8-2"></a>

### 2.8.2 小 $G$ 的统计风险

当 $G$ 过小时，std 的估计极不可靠。对于正态分布样本，样本标准差 $\hat{\sigma}$ 的相对标准误（coefficient of variation）约为：

$$
\frac{\text{SE}[\hat{\sigma}]}{\sigma} \approx \frac{1}{\sqrt{2(G-1)}}
$$

当 $G = 2$ 时，相对标准误为 $1/\sqrt{2} \approx 0.707$，即 std 估计量的波动幅度与其真值同量级。当 $G = 4$ 时，相对标准误为 $1/\sqrt{6} \approx 0.408$，仍然很大。当 $G = 8$ 时，相对标准误为 $1/\sqrt{14} \approx 0.267$，开始进入可用范围。

这解释了为什么实践中 $G$ 很少低于 4：不仅是因为基线估计不准，更是因为 **std 归一化在小 $G$ 下极度不稳定**。

<a name="s2-8-3"></a>

### 2.8.3 实用区间

论文与实践中的常用范围：

- DeepSeekMath 使用 $G = 8$ 量级的采样；
- 多数后续工作使用 $G \in [8, 64]$；
- $G > 64$ 的情况较少见，通常只在奖励方差极大或需要极高精度时使用。

<a name="s2-8-4"></a>

### 2.8.4 $G$ 与 Prompt 多样性的权衡

在固定算力预算下，总采样数 $N = N_q \times G$（$N_q$ 为 prompt 数量）是固定的。这构成了一个权衡：

- **大 $G$、少 prompt**：每个 prompt 的优势估计更准确，但 prompt 多样性下降，batch 内的覆盖面窄，可能导致策略更新的泛化性不足。
- **小 $G$、多 prompt**：prompt 多样性高，覆盖面广，但每个 prompt 的优势估计噪声大，单步更新的方向可能不准。

实践中通常采取折中策略。此外，第 6 章介绍的 DAPO 动态采样技术改变了这个权衡：它允许在不同 prompt 上使用不同的有效 $G$（过滤掉全同奖励的 prompt 后补充新样本），从而在一定程度上解耦了 $G$ 与 prompt 多样性的矛盾。

<a name="s2-9"></a>

## 2.9 本章小结与自测

<a name="h73"></a>

### 本章要点

1. **蒙特卡洛基线的无偏性与方差**：$\hat{V}_G(q) = \frac{1}{G}\sum r_i$ 是 $V^\pi(q)$ 的无偏估计，方差为 $\sigma^2(q)/G$，其中 $\sigma^2(q)$ 是同一 prompt 下回答质量的内在方差。
2. **全组均值基线的自相关偏差**：用包含 $r_i$ 的全组均值作基线会引入 $(G-1)/G$ 的缩放偏差（命题 2.3），本质上是等效学习率偏移而非方向性错误，但该结论仅在不含 std 归一化和 clip 时严格成立。
3. **留一法基线的无偏性**：$b_i^{\text{LOO}}$ 与 $o_i$ 独立，保证了梯度估计的严格无偏性（命题 2.2），代价是方差略增为 $\sigma^2/(G-1)$。
4. **除以 std 的双刃剑效应**：实现了跨 prompt 的梯度尺度统一，但引入了难度偏差和非线性，破坏了常数缩放论证；全同奖励下梯度信号完全消失。
5. **组内基线 vs critic 的条件优越性**：在 outcome reward 设定下，critic 的状态相关性退化为近似常数，组内基线因此成为合理替代；在 process reward 设定下此论证不成立。
6. **GAE 的退化与信用分配的丧失**：在 outcome reward + $\gamma=1$ 下 GAE 退化为序列级常量，GRPO 据此丢弃 GAE，代价是无法进行 token 级信用分配。

<a name="h74"></a>

### 自测题

**题目 1.** 设 $G=8$，某 prompt 下模型回答的正确率 $p=0.5$（二值奖励），计算蒙特卡洛基线 $\hat{V}_G$ 的标准差。若要将该标准差降至 0.05 以下，至少需要多大的 $G$？

<details>
<summary>查看参考答案</summary>

参考答案：$\sigma^2 = p(1-p) = 0.25$，$\text{SD}[\hat{V}_8] = \sqrt{0.25/8} \approx 0.177$。要求 $\sqrt{0.25/G} < 0.05$，即 $G > 0.25/0.0025 = 100$，故至少需要 $G = 101$（取整数上界）。

</details>

**题目 2.** 证明命题 2.3 中的关系式 $r_i - \bar{r} = \frac{G-1}{G}(r_i - b_i^{\text{LOO}})$，并解释为什么这个关系在引入 std 归一化后不再成立。

<details>
<summary>查看参考答案</summary>

参考答案：证明见 2.4.4 节。引入 std 后，$\hat{A}_i = (r_i - \bar{r})/\hat{\sigma}$，而 $\hat{\sigma}$ 本身依赖于 $r_i$（通过 $(r_i - \bar{r})^2$ 项），使得 $\hat{A}_i$ 成为 $r_i$ 的非线性函数。常数缩放 $(G-1)/G$ 只对分子成立，对整个比值不成立，因为分母 $\hat{\sigma}$ 在全组均值和留一法两种情形下的值也不同。

</details>

**题目 3.** 在一个二值奖励的任务中，若某 prompt 的 8 个采样回答全部正确（$r_1 = \cdots = r_8 = 1$），GRPO 的优势 $\hat{A}_i$ 是多少？这对训练意味着什么？

<details>
<summary>查看参考答案</summary>

参考答案：$\bar{r} = 1$，$\hat{\sigma} = 0$，$r_i - \bar{r} = 0$，故 $\hat{A}_i = 0$（实践中 $0/\epsilon = 0$）。该 prompt 对梯度贡献为零，不提供任何训练信号。随着模型变强，越来越多 prompt 进入此状态，有效训练信号枯竭。

</details>

**题目 4.** 比较全组均值基线和留一法基线在 $G=8$ 时的方差比，并说明为什么实践中这个差异通常可以忽略。

<details>
<summary>查看参考答案</summary>

参考答案：$\text{Var}[b^{\text{LOO}}]/\text{Var}[\bar{r}] = G/(G-1) = 8/7 \approx 1.14$，即留一法方差仅大 14%。当 $G$ 更大时比值更接近 1。相对于梯度估计本身的高方差（来自策略采样的随机性），14% 的基线方差增量几乎不可察觉。

</details>

**题目 5.** 在什么条件下 GRPO 的序列级基线假设会失效？此时应该用什么替代方案？

<details>
<summary>查看参考答案</summary>

参考答案：当奖励是 process-level（每步/token 都有独立奖励信号）时，不同前缀的 reward-to-go 不同，序列级基线无法捕捉这种差异。此时应使用训练的 critic 提供状态相关的价值估计，或采用过程奖励模型配合 token 级信用分配方法。

</details>

**题目 6.** 解释为什么 GRPO 可以在 LLM RL 中成功丢弃 GAE，而在传统 RL（如机器人控制）中不行。关键的结构差异是什么？

<details>
<summary>查看参考答案</summary>

参考答案：LLM RL 通常使用 outcome reward（奖励仅在序列末尾给出），且 $\gamma=1$，此时所有 token 的 reward-to-go 相同，GAE 退化为序列级常量。传统 RL 中奖励分布在每一步（dense reward），不同状态的 value 差异显著，GAE 的信用分配功能不可或缺。关键结构差异是奖励的时间分布模式：sparse outcome reward vs. dense step reward。

</details>

---

[上一章](chapter-01.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-03.md)
