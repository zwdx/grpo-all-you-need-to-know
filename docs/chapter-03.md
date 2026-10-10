[上一章](chapter-02.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-04.md)

---

<a id="chapter-3"></a>

# 第 3 章 GRPO 算法本体

**本章目录**

- [3.1 符号与设定](#user-content-s3-1)
- [3.2 GRPO 完整目标函数](#user-content-s3-2)
- [3.3 KL 项的无偏估计器](#user-content-s3-3)
- [3.4 GRPO 与 PPO 的逐项对照](#user-content-s3-4)
- [3.5 完整算法伪代码](#user-content-s3-5)
- [3.6 计算与显存分析](#user-content-s3-6)
- [3.7 一个极简的数值算例](#user-content-s3-7)
- [3.8 本章小结与自测](#user-content-s3-8)

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

<details>
<summary>查看参考答案</summary>

参考答案：$`\mathbb{E}_{y \sim \pi_\theta}[\pi_{ref}/\pi_\theta] = \sum_y \pi_\theta \cdot (\pi_{ref}/\pi_\theta) = \sum_y \pi_{ref} = 1`$；$`\mathbb{E}_{y \sim \pi_\theta}[\log(\pi_{ref}/\pi_\theta)] = -\mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]`$；故期望 $`= 1 - (-\text{KL}) - 1 = \text{KL}`$。

</details>

**题目 2**：证明 k3 估计器恒非负，并说明这一性质在用作损失函数惩罚项时的重要性。

<details>
<summary>查看参考答案</summary>

参考答案：令 $`u = \log r`$，则 $`\hat{D}_{KL}^{(k3)} = e^u - u - 1`$。由 $`e^u \geq u+1`$（凸函数切线不等式），得 $`e^u - u - 1 \geq 0`$。非负性保证 KL 惩罚项不会产生负值从而变成"奖励"，避免策略被错误地推向远离参考模型的方向。

</details>

**题目 3**：当内层 epoch 数 $`K=1`$ 时，GRPO 的 clip 机制是否生效？为什么？此时 GRPO 等价于什么算法？

<details>
<summary>查看参考答案</summary>

参考答案：不生效。$`K=1`$ 时 $`\theta = \theta_{old}`$，故 $`r_{i,t} = \pi_\theta/\pi_{\theta_{old}} = 1`$，$`\text{clip}(1, 1-\varepsilon, 1+\varepsilon) = 1`$，min 的两个参数相等，clip 不起任何作用。此时 GRPO 等价于带组内基线的 REINFORCE + KL 惩罚。

</details>

**题目 4**：GRPO 目标函数中的 $`\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}`$ 归一化会导致什么问题？为什么？（简要回答即可，详细分析在第 5 章。）

<details>
<summary>查看参考答案</summary>

参考答案：导致长度偏差。短回答的每个 token 获得的梯度幅度大于长回答（因为总信号被更少的 token 分摊），使得模型倾向于生成较短的回答。

</details>

**题目 5**：比较 KL 项"写在损失函数内"与"塑形进奖励"两种方式在梯度传播路径上的区别。

<details>
<summary>查看参考答案</summary>

参考答案：损失内 KL 的梯度 $`-\beta \nabla_\theta D_{KL}`$ 直接反传，不经过任何中间变换。奖励塑形 KL 要先经 $`\tilde{r} = r - \beta \cdot KL`$，再经基线减除和归一化变成优势 $`\hat{A}`$，最后进入 surrogate 目标的梯度。后者中 $`\beta`$ 的有效量级被优势尺度调制，约束强度难以精确控制。

</details>

**题目 6**：设 $`G=4`$，某 prompt 的奖励为 $`r=[1,1,1,0]`$。手算归一化优势 $`\hat{A}`$，并解释为什么错误回答的惩罚绝对值大于正确回答的奖励绝对值。

<details>
<summary>查看参考答案</summary>

参考答案：$`\bar{r}=0.75`$，$`\sigma=\sqrt{3}/4 \approx 0.433`$，$`\hat{A} \approx [0.577, 0.577, 0.577, -1.732]`$。因为组内归一化要求 $`\sum \hat{A}_i = 0`$，三个正优势之和必须被一个负优势平衡，故负优势的绝对值是正优势的三倍。这是数学必然，也使得"避免错误"的信号强于"鼓励正确"。

</details>

---

[上一章](chapter-02.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-04.md)
