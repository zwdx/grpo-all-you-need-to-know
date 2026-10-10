[上一章](chapter-05.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-07.md)

---

<a id="chapter-6"></a>

# 第 6 章 DAPO：面向长链推理的四项关键改进

**本章目录**

- [6.1 DAPO 的定位与整体思路](#s6-1)
- [6.2 Clip-Higher：解耦上下裁剪阈值](#s6-2)
- [6.3 Dynamic Sampling：动态采样过滤无效 prompt](#s6-3)
- [6.4 Token-Level Policy Gradient Loss](#s6-4)
- [6.5 Overlong Reward Shaping：超长奖励塑形](#s6-5)
- [6.6 DAPO 移除 KL 项的决策](#s6-6)
- [6.7 DAPO 完整目标函数与算法](#s6-7)
- [6.8 DAPO 的工程系统与可复现性](#s6-8)
- [6.9 本章小结与自测](#s6-9)

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

$$
H(\pi_\theta(\cdot|s)) = -\sum_{a \in \mathcal{V}} \pi_\theta(a|s) \log \pi_\theta(a|s).
$$

当 $`\pi_\theta(\cdot|s)`$ 趋近 one-hot 分布（即某个动作的概率趋近 1，其余趋近 0）时，$`H \to 0`$。**熵崩塌（entropy collapse）** 指的就是训练过程中策略熵过快下降到极低水平的现象。其危害在于：

1. **探索丧失**：策略几乎只生成一个确定性的输出，无法发现可能更优的替代路径。
2. **性能停滞**：一旦陷入局部最优的确定性策略，后续训练无法改善。
3. **多样性退化**：对于需要多种解题策略的数学推理任务，单一模式意味着覆盖率的天花板。

在 long-CoT RL 训练中，熵崩塌尤其致命：推理路径的搜索空间极大，过早收敛意味着模型放弃了绝大部分潜在的推理链路。

<a id="s6-2-2"></a>

### 6.2.2 对称 clip 如何加速熵崩塌

PPO-Clip 和标准 GRPO 均使用对称裁剪阈值 $`\varepsilon_{low} = \varepsilon_{high} = \varepsilon`$（常用 $`\varepsilon = 0.2`$），即：

$$
\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) = \max(1-\varepsilon, \min(1+\varepsilon, r_t)).
$$

承接 PPO 文档第 6 章的六情形分析，当 $`\hat{A} > 0`$ 且 $`r_t > 1+\varepsilon`$ 时，$`\min`$ 操作选择被裁剪的分支，梯度为零——策略不再继续增大该 token 的概率。这个机制在防止策略更新过大方面功不可没，但它对**低概率好 token** 产生了系统性的压制效应。推理链条如下：

1. 考虑一个当前概率很低的 token $`a^*`$，即 $`\pi_\theta(a^*|s) \approx \delta`$，其中 $`\delta \ll 1`$。假设它是好 token（$`\hat{A} > 0`$），我们希望增大概率。
2. 重要性比率 $`r_t = \pi_\theta(a^*|s) / \pi_{\theta_{old}}(a^*|s)`$。由于 $`\pi_\theta(a^*|s)`$ 本身很小，即使绝对概率翻倍（从 $`\delta`$ 到 $`2\delta`$），$`r_t`$ 也只是从 1 变为 2——而 $`1+\varepsilon = 1.2`$ 的上界远在 2 之下。
3. 这意味着：低概率 token 只需极小的相对增幅就会触发 $`r_t > 1+\varepsilon`$ 的截断条件，梯度立即归零。
4. 相比之下，一个已经高概率的 token（$`\pi_\theta \approx 0.9`$），要从 0.9 涨到触发截断，需要 $`r_t > 1.2`$，即概率需涨到 $`0.9 \times 1.2 = 1.08`$——这不可能发生（概率上界为 1），所以高概率 token 实际上很少被上界截断。

**结论**：对称 clip 的上界对低概率 token 构成了不成比例的限制。低概率好 token 难以被"捞起来"，高概率 token 则不受影响。长期累积效应是：概率质量持续向已有的高概率 token 集中，低概率区域被系统性忽视，分布越来越尖锐，熵持续下降直至崩塌。

<a id="s6-2-3"></a>

### 6.2.3 Clip-Higher 的解法

DAPO 将上下裁剪阈值解耦（Decoupled Clip），定义非对称裁剪算子：

$$
\text{clip}_{asym}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high}) = \max(1-\varepsilon_{low}, \min(1+\varepsilon_{high}, r_t)),
$$

其中 $`\varepsilon_{high} > \varepsilon_{low}`$。DAPO 的典型取值为 $`\varepsilon_{low} = 0.2`$，$`\varepsilon_{high} = 0.28`$。

修改后的单 token 目标为：

$$
\ell_{i,t}^{clip-higher} = \min\!\left(r_{i,t}\,\hat{A}_i,\; \text{clip}_{asym}(r_{i,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high})\,\hat{A}_i\right).
$$

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

$$
P(\text{全同}) = p^G + (1-p)^G,
$$

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

$$
\mathcal{B}_{valid} = \left\{ j : 0 < \frac{1}{G}\sum_{i=1}^G r_{j,i} < 1 \right\}, \quad |\mathcal{B}_{valid}| = N_{batch}.
$$

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

$$
L^{GRPO} = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \ell_{i,t},
$$

其中 $`\ell_{i,t}`$ 是 token $`(i,t)`$ 的 clip 损失（含 KL 项或不包含）。在这个双层平均结构下，每条回答无论长短，对总损失的贡献权重都是 $`1/G`$。但长回答的损失被分摊到了更多 token 上，每个 token 获得的梯度幅度与 $`1/|o_i|`$ 成正比。

在 long-CoT 场景中，回答长度差异可达数十倍（几百到上万 token）。一条 5000 token 的推理链中，每个关键推理步骤的梯度信号被稀释为短回答的 $`1/50`$ 甚至更低。这意味着模型更难从长回答中学到精细的推理模式。

<a id="s6-4-2"></a>

### 6.4.2 DAPO 的 token-level 聚合

DAPO 将损失聚合方式改为对整个 batch 的所有 token 直接平均：

$$
L^{DAPO-token} = \frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G \sum_{t=1}^{|o_i|} \ell_{i,t}.
$$

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

$$
m_i = \begin{cases} 0 & \text{if } |o_i| = L_{max} \quad (\text{被截断}) \\ 1 & \text{otherwise} \end{cases}
$$

在 token-level loss 中，只有 $`m_i = 1`$ 的回答参与聚合。这直接消除了"因截断而误判"的噪声源。注意这与动态采样不同：动态采样过滤的是全对/全错的 prompt；超长过滤针对的是单个被截断的回答，即使同一 prompt 的其他回答未被截断且有效。

**② Soft Overlong Punishment（软超长惩罚）**：在硬上限之前设置一个缓冲区 $`[L_{max} - L_{cache}, L_{max}]`$，在此区间内施加线性递增的长度惩罚：

$$
R_{length}(o) = \begin{cases} 
0 & |o| \le L_{max} - L_{cache} \\
\displaystyle\frac{(L_{max} - L_{cache}) - |o|}{L_{cache}} & L_{max} - L_{cache} < |o| \le L_{max} \\
-1 & |o| > L_{max}
\end{cases}
$$

其中 $`L_{cache}`$ 是缓冲区宽度（DAPO 论文中的超参）。最终奖励为：

$$
R(o) = R_{verifier}(o) + R_{length}(o).
$$

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

$$
L^{DAPO} = -\mathbb{E}\left[\frac{1}{\sum_{i \in \mathcal{B}_{valid}} |o_i|}\sum_{i \in \mathcal{B}_{valid}} m_i \sum_{t=1}^{|o_i|} \min\!\left(r_{i,t}\,\hat{A}_i,\; \text{clip}_{asym}(r_{i,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high})\,\hat{A}_i\right)\right]
$$

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

<details>
<summary>查看参考答案</summary>

参考答案：对于一个当前概率很低的 token（$`\pi_\theta \approx \delta \ll 1`$），即使其绝对概率翻倍，$`r_t`$ 也只从 1 变为 2。对称 clip 的上界 $`1+\varepsilon`$（如 1.2）远低于 2，因此低概率 token 只需极小的相对增幅就触发 $`r_t > 1+\varepsilon`$ 的截断条件，梯度归零。相比之下，高概率 token 很难触及上界。长期累积效应是低概率好 token 无法被提升，概率质量持续向高概率 token 集中，熵持续下降。

</details>

**题目 2**：动态采样如何与第 4 章的全同奖励概率 $`P(\text{全同}) = p^G + (1-p)^G`$ 相关联？它如何实现隐式课程学习？

<details>
<summary>查看参考答案</summary>

参考答案：$`P(\text{全同})`$ 在 $`p \approx 0`$ 或 $`p \approx 1`$ 时接近 1，意味着简单题和极难题的大部分采样都是零梯度的。动态采样过滤掉这些 prompt，只保留 $`0 < \text{acc} < 1`$ 的 prompt，即 $`p`$ 处于中间区域的题目。随着训练推进，模型变强，原来的中等题变成简单题（$`p \to 1`$）被自动过滤，新的中等难度题进入 batch。这形成了自适应的课程学习，始终将训练集中在 $`p \approx 0.5`$ 附近——信号最强的区域。

</details>

**题目 3**：写出 GRPO sequence-level loss 和 DAPO token-level loss 的数学表达式，并说明为什么后者能消除长度偏差。

<details>
<summary>查看参考答案</summary>

参考答案：GRPO: $`L = \frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \ell_{i,t}`$；DAPO: $`L = \frac{1}{\sum_i |o_i|}\sum_i \sum_t \ell_{i,t}`$。在 GRPO 中，每条回答权重为 $`1/G`$，每 token 权重为 $`1/(G|o_i|)`$，与长度成反比。在 DAPO 中，每 token 权重为 $`1/\sum_j |o_j|`$，与所属回答长度无关。因此长回答的 token 不再被稀释，长度偏差被消除。

</details>

**题目 4**：DAPO 的软超长惩罚为什么优于硬截断？从优化景观的角度解释。

<details>
<summary>查看参考答案</summary>

参考答案：硬截断在 $`L_{max}`$ 处产生阶跃间断，奖励从 0 突变为可能的误判分数。这个不可微的突变使策略梯度估计方差极大，且无法提供"应提前多少停止"的方向信息。软惩罚在缓冲区 $`[L_{max}-L_{cache}, L_{max}]`$ 内提供连续的线性递减信号，将悬崖改造为斜坡，策略可以沿平滑梯度逐步学到合适的长度控制。

</details>

**题目 5**：DAPO 移除 KL 项的三个理由是什么？在什么场景下不应该移除 KL？

<details>
<summary>查看参考答案</summary>

参考答案：三个理由：① 长链推理需要策略大幅偏离 SFT 初始分布以探索新推理路径，KL 约束会扼杀探索；② RLVR 使用确定性 verifier，reward hacking 风险低于 RM 场景，KL 的防 hacking 作用不是必需；③ 去掉 KL 省去 reference model 的前向传播，节省显存。不应移除的场景：RLHF 对齐任务（需锚定人类偏好）、非推理类生成任务（语言质量可能退化）、训练数据覆盖面窄的场景。

</details>

---

[上一章](chapter-05.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-07.md)
