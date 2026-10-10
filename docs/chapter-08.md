[上一章](chapter-07.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md)

---

<a id="chapter-8"></a>

# 第 8 章 工程实现与实践

**本章目录**

- [8.1 框架选型与上手路径](#s8-1)
- [8.2 从零实现 GRPO 的核心逻辑](#s8-2)
- [8.3 显存、并行与 Rollout 加速](#s8-3)
- [8.4 训练诊断指标](#s8-4)
- [8.5 常见坑清单](#s8-5)
- [8.6 学习资源与进阶路径](#s8-6)
- [8.7 本章小结与全文档收尾](#s8-7)

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

$$
\text{KV Cache} \approx 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes\_per\_elem}
$$

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

---

[上一章](chapter-07.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md)
