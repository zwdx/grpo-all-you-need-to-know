# GRPO — All you need to know

> 从“为什么能去掉 critic”讲到“训练为什么学不动”：手算组内优势，拆解奖励与归一化，串起 Dr.GRPO、DAPO、GSPO 的改进动机。中文推导、核心代码、训练排坑，含 37 道章末练习与参考答案。

中文学习笔记，以数学推导、算法理解与工程实践为主。建议先读 [PPO 学习文档](https://github.com/zwdx/ppo-all-you-need-to-know)，尤其是策略梯度、GAE、重要性采样与 PPO-Clip。

**[开始阅读](docs/chapter-01.md)** · **[完整单文件版](GRPO系统学习文档.md)** · **[阅读路线、依赖关系与符号表](docs/README.md)** · [配套 PPO 文档](https://github.com/zwdx/ppo-all-you-need-to-know)

## 你会沿着怎样的路线学习

- **从去掉 critic 的取舍讲起**：理解组内基线的来源、采样换模型的代价，以及序列级奖励带来的信用分配问题。
- **用小数字建立直觉**：在 [手算例子](docs/chapter-03.md#user-content-s3-7) 中计算组内优势，观察不同奖励组合如何影响学习信号。
- **读懂变体的改动动机**：沿着 [优势估计、裁剪、优化粒度与采样](docs/chapter-07.md#user-content-s7-1) 四个位置理解 Dr.GRPO、DAPO、GSPO 等方法。
- **带着训练症状查问题**：对照 [诊断指标](docs/chapter-08.md#user-content-s8-4) 和 [排坑清单](docs/chapter-08.md#user-content-s8-5)，检查奖励不涨、长度失控与奖励失真等现象。
- **37 道章末练习，附参考答案**：先独立思考或手算，再展开答案检查理解；分章版默认折叠答案。

## 如何使用

| 你的目标 | 阅读建议 |
|---|---|
| 系统学习 | 按下方章节顺序阅读，结合卷首的依赖关系与阅读路线 |
| 先看算法 | 从 [第 3 章](docs/chapter-03.md) 开始，遇到前置概念再回查 |
| 先看实现 | 从 [第 8 章](docs/chapter-08.md) 开始，对照公式理解实现细节 |
| 复习与查找 | 用完整文档的可点击目录、GitHub 的 Outline 或浏览器查找定位关键词 |
| 检查掌握程度 | 先独立完成章末自测，再展开分章版中的参考答案 |

## 章节导航

<!-- chapters:start -->
| 章节 | 阅读入口 |
|---|---|
| 第 1 章 从 PPO 到 GRPO：动机与问题设定 | [分章阅读](docs/chapter-01.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-1) |
| 第 2 章 组内基线的理论基础 | [分章阅读](docs/chapter-02.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-2) |
| 第 3 章 GRPO 算法本体 | [分章阅读](docs/chapter-03.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-3) |
| 第 4 章 奖励信号与可验证奖励（RLVR） | [分章阅读](docs/chapter-04.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-4) |
| 第 5 章 GRPO 的已知偏差与 Dr.GRPO 修正 | [分章阅读](docs/chapter-05.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-5) |
| 第 6 章 DAPO：面向长链推理的四项关键改进 | [分章阅读](docs/chapter-06.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-6) |
| 第 7 章 序列级方法与变体谱系 | [分章阅读](docs/chapter-07.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-7) |
| 第 8 章 工程实现与实践 | [分章阅读](docs/chapter-08.md) · [全文位置](GRPO系统学习文档.md#user-content-chapter-8) |
<!-- chapters:end -->

## 阅读体验

- 分章版提供本章目录、上一章、下一章与返回总目录；完整单文件版提供章与小节目录。
- 公式直接使用 GitHub 的 LaTeX 渲染，代码块保留语法高亮；学习时建议手推关键公式。
- 卷首保留原文的逻辑主线、章节依赖、阅读路线与通用符号约定。

## 勘误与贡献

发现问题时，请在 [Issues](https://github.com/zwdx/grpo-all-you-need-to-know/issues) 中写明章节编号、原文、建议修改与依据；也欢迎提交 Pull Request。

维护时修改根目录的完整文档，再执行以下命令更新分章版与目录。Python 3.10 及以上即可，无需安装额外依赖。

```bash
python scripts/build_docs.py
python scripts/build_docs.py --check
```

公式维护：行内小于/大于号写成 `\lt` / `\gt`；表格中条件分隔用 `\mid`，单竖线用 `\lvert` / `\rvert`，双竖线用 `\Vert`。行间 `math` 围栏保持顶层，不缩进进列表。`--check` 会拦截这些已知渲染冲突。

## 许可证

本仓库采用 [Creative Commons Attribution 4.0 International（CC BY 4.0）](LICENSE)。转载或改编时请注明文档名称、作者 zwdx 与原仓库链接，并说明修改。
