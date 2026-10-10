# GRPO — All you need to know

> 从“为什么能去掉 critic”讲到“训练为什么学不动”：手算组内优势，拆解奖励与归一化，串起 Dr.GRPO、DAPO、GSPO 的改进动机。中文推导、核心代码、训练排坑，含 37 道章末练习与参考答案。

中文学习笔记，以数学推导、算法理解与工程实践为主。建议先读 [PPO 学习文档](https://zwdx.github.io/ppo-all-you-need-to-know/)，尤其是策略梯度、GAE、重要性采样与 PPO-Clip。

**[开始在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-01.html)** · [阅读路线与符号表](https://zwdx.github.io/grpo-all-you-need-to-know/guide.html) · [配套 PPO 教程](https://zwdx.github.io/ppo-all-you-need-to-know/)

## 你会沿着怎样的路线学习

- **从去掉 critic 的取舍讲起**：理解组内基线的来源、采样换模型的代价，以及序列级奖励带来的信用分配问题。
- **用小数字建立直觉**：在 [手算例子](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-03.html#s3-7) 中计算组内优势，观察不同奖励组合如何影响学习信号。
- **读懂变体的改动动机**：沿着 [优势估计、裁剪、优化粒度与采样](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-07.html#s7-1) 四个位置理解 Dr.GRPO、DAPO、GSPO 等方法。
- **带着训练症状查问题**：对照 [诊断指标](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-08.html#s8-4) 和 [排坑清单](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-08.html#s8-5)，检查奖励不涨、长度失控与奖励失真等现象。
- **37 道章末练习，附参考答案**：先独立思考或手算，再展开答案检查理解；在线版默认折叠答案。

## 如何使用

| 你的目标 | 阅读建议 |
|---|---|
| 系统学习 | 按下方章节顺序阅读，结合卷首的依赖关系与阅读路线 |
| 先看算法 | 从 [第 3 章](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-03.html) 开始，遇到前置概念再回查 |
| 先看实现 | 从 [第 8 章](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-08.html) 开始，对照公式理解实现细节 |
| 复习与查找 | 用在线版的章节目录、站内搜索或浏览器查找定位关键词 |
| 检查掌握程度 | 先独立完成章末自测，再展开在线版中的参考答案 |

## 章节导航

<!-- chapters:start -->
| 章节 | 阅读入口 |
|---|---|
| 第 1 章 从 PPO 到 GRPO：动机与问题设定 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-01.html) |
| 第 2 章 组内基线的理论基础 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-02.html) |
| 第 3 章 GRPO 算法本体 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-03.html) |
| 第 4 章 奖励信号与可验证奖励（RLVR） | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-04.html) |
| 第 5 章 GRPO 的已知偏差与 Dr.GRPO 修正 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-05.html) |
| 第 6 章 DAPO：面向长链推理的四项关键改进 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-06.html) |
| 第 7 章 序列级方法与变体谱系 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-07.html) |
| 第 8 章 工程实现与实践 | [在线阅读](https://zwdx.github.io/grpo-all-you-need-to-know/chapter-08.html) |
<!-- chapters:end -->

## 阅读体验

- 可点击章节与小节目录、上一章/下一章、站内搜索、公式源码复制。
- 页顶「阅读设置」可调整正文字号与行距，设置自动记忆；公式随字号缩放，长公式和表格可横向滚动。
- 章末参考答案默认折叠，方便先独立作答；卷首提供学习路线、章节依赖与符号表。

## 支持教程

如果推导、代码或练习帮你解决了问题，欢迎 [在 GitHub 点亮 Star ☆](https://github.com/zwdx/grpo-all-you-need-to-know)，方便以后回来复习，也让更多学习者发现这份教程。

## 勘误与贡献

发现问题时，请在 [Issues](https://github.com/zwdx/grpo-all-you-need-to-know/issues) 中写明章节编号、原文、建议修改与依据；也欢迎提交 Pull Request。

<!-- maintenance:start -->
<details>
<summary>维护与本地预览</summary>

维护时修改根目录的完整文档，再执行以下命令更新分章版与目录。Python 3.10 及以上即可，无需安装额外依赖。

```bash
python scripts/build_docs.py
python scripts/build_docs.py --check
```

公式维护：行内小于/大于号写成 `\lt` / `\gt`；表格中条件分隔用 `\mid`，单竖线用 `\lvert` / `\rvert`，双竖线用 `\Vert`。行间 `math` 围栏保持顶层，不缩进进列表。`--check` 会拦截这些已知渲染冲突。

在线阅读站会在 `main` 更新后由 GitHub Actions 自动发布；公式解析错误会使构建失败。构建工具版本由 `package-lock.json` 锁定。本地预览需要 Node.js 20 或以上：

```bash
npm ci
npm run build
npm test
python -m http.server 8000 --directory _site
```

浏览器打开 `http://localhost:8000`。网站读取同一套分章文档；先执行上方的 `build_docs.py` 同步 Markdown。

</details>
<!-- maintenance:end -->

## 许可证

本仓库采用 [Creative Commons Attribution 4.0 International（CC BY 4.0）](LICENSE)。转载或改编时请注明文档名称、作者 zwdx 与原仓库链接，并说明修改。
