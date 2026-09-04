[English](./README.md) | **中文**

<p align="center">
  <img src="docs/assets/iai-memory-banner.png" alt="iai-memory —— 为你的 AI 编程工作流打造的个人记忆引擎" width="100%">
</p>

<p align="center">
  <b>逐字捕获对话，跨会话召回相关上下文，<br>
  并在事实变化时同时保留新旧措辞、二者都可检索。</b>
</p>

<p align="center">
  <img src="docs/assets/iai-brain-demo.gif" alt="iai-memory 搜索、召回、置顶、淡出、抢救记忆并学习文件" width="850">
</p>

<p align="center">
  <a href="https://pypi.org/project/iai-pme/"><img src="https://img.shields.io/pypi/v/iai-pme?style=flat-square&color=1f6feb&label=pypi" alt="iai-memory 在 PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f6feb?style=flat-square" alt="MIT 许可证"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 或 3.12">
  <img src="https://img.shields.io/badge/macOS%20%7C%20Linux-supported-555?style=flat-square" alt="支持 macOS 与 Linux">
  <img src="https://img.shields.io/badge/Windows-beta-dbab09?style=flat-square&logo=windows&logoColor=white" alt="Windows 测试版">
  <img src="https://img.shields.io/badge/MCP-compatible-8957e5?style=flat-square" alt="兼容 MCP">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rescue%4010-1.000-2ea043?style=flat-square" alt="Rescue@10 1.000">
  <img src="https://img.shields.io/badge/LongMemEval%20R%405-0.962-2ea043?style=flat-square" alt="LongMemEval R@5 0.962">
  <img src="https://img.shields.io/badge/historical--verbatim-1.000-2ea043?style=flat-square" alt="历史逐字命中@10 1.000">
  <img src="https://img.shields.io/badge/at%20rest-AES--256--GCM-2ea043?style=flat-square" alt="静态存储 AES-256-GCM">
</p>

<p align="center">
  <a href="#快速开始"><b>快速开始</b></a> ·
  <a href="#工作原理"><b>工作原理</b></a> ·
  <a href="#基准测试"><b>基准测试</b></a> ·
  <a href="#兼容性"><b>兼容性</b></a> ·
  <a href="docs/REFERENCE.md"><b>技术参考</b></a>
</p>

---

## 这是什么

iai-memory 为你已经在用的编程助手，在你自己的机器上提供持久记忆。启用环境挂钩（ambient hooks）后，它会记录对话的双方、保留捕获时的原始措辞，并在会话开始或推进时提供一小段有界的相关历史。你无需维护记忆文件，也无需反复说“记住这个”。

纠正不会改写历史。一条被更新的事实会成为一条新记录，并与被取代的旧记录相互链接，因此当前陈述和早先措辞都仍可检索。召回可以把相互矛盾或已被取代的记录与匹配结果并列返回，而不是让过时的事实冒充当前事实。

这是**为你已经在用的助手打造的个人引擎**，而不是面向应用的多租户记忆 API。情景（episodic）捕获是一次写入、逐字保存的；存储、嵌入、检索、图操作和仪表盘全部在本地运行。无需任何外部向量数据库或图数据库。

**其记忆风格是刻意“自闭式”的：** 逐字优先于转述、线索精确、专注持续、罕见事件被当作罕见来保留。[名称由来](#名称由来)。

---

## 快速开始

### Claude Code

```bash
python3.12 -m pip install -U iai-pme
```

然后在 Claude Code 中运行：

```text
/plugin marketplace add CodeAbra/iai-personal-memory-engine
/plugin install iai-memory@iai-pme
```

重启会话，然后验证：

```bash
iai --version
iai-mcp daemon status
iai-mcp doctor
```

也支持 Python 3.11。

### macOS 或 Linux：一体化源码安装

```bash
curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash
```

它会构建 Rust 引擎和 TypeScript 包装器，安装后台服务与挂钩，注册 Claude Code，并运行健康检查。需要 Git、Python 3.11/3.12、Node.js 18+ 和 Rust。若想在不改动机器的情况下查看步骤：

```bash
curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash -s -- --dry-run
```

### 其他宿主

```bash
python3.12 -m pip install -U iai-pme
iai-mcp crypto init
iai-mcp daemon install
iai-mcp capture-hooks install --target codex
```

把 `codex` 换成 `cursor`、`antigravity`、`hermes`、`openclaw` 或 `all`。MCP 工具适用于任何支持 MCP-over-stdio 的客户端；自动捕获与上下文注入取决于宿主暴露的挂钩。详见[技术参考](docs/REFERENCE.md)。

新建的存储默认使用原生引擎格式；升级时既有存储会保持其当前格式。要把既有的旧版 SQLite 存储迁移到原生引擎，运行 `iai-mcp migrate-to-lilli`——`iai-mcp doctor` 会打印确切命令，[技术参考](docs/REFERENCE.md)记录了完整流程。

---

## 安装之后会发生什么

| 事件 | 动作 |
|---|---|
| 提示提交 | 新的对话轮次以文件 IO 追加到会话缓冲区；捕获路径上无需嵌入或引擎 RPC |
| 会话结束 | 剩余的转录内容被转存以待摄取；挂钩失败不会阻塞宿主 |
| 会话开始 | 一段有界的记忆前缀作为宿主上下文暴露；空存储或引擎不可用时输出为空 |
| 后续轮次 | 支持的宿主会收到一小份前瞻（foresight）或增量包，附带年龄与修订标记 |
| 空闲时 | 捕获内容被嵌入、去重、加密、插入、聚类、巩固、强化并衰减 |

后台进程在 CLI 中称为 `daemon`。当它处于休眠或临时不可用时，MCP 包装器和 `iai` 仍能直接读取本地存储。

---

## 工作原理

### 记忆模型

| 层 | 内容 |
|---|---|
| **情景（Episodic）** | 带时间戳、一次写入的“当时所说”的片段 |
| **语义（Semantic）** | 空闲巩固期间由相关情景归纳出的摘要 |
| **程序（Procedural）** | 随时间习得的十个有界行为参数 |

不同的超维（hyperdimensional）表示让字面细节、语义结构和行为倾向不会坍缩到同一个向量面上。

本地、无需大模型的召回路径结合了语义相似度、图证据、时近性、时间有效性和词法证据。`memory_recall` 同时返回 `hits` 和 `anti_hits`；`memory_contradict` 会关闭旧记录的有效区间、创建新记录并把二者链接起来。

空闲时，引擎会把相关情景分组、归纳语义记忆、强化有用路径、衰减未复核的弱边。一个可选的 REM 步骤可能通过用户已有的 Claude 订阅调用一次 `claude -p`，上限不超过每日配额的 1%。无需 Anthropic API 密钥。

### 自研核心组件

| 组件 | 作用 |
|---|---|
| **Hippo** | 在一个本地存储中容纳加密记录、向量索引和图 |
| **MOSAIC** | Leiden 家族的社区检测，具备稳定的社区身份 |
| **Lilli HD** | 超维基底与结构化召回 |
| **原生引擎** | Rust 嵌入器与图内核 |

---

## 仪表盘与 CLI

```bash
iai brain
```

本地仪表盘可搜索存储、展示图邻域与矛盾、置顶或淡出记忆、摄取文件、控制后台引擎，并根据你自己的存储报告 token 用量估计。

```text
iai recall · temporal-recall · search · ask · capture · teach · upload
iai watch · brain · status · last
```

`iai upload` 接受文档、Office 文件、电子书、源代码、配置文件和目录。完整格式与运维命令列在 [`docs/REFERENCE.md`](docs/REFERENCE.md)。

---

## 基准测试

每个测试脚手架都随 `bench/` 一同发布；方法学与复现命令见 [`BENCHMARKS.md`](BENCHMARKS.md)。

| 基准 | 结果 |
|---|---:|
| 矛盾后 Rescue@10 | **1.000** |
| 历史逐字命中@10 | **1.000** |
| LongMemEval-S R@5（产品嵌入器） | **0.962** |
| LongMemEval-S R@10（产品嵌入器） | **0.978** |

历史逐字检索的平面余弦（flat-cosine）基线约为 0.71。使用相同的 `all-MiniLM-L6-v2` 嵌入器时，iai-memory 与 mempalace v3.3.6 的 R@5 均为 `0.966`、R@10 均为 `0.978`；不宣称胜出。

在作者自己的存储上，自动注入的一份记忆包平均约 350 tokens，而它所替代的一次智能体搜索往返约 2,850 tokens：在该实测负载上约便宜 88%。这不适用于显式的 `memory_recall`，后者的默认响应预算为 1,500 tokens。

---

## MCP 工具

```text
memory_recall              memory_temporal_recall
memory_recall_structural   memory_search
memory_capture             memory_contradict
memory_reinforce           memory_consolidate
profile_get_set            topology
schema_list                events_query
episodes_recent            curiosity_pending
```

共十四个工具，涵盖线索式、时间式、结构式和词法式召回；捕获与纠正；强化与巩固；行为画像控制；以及存储自省。

---

## 兼容性

| 宿主 | 环境行为 |
|---|---|
| **Claude Code** | 会话开始召回、逐轮更新、逐轮捕获与会话捕获 |
| **Codex CLI** | 通过 Codex 挂钩完整集成 |
| **Cursor** | 会话开始召回与捕获；无逐轮文本注入 |
| **Antigravity** | 每次调用召回，并从无损转录中捕获 |
| **Hermes 0.5.0+** | 模型调用前召回，并从其消息存储中捕获 |
| **OpenClaw** | 按需提供 MCP 工具；无环境 shell 挂钩 |
| **Gemini CLI 及其他 MCP 宿主** | 提供 MCP 工具；除上表所列外不附带宿主专用挂钩 |
| **Claude Desktop** | 提供 MCP 工具；纯 Chat 不暴露 Claude Code 式的环境挂钩 |

---

## 隐私与限制

- 记录以 AES-256-GCM 静态加密。存储与密钥位于 `~/.iai-mcp/`；请一起备份。
- macOS 与 Linux 使用 Unix 套接字。Windows 使用带每用户令牌的临时回环端口。
- 没有 iai-memory 账号、遥测管线、托管仪表盘或跨机同步。
- 可选的 iai-memory 网络活动仅有 REM 的 `claude -p` 步骤和每日一次 PyPI 版本检查。设置 `IAI_MCP_VERSION_CHECK=0` 可关闭该检查。
- 存储拒绝混用不兼容的嵌入代（generation）；更换嵌入器需要一次显式迁移。
- 在最初大约十个会话里召回通常一般，其质量与延迟取决于语料规模、语言、嵌入器与已存历史。
- 默认存储以英语为先。原始的非英语记录需要显式的 `raw:<lang>` 标签以及一个多语言或自定义嵌入器。
- Windows 为测试版。环境行为随宿主挂钩支持而不同。
- 本项目由个人维护，无企业级 SLA。

健康检查与更新：

```bash
iai-mcp doctor          # 36 项检查
iai-mcp daemon status
iai-mcp self-update
```

---

## 名称由来

**IAI —— Independent Autistic Intelligence（独立·自闭式·智能）** 描述的是这套记忆设计。

- **Independent（独立）：** 引擎、存储、嵌入与仪表盘都在本地运行。
- **Autistic（自闭式）：** 字面保存、线索精确、专注持续，把罕见事件当作罕见来保留，而不是抹平成一份普通摘要。这是对运行设计的描述，而非诊断或随意的比喻。
- **Intelligence（智能）：** 取系统层面的含义——一个观察、适应、自我重组并长期保持可用的过程。

“个人记忆引擎”界定了它的范围：一个人的记忆，在一台机器上，供其已经在用的助手使用。

---

## 文档

- [`docs/REFERENCE.md`](docs/REFERENCE.md) —— 技术与运维参考
- [`BENCHMARKS.md`](BENCHMARKS.md) —— 方法学与复现命令
- [`docs/EMBEDDERS.md`](docs/EMBEDDERS.md) —— 提供器、语言与迁移
- [`CHANGELOG.md`](CHANGELOG.md) —— 发布历史
- [`CONTRIBUTING.md`](CONTRIBUTING.md) —— 开发与测试设置
- [`SECURITY.md`](SECURITY.md) —— 私密漏洞上报

欢迎提交 issue 与 pull request。对检索、捕获、矛盾处理或巩固的改动，应附带相关的基准重跑。

## 作者

由 Areg Aramovich Noya 与 Lilli Noya，与 [lcgc.dev](https://lcgc.dev) 团队合作打造。

## 许可证

[MIT](LICENSE)
