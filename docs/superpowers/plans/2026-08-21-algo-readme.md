# Algo README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库根目录创建一份面向团队内部开发者、以 EEG 算法核心为主入口的中文 README。

**Architecture:** README 作为项目总入口，只提供已核实的算法概览、数据流、目录导航、模拟数据快速开始、统一入口说明、测试状态和文档索引。详细 Trace 解释继续由 `docs/` 下现有文档负责，README 不复制其内容。

**Tech Stack:** Markdown、Mermaid、Python 3.10+、NumPy、SciPy、pytest。

## Global Constraints

- 中文为主，保留 EEG、ERD、ROI、Profile、Trace 等术语。
- 以 `erd/` 为主入口，不把根目录 `main.py` 描述为推荐入口。
- 不承诺准确率、临床效果或生产就绪状态。
- 不添加无法确认的许可证、版本徽章、安装包或性能指标。
- 明确说明患者命名数据均为模拟数据，不用于临床诊断。
- 明确记录依赖清单不完整及 `dao` 外部模块导致的测试收集失败。
- 只创建 `README.md`；不修改算法、接口、依赖或测试。

---

### Task 1: 创建并验证团队 README

**Files:**
- Create: `README.md`
- Reference: `docs/superpowers/specs/2026-08-21-algo-readme-design.md`
- Reference: `docs/EEG_CALIBRATION_TRACE_GUIDE.md`
- Reference: `docs/EEG_TRACE_FIELD_MAP.md`
- Reference: `docs/EEG_TRACE_PREVIEW.md`

**Interfaces:**
- Consumes: `erd.enter.sim_demo`、`erd.algorithm_core`、`tools/run_synthetic_eeg_trace.py` 和现有 Trace 文档。
- Produces: 团队成员可从仓库首页阅读的 `README.md`。

- [x] **Step 1: 创建 README**

创建 `README.md`，内容如下：

````markdown
# EEG 运动想象算法

面向后端集成的 EEG 运动想象（Motor Imagery, MI）校准与推理模块。项目以 `erd/` 为核心，接收后端原始 EEG 数据，完成通道映射、信号预处理、个人静息基线校准、ERD 特征计算、滑动窗口预测和诊断 Trace 输出。

> [!IMPORTANT]
> 本仓库仍处于团队内部开发阶段。仓库中的患者命名 Profile、拓扑图和 Trace 样例均为模拟数据，不是真实患者数据；本项目不能用于临床诊断或医疗决策。

## 核心能力

- 从后端 16 通道原始 ADC 数据中选择并重排 8 个 EEG 通道：`FC3`、`FC4`、`C3`、`C4`、`CP3`、`CPz`、`CP4`、`Cz`。
- 执行 50 Hz 陷波和 8–30 Hz 带通预处理。
- 在 4 秒 trial 中使用 0.5–2.5 秒分析窗计算通道功率与 ERD 特征。
- 默认收集 10 个有效静息 trial，经过逐通道 IQR 清洗后生成个人 Profile。
- 使用个人基线和动态阈值输出 Rest/Action 判定及详细诊断信息。
- 默认以 1 秒步长执行预测窗口，并提供实时通道质量检查。
- 输出校准、预测、阈值和窗口级 Trace，便于联调与问题定位。

## 数据流程

```mermaid
flowchart LR
    A["后端原始 EEG<br/>time × 16 channels"] --> B["通道选择与重排<br/>8 EEG channels"]
    B --> C["预处理<br/>50 Hz notch + 8–30 Hz band-pass"]
    C --> D{"运行模式"}
    D -->|"type = 0"| E["静息校准<br/>默认 10 个有效 trial"]
    E --> F["IQR 清洗与个人基线"]
    F --> G["个人 Profile"]
    D -->|"type = 1"| H["4 秒预测窗口"]
    G --> H
    H --> I["功率下降 / ERD 特征"]
    I --> J["Rest / Action 判定"]
    J --> K["业务结果与 Trace"]
```

## 项目结构

| 路径 | 职责 |
| --- | --- |
| `erd/algorithm_core/` | 预处理、校准、特征、阈值、推理、Profile 与 Trace 的核心实现 |
| `erd/` | 流式训练、预测、统一入口及后端适配桥接 |
| `erd/regulatory/` | 频带功率、左右侧指标和阈值候选逻辑 |
| `strategy/` | 面向业务系统的策略层；依赖仓库外部的 DAO、数据库容器和 Schema |
| `tools/` | 模拟 EEG 校准与预测 Trace 工具 |
| `tests/` | 校准、ROI、预测报告和质量入口测试 |
| `docs/` | Trace 字段、校准过程和输出预览文档 |

## 环境与依赖

代码使用 Python 3.10+ 语法。当前仓库尚未提供完整的锁定依赖文件；从源码可以确认的基础依赖包括：

- NumPy
- SciPy
- pytest（测试）

部分集成模块还依赖当前仓库之外的 `dao`、`db`、`schemes` 和 `impedance`。根目录 FastAPI 入口还引用 FastAPI、Uvicorn 及未完整接入的服务模块，因此目前不作为推荐入口。请优先使用团队现有运行环境，不要根据模块名猜测内部依赖的 PyPI 包。

## 快速开始：运行模拟 EEG Trace

前置条件：团队运行环境已提供内部 `impedance` 模块。当前仓库不包含该模块，也无法仅根据模块名确定其安装来源。

依赖接入后，在仓库根目录执行：

```powershell
python tools/run_synthetic_eeg_trace.py
```

工具使用固定随机种子生成 16 通道模拟 ADC 数据，执行一次静息校准和多窗口预测，并将结果写入被 Git 忽略的 `artifacts/synthetic_trace/`：

- 个人 Profile
- 校准 Trace
- 预测 Trace
- 业务报告预览
- 运行元数据
- 本次模拟运行说明

模拟场景只用于联调与观察算法 Trace，不代表真实生理信号，也不能用于评估模型准确率。

如果尚未接入 `impedance`，命令会报告 `ModuleNotFoundError: No module named 'impedance'`，随后因无法生成个人 Profile 而中止。这是当前环境的已知依赖限制。

## 统一算法入口

后端统一调用入口为 `erd.enter.sim_demo`：

- `type=0`：缓存静息数据，达到有效 trial 数后生成个人 Profile。
- `type=1`：加载个人 Profile，执行 Rest/Action 预测。
- `filtered_samples`：兼容历史命名，实际输入应为 `[time_steps, backend_channels]` 的原始 EEG/ADC 数组。
- `required_rest_trials`：个人校准所需有效 trial 数，默认值为 10。
- `allow_uncalibrated_prediction`：默认关闭；启用时仍必须提供并通过校验的 `fallback_profile_path`。

调用方需要提供 `case_no`、年龄、性别、采样率、运行类型、业务类型、原始数据和通道映射。完整参数与演示代码见 [`erd/enter.py`](erd/enter.py)。

## 测试

由于仓库根目录本身是 `algo` Python 包，请从它的父目录执行：

```powershell
python -m pytest algo/tests
```

当前状态：pytest 能发现根目录测试，但 `tests/test_prediction_report_timing.py` 依赖策略层的外部 `dao` 模块；在该模块未接入时，测试会在收集阶段因 `ModuleNotFoundError: No module named 'dao'` 中止。这是已知的集成依赖限制，不应被报告为测试通过。

## Trace 文档

- [EEG 校准 Trace 指南](docs/EEG_CALIBRATION_TRACE_GUIDE.md)
- [EEG Trace 字段映射](docs/EEG_TRACE_FIELD_MAP.md)
- [EEG Trace 输出预览](docs/EEG_TRACE_PREVIEW.md)
- [EEG Trace 样例](docs/EEG_TRACE_SAMPLE.log)

## 当前限制

- 尚无完整的依赖清单或锁定文件。
- 策略层依赖未包含在当前仓库中的业务基础设施模块。
- 模拟 Trace 工具需要仓库外部的 `impedance` 模块；当前环境无法完成模拟校准。
- 根目录 `main.py` 的 FastAPI 服务依赖尚未完整接入。
- 当前测试套件需要外部 `dao` 模块才能完整收集和运行。
- 项目尚未提供临床验证、准确率报告或生产部署保证。
````

- [x] **Step 2: 静态验证 README**

运行：

```powershell
git diff --check -- README.md
rg -n '^# |^## ' README.md
rg -n 'docs/EEG_CALIBRATION_TRACE_GUIDE.md|docs/EEG_TRACE_FIELD_MAP.md|docs/EEG_TRACE_PREVIEW.md|docs/EEG_TRACE_SAMPLE.log' README.md
```

预期结果：`git diff --check` 无输出；README 包含 1 个一级标题、9 个二级标题和 4 个有效的相对文档链接。

- [x] **Step 3: 验证 README 中的运行命令**

从仓库根目录运行：

```powershell
python tools/run_synthetic_eeg_trace.py
```

当前环境预期结果：命令因缺少内部 `impedance` 模块返回非零退出码，并最终报告无法创建个人 Profile；该结果必须与 README 的依赖限制一致。依赖完整的团队环境中，预期退出码为 0，输出包含 `Synthetic EEG trace run completed.`，生成文件位于被忽略的 `artifacts/synthetic_trace/`。

从仓库父目录运行：

```powershell
python -m pytest algo/tests
```

预期结果：在外部 `dao` 模块未安装的当前环境中，测试收集以 `ModuleNotFoundError: No module named 'dao'` 中止，与 README 的已知限制一致。

- [x] **Step 4: 检查范围并提交**

运行：

```powershell
git status --short
git add -- README.md docs/superpowers/plans/2026-08-21-algo-readme.md
git diff --cached --check
git commit -m "Add project README"
```

预期结果：提交只包含 `README.md` 和本实施计划；算法源码、依赖和测试均未修改。
