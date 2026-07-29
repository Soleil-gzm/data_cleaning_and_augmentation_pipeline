# 架构文档

> **版本**：v1.2.0

> **最后更新**：2026-07-28
---

## 1. 分层架构

```

┌─────────────────────────────────────────────────────────────────┐

│ 入口层 (Entry) │

│ run_pipeline.py │

├─────────────────────────────────────────────────────────────────┤

│ 协调层 (Pipeline) │

│ ┌─────────────────────────────────────────────────────────┐ │

│ │ Pipeline 主类 │ │

│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │

│ │ │ ConfigManager│ │ PathResolver │ │ StateTracker │ │ │

│ │ └──────────────┘ └──────────────┘ └──────────────┘ │ │

│ └─────────────────────────────────────────────────────────┘ │

├─────────────────────────────────────────────────────────────────┤

│ 步骤层 (Steps) │

│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │

│ │01_split │ │02_bucket │ │03_clean │ │05_augment │ │

│ │_dialogues │ │ │ │+ finalize │ │ │ │

│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │

├─────────────────────────────────────────────────────────────────┤

│ 领域层 (Augmenters) │

│ ┌─────────────────────────────────────────────────────────┐ │

│ │ CompositeAugmenter │ │

│ │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │ │

│ │ │ lexical │ │ order │ │ model │ │ registry │ │ │

│ │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │ │

│ └─────────────────────────────────────────────────────────┘ │

├─────────────────────────────────────────────────────────────────┤

│ 分析/报告层 (Analytics) │

│ ┌────────────┐ ┌────────────┐ ┌────────────┐ │

│ │ Retention │ │TurnDistrib │ │ Reporter │ │

│ │ Analyzer │ │ Analyzer │ │ Registry │ │

│ └────────────┘ └────────────┘ └────────────┘ │

├─────────────────────────────────────────────────────────────────┤

│ 基础设施层 (IO/Utils) │

│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │

│ │ reader │ │ writer │ │random_utils│ │ logger │ │

│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │

│ ┌────────────┐ ┌────────────┐ ┌────────────┐ │

│ │pickle_io │ │file_utils │ │ exceptions │ │

│ └────────────┘ └────────────┘ └────────────┘ │

└─────────────────────────────────────────────────────────────────┘

```

## 2. 模块结构

### 2.1 包结构

```
pipeline/
├── __init__.py # 包入口，导出核心类
├── exceptions.py # 统一异常体系
├── config/
│ ├── __init__.py
│ └── loader.py # 配置加载器（含占位符替换）
├── core/
│ ├── __init__.py
│ ├── pipeline.py # Pipeline 主类
│ ├── step.py # PipelineStep 基类
│ ├── step_registry.py # 步骤注册表
│ ├── config_manager.py # 配置管理器
│ ├── path_resolver.py # 路径解析器
│ └── state_tracker.py # 状态追踪器
├── steps/
│ ├── __init__.py # 步骤注册
│ ├── split_dialogues.py # 01_split
│ ├── bucket.py # 02_bucket
│ ├── clean.py # 03_clean（含 finalize 逻辑）
│ ├── finalize.py # 04_finalize
│ └── augment.py # 05_augment
├── augmenters/
│ ├── __init__.py # 增强器注册
│ ├── base.py # BaseAugmenter + AugmenterRegistry
│ ├── composite.py # CompositeAugmenter
│ ├── categories.py # 分类元信息
│ ├── registry.py # 注册辅助
│ ├── utils.py # 分词、切句、词典加载
│ └── methods/
│ ├── lexical/ # 词法增强器
│ ├── model/ # 模型增强器（ASR）
│ └── other/ # 其他增强器（order）
├── analyzers/
│ ├── __init__.py
│ ├── base.py
│ ├── registry.py
│ ├── retention.py
│ └── turn_distribution.py
├── reporters/
│ ├── __init__.py
│ ├── base.py
│ ├── csv_reporter.py
│ ├── json_reporter.py
│ ├── matplotlib_reporter.py
│ └── registry.py
├── io/
│ ├── __init__.py
│ ├── reader.py # JSON/JSONL 读取
│ ├── writer.py # JSON/JSONL 写入
│ ├── pickle_io.py # Pickle 序列化
│ └── file_utils.py # 文件统计与查找
└── utils/
├── __init__.py
├── logger.py # 日志配置
├── progress.py # 进度条
└── random_utils.py # 统一随机数工具

```

### 2.2 模块职责

| 模块           | 职责            | 对外接口                                                                           |
| ------------ | ------------- | ------------------------------------------------------------------------------ |
| `core`       | 核心调度，管理步骤生命周期 | `Pipeline`, `PipelineStep`, `StepRegistry`                                     |
| `config`     | 配置加载和占位符替换    | `ConfigLoader`                                                                 |
| `steps`      | 具体步骤实现        | `SplitDialoguesStep`, `BucketStep`, `CleanStep`, `FinalizeStep`, `AugmentStep` |
| `augmenters` | 文本增强逻辑        | `BaseAugmenter`, `CompositeAugmenter`, `AugmenterRegistry`                     |
| `analyzers`  | 数据分析          | `RetentionAnalyzer`, `TurnDistributionAnalyzer`                                |
| `reporters`  | 报告生成          | `CSVReporter`, `JSONReporter`, `MatplotlibReporter`                            |
| `io`         | 文件读写          | `read_json`, `write_json`, `jsonl_reader`, `JsonlWriter`                       |
| `utils`      | 通用工具          | `rand`, `choice`, `shuffle`, `setup_task_logger`                               |

## 3. 核心数据流

### 3.1 Pipeline 初始化流程

```

Pipeline(config_path)
│
├── ConfigLoader.load(config_path)
│ ├── 读取 YAML 文件
│ ├── 替换 {task_name}, {timestamp}, {task_dir} 占位符
│ └── 补全默认配置
│
├── ConfigManager(config)
│ └── 存储配置字典
│
├── PathResolver(config)
│ └── 计算 task_dir, intermediate_root 等
│
├── StateTracker(task_dir)
│ └── 初始化断点续跑状态
│
├── setup_task_logger(task_name, log_dir)
│ └── 配置日志输出
│
└── 确定步骤执行顺序

```

### 3.2 步骤执行流程

```

Pipeline.run()
│
├── for step_name in steps_order:
│ │
│ ├── StepRegistry.get_step(name, config_manager, path_resolver, state_tracker)
│ │ └── 创建步骤实例（依赖注入）
│ │
│ ├── step.pre_run()
│ │ └── 检查前置条件（输入文件等）
│ │
│ ├── step.run()
│ │ └── 执行核心逻辑
│ │
│ ├── step.post_run()
│ │ └── 输出统计信息
│ │
│ └── state_tracker.mark_step_done(name)
│ └── 写入完成标记文件
│
└── return all(success)

```

### 3.3 增强器调用流程

```

AugmentStep.run()

│
├── CompositeAugmenter(config)
│ ├── 读取 augmenters 配置
│ ├── 过滤 weight > 0 的增强器
│ ├── 按类别分组
│ └── 初始化策略（single / multi_step）
│
├── for dialogue in dialogues:
│ │
│ ├── 确定可增强消息（target_roles, only_loss_true）
│ │
│ ├── for variant in range(num_variants):
│ │ │
│ │ ├── for idx in enhanceable_indices:
│ │ │ │
│ │ │ ├── composite.apply(text, rng)
│ │ │ │ ├── single 策略:
│ │ │ │ │ ├── 按权重选择增强器
│ │ │ │ │ ├── augmenter.apply(text)
│ │ │ │ │ └── fallback 重试
│ │ │ │ │
│ │ │ │ └── multi_step 策略:
│ │ │ │ ├── 随机选择 N 个增强器
│ │ │ │ └── 依次叠加 apply
│ │ │ │
│ │ │ └── 记录变化
│ │ │
│ │ └── 生成变体
│ │
│ └── 收集所有变体
│
└── 输出原始 + 变体

```

## 4. 设计模式

| 模式 | 应用场景 | 涉及类 |
| ------------------------ | --------- | --------------------------------------------- |
| **Template Method** | 统一步骤执行流程 | `PipelineStep`（`pre_run → run → post_run`） |
| **Factory Method** | 动态创建步骤实例 | `StepRegistry.get_step()` |
| **Strategy** | 支持多种增强策略 | `CompositeAugmenter`（`single` / `multi_step`） |
| **Strategy** | 支持多种分桶策略 | `BucketStep`（预留 `manual` / `percentile`） |
| **Singleton** | 全局唯一的注册表 | `StepRegistry`, `AugmenterRegistry` |
| **Registry** | 增强器管理 | `AugmenterRegistry`（支持别名映射） |
| **Lazy Loading** | 延迟加载重量级资源 | `BaseAugmenter.initialize()` |
| **Dependency Injection** | 解耦步骤与基础设施 | `PipelineStep.__init__()` 接收三个组件 |
| **Decorator** | 步骤注册 | `StepRegistry.register()` |

## 5. 依赖关系

```

  ┌─────────────┐
  
  │   Pipeline  │
  
  └──────┬──────┘
│
┌──────────────┼──────────────┐

│ │ │

┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐

│ConfigManager│ │PathRes.. │ │StateTracker│

└─────────────┘ └─────────┘ └─────────────┘

│ │ │

└──────────────┼──────────────┘

│

┌──────┴──────┐

│ PipelineStep │

└──────┬──────┘

│

┌─────────────────┼─────────────────┐

│ │ │

┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐

│ SplitStep │ │ CleanStep │ │ AugmentStep │

└─────────────┘ └─────────────┘ └──────┬──────┘

│

┌──────┴──────┐

│CompositeAug..│

└──────┬──────┘

│

┌─────────┼─────────┐

│ │ │

┌──────┴──┐ ┌───┴────┐ ┌──┴──────┐

│ lexical │ │ order │ │ model │

└─────────┘ └────────┘ └─────────┘

```

**依赖方向规则**：

- `Pipeline` 依赖 `ConfigManager`, `PathResolver`, `StateTracker`

- `PipelineStep` 依赖上述三个组件

- 具体步骤依赖 `PipelineStep` 基类和 `io` 模块

- `AugmentStep` 额外依赖 `augmenters` 模块

- 无循环依赖

## 6. 关键技术决策

### 6.1 为什么拆分为三个组件？

PipelineContext 曾是 God Object，拆分后每个组件职责清晰：

- **ConfigManager** 只管配置，便于测试和替换

- **PathResolver** 只管路径，避免与配置逻辑耦合

- **StateTracker** 只管状态，便于扩展（如引入数据库）

### 6.2 为什么保留 Executor 抽象层？

最初设计引入了 Executor 层作为步骤调度的中间层。但实际使用中发现：

- 增加不必要的间接调用

- 调试复杂度高

- 实际逻辑简单

最终决定在 `Pipeline._run_single()` 中直接内联生命周期管理逻辑，简化架构。

### 6.3 增强器如何支持权重配置？

采用 `weight` 参数统一控制：

- `weight > 0`：启用增强器，同时作为选择权重

- `weight <= 0`：禁用增强器

这种设计避免了 `enabled` 和 `weight` 两个参数的不一致问题。

### 6.4 如何实现断点续跑？

基于文件标记机制：

- 每个步骤完成后，在 `task_dir` 下创建 `.step_{name}_done` 文件

- 重新运行时，`StateTracker.is_step_done()` 检查该文件是否存在

- 配置 `resume: true` 时生效

这种实现简单可靠，便于排查问题。

### 6.5 如何优化增强步骤性能？

三个层面的优化：

1. **分词缓存**：`lru_cache` 缓存 jieba 分词结果，避免重复调用

2. **延迟拷贝**：在增强器确认文本变化后才执行 `deepcopy`

3. **预缓存文本**：循环外预存所有可增强消息的文本内容

测试显示 1w 条数据提速约 16 秒。
