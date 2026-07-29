# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-28

### Added

- 新增分词全局缓存机制，`tokenize_cached()` 使用 `lru_cache` 缓存 jieba 分词结果

- 新增缓存监控接口 `get_tokenize_cache_info()` 和 `log_tokenize_cache_stats()`

### Changed

- ASR 噪声增强器优化：分词操作移至循环外层，循环内先执行概率判断再编码缓存

- 增强器注册机制简化：消除 `categories.py` 和 `base.py` 之间的注册冗余

- 增强器采用懒加载策略，仅在首次调用 `apply()` 时加载资源

### Fixed

- 修复 `_run_single` 中 `step.pre_run()` 返回 `False` 时静默通过问题，遵循快速失败（Fail-Fast）原则

### Performance

- ASR 噪声增强优化：1w 条数据提速 16 秒

- 分词缓存优化：1w 条对话 augment 耗时减少 19 秒

## [1.1.0] - 2026-07-24

### Added

- 新增 `random_utils` 统一随机数工具模块，支持 `RandomLike` 协议

- 新增 `protected_words` 机制，`RandomDeleteAugmenter` 可保护关键术语（如肯定词"嗯"、"是的"、"对的"）

- 新增 `PipelineIOError` 统一 IO 异常体系

- 新增 IO 模块拆分：`reader.py`、`writer.py`、`pickle_io.py`、`file_utils.py`

### Changed

- `random_utils.shuffle()` 返回原序列以支持链式调用

- 增强器配置方式简化：移除 `enabled` 字段，统一使用 `weight: 0` 禁用增强器

- 增强器配置移除 `enabled_categories`，策略参数统一到 `composite_config`

- logger 模块调整，优化控制台输出控制

- 删除 `context.py`，增强器直接分发方法调用

### Fixed

- 修复 `_run_single` 中前置条件不满足时的静默通过问题，改为快速失败并终止流水线

### Removed

- 删除 `random_entity_replace` 增强方法

## [1.0.0] - 2026-07-22

### Added

- 新增 `ConfigManager` 配置管理器，统一配置读取和查询接口

- 新增 `PathResolver` 路径解析器，支持占位符（`{task_dir}`、`{task_name}`、`{timestamp}`）安全替换

- 新增 `StateTracker` 状态追踪器，基于文件标记实现断点续跑

- 新增 `StepRegistry` 步骤注册表，支持装饰器模式注册步骤

- 新增 `PipelineError` 异常体系（`PipelineIOError`、`PipelineConfigError`、`PipelineValidationError`、`PipelineStepError`、`PipelineRuntimeError`）

- 新增 `CompositeAugmenter` 组合增强器，支持 `single` 和 `multi_step` 两种策略

- 新增增强器分类元信息系统（`categories.py`），管理 lexical/order/model 三类增强器

- 新增 `augmenters/utils.py` 通用工具函数（分词、切句、词典加载）

### Changed

- 拆分 `PipelineContext` 为三个职责单一的组件：`ConfigManager`、`PathResolver`、`StateTracker`

- 合并 `03_clean` 和 `04_finalize` 为 `03_clean_finalize` 流水线步骤

- 统一全局输入路径管理，`raw_dialogues` 通过全局配置注入

- `adaptive_max_variants` 参数抽取到 YAML 配置文件中

- 增强方法标识符规范化：`asr`（ASR 噪声）、`lexical`（词法增强）、`order`（语序增强）

- 简化 Executor 抽象层，直接顺序调用步骤执行

### Fixed

- 修复 `project_root` 在 `intermediate_root` 为绝对路径时推断失败问题

### Removed

- 删除 `00_generate_raw` 步骤

- 删除 `06_replace_text` 步骤

- 删除 `print_task_tree` 功能

- 删除全局并行控制，并行处理改由步骤级配置控制

- 删除冗余代码和未使用的函数或模块

## [0.1.0] - 2026-06-25

### Added

- 新增 Pipeline 框架初始版本，支持步骤化流水线执行

- 新增 `SplitDialoguesStep` 对话拆分步骤

- 新增 `BucketStep` 分桶步骤，支持手动分桶策略

- 新增 `CleanStep` 清洗步骤，集成 Data-Juicer 进行数据质量清洗

- 新增 `FinalizeStep` 最终化步骤，应用 loss 标记生成训练数据

- 新增 `AugmentStep` 语义增强步骤，支持并行处理

- 新增 `AsrNoiseAugmenter` ASR 噪声增强器，模拟语音识别错误

- 新增 `InsertFillerAugmenter`、`StutterAugmenter`、`HomophoneAugmenter`、`RandomDeleteAugmenter`、`SynonymAugmenter`、`WordRepetitionAugmenter`、`ReorderAugmenter` 基础增强器

- 新增 `StepRegistry` 步骤注册表

- 新增 `AnalyzerRegistry` 分析器注册表（RetentionAnalyzer、TurnDistributionAnalyzer）

- 新增 `ReporterRegistry` 报告器注册表（CSV、JSON、Matplotlib）

### Performance

- 清洗步骤实现多进程并行处理

- ASR 噪声增强实现 encode 缓存机制

[1.2.0]: https://github.com/example/DataCleaning_Augmentation_Rebuild/compare/v1.1.0...v1.2.0

[1.1.0]: https://github.com/example/DataCleaning_Augmentation_Rebuild/compare/v1.0.0...v1.1.0

[1.0.0]: https://github.com/example/DataCleaning_Augmentation_Rebuild/compare/v0.1.0...v1.0.0

[0.1.0]: https://github.com/example/DataCleaning_Augmentation_Rebuild/releases/tag/v0.1.0
