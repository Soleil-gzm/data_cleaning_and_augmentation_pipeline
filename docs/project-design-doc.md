# 项目设计文档

> **版本**：v1.2.0

> **最后更新**：2026-07-28

---

## 1. 系统定位

  对话数据清洗与语义增强流水线，是数据处理管线的核心组件。负责将原始多轮对话数据转换为高质量的模型训练数据，支持配置驱动、断点续跑、任务隔离。

## 2. 功能范围

### 2.1 核心步骤

| 步骤 | 名称 | 功能描述 | 并行支持 |
| --- | ---------------- | --------------------------- | ---- |
| 01 | split_dialogues | 将多轮对话拆分为单轮样本 | 否 |
| 02 | bucket | 按轮次分桶（支持手动策略） | 否 |
| 03 | clean + finalize | Data-Juicer 清洗 + 应用 loss 标记 | 是 |
| 04 | augment | 语义增强生成文本变体 | 是 |

### 2.2 增强方法

项目提供两大类共 8 种增强方法：

**词法/替换级（lexical）**

- `insert_filler` - 插入语气词（如"嗯"、"那个"）

- `stutter` - 重复首个汉字模拟口吃

- `homophone` - 同音字替换

- `random_delete` - 随机删除字符（支持保护词表）

- `synonym_replace` - 同义词替换

- `word_repetition` - 词语重复

**语序/重排级（order）**

- `reorder` - 语序重排

**模型/ASR级（model）**

- `asr_noise` - ASR 噪声模拟（语义+拼音混合）

## 3. 核心抽象

### 3.1 Pipeline（流水线主类）

```python

class Pipeline:

def run(self, step_name: str = None) -> bool

```

负责协调所有步骤的顺序执行，支持：

- 全流程执行或单步骤调试

- 断点续跑（通过 `StateTracker`）

- 进度条控制

- 日志管理

### 3.2 PipelineStep（步骤基类）

```python

class PipelineStep(ABC):

@abstractmethod

def run(self) -> bool

def pre_run(self) -> bool # 前置检查

def post_run(self) -> bool # 后置处理

```

采用 Template Method 模式，定义统一的执行生命周期：`pre_run → run → post_run`。

### 3.3 ConfigManager（配置管理器）

负责配置读取和查询，提供：

- `get_step_config(step_name)` - 获取指定步骤配置

- `is_step_enabled(step_name)` - 判断步骤是否启用

### 3.4 PathResolver（路径解析器）

统一管理项目中所有路径的解析逻辑，支持：

- 绝对路径：直接返回

- 占位符路径：`{task_dir}`、`{task_name}`、`{timestamp}` 安全替换

- 相对路径：相对于项目根目录解析

### 3.5 StateTracker（状态追踪器）

基于文件标记实现断点续跑：

- `is_step_done(step_name)` - 检查步骤是否已完成

- `mark_step_done(step_name)` - 标记步骤为已完成

### 3.6 BaseAugmenter（增强器基类）

```python

class BaseAugmenter(ABC):

@abstractmethod

def apply(self, text: str, rng: random.Random = None) -> str

def initialize(self) # 懒加载资源

```

### 3.7 CompositeAugmenter（组合增强器）

支持两种增强策略：

- `single` - 随机选择一个增强器应用，支持重试 fallback

- `multi_step` - 随机选择 N 个增强器依次叠加应用

## 4. 配置体系

### 4.1 YAML 配置结构

```yaml

task_name: "your_task_name"

resume: false

  

paths:

intermediate: "./intermediate"

input:

raw_dialogues: "data/raw_dialogues.json"

output: "./output"

  

steps_order:

- 01_split

- 02_bucket

- 03_clean

- 05_augment

  

steps:

01_split:

enabled: true

batch_size: 120000

02_bucket:

enabled: true

strategy: "manual"

manual_buckets: [[0, 0], [1, 1], [2, 2], [3, 5], [6, 10], [11, 20], [21, 9999]]

03_clean:

enabled: true

max_workers: 4

configs_dir: "configs/configs_qa"

bucket_config_map:

- pattern: ".*"

config: "overal_config.yaml"

05_augment:

enabled: true

max_workers: 4

num_variants: 3

strategy: "single"

augmenters:

insert_filler:

weight: 1.0

asr_noise:

weight: 5.0

  

logging:

level: "INFO"

show_progress: true

```

### 4.2 占位符说明

| 占位符                   | 说明      | 示例                         |
| --------------------- | ------- | -------------------------- |
| `{task_dir}`          | 任务根目录   | `./intermediate/task_name` |
| `{task_name}`         | 任务名称    | `your_task_name`           |
| `{timestamp}`         | 运行时间戳   | `20260728_153000`          |
| `{intermediate_root}` | 中间文件根目录 | `./intermediate`           |
| `{output_root}`       | 输出根目录   | `./output`                 |

## 5. 数据流

### 5.1 处理流程

```

raw_dialogues.json

  ↓ [01_split_dialogues]

samples/sample_*.jsonl

  ↓ [02_bucket]

bucketed/bucket_*/

  ↓ [03_clean + finalize]

final_training_data/{run_id}_final/

  ↓ [05_augment]

augmented/{task_name}_augmented_{timestamp}/

```

### 5.2 输入输出格式

**输入**：原始对话 JSON 文件

```json

[

{

"messages": [

{"role": "user", "content": "你好"},

{"role": "assistant", "content": "您好，请问有什么可以帮助您的？"}

]

}

]

```

**中间格式**：单轮样本 JSONL

```json

{

"id": 0,

"turn": 0,

"user_input": "Q：你好",

"target_output": "A：您好，请问有什么可以帮助您的？",

"loss": false,

"text": "你好\n您好，请问有什么可以帮助您的？"

}

```

**最终输出**：带 loss 标记的训练数据

```json

[

{

"messages": [

{"role": "user", "content": "你好"},

{"role": "assistant", "content": "您好", "loss": "True"}

]

}

]

```

## 6. 设计原则

1. **单一职责**：每个组件只负责一件事

2. **依赖倒置**：高层模块不依赖低层模块，都依赖抽象

3. **配置驱动**：行为通过配置文件控制，而非硬编码

4. **防御性编程**：路径安全替换、异常统一处理、快速失败

5. **可观测性**：每个步骤输出元数据，支持统计分析

6. **性能优先**：分词缓存、延迟拷贝、并行处理

## 7. 命名规范

### 7.1 步骤命名

采用 `{sequence}_{verb}_{object}` 格式：

- `01_split_dialogues`

- `02_bucket`

- `03_clean`

- `05_augment`

### 7.2 输出文件命名

`{task_name}_{step}_{metadata}_{timestamp}.{ext}`

示例：`my_task_clean_20260728_153000.json`

### 7.3 配置文件

放置于 `configs/` 目录，按用途组织：

- `configs_qa/` - QA 场景清洗配置

- `configs_q/` - Q 场景清洗配置

- `templates/` - 配置模板

## 8. 约束与限制

### 8.1 硬约束

- `00_generate_raw` 和 `06_replace_text` 步骤已移除

- 流水线从 `01_split_dialogues` 开始

- `02_bucket` 只支持 `manual` 分桶策略

- `augment` 步骤仅支持串行增强执行

### 8.2 配置约束

- `PathResolver.resolve()` 只替换已知占位符，未知占位符保留原文本

- `project_root` 必须显式配置或可从 `intermediate_root` 推断

- 并行处理仅由步骤级 `max_workers` 配置控制
