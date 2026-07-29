# WORKLOG - 开发决策记录

本日志记录项目开发过程中的关键设计决策、技术选型和问题解决过程。

---

## 决策记录 #1：删除 Executor 抽象层

**日期**：2026-07-22

**Commit**：`57459f8` - "删除冗余代码，移除 Executor 抽象层"

### 背景

最初设计中引入了 `Executor` 抽象层作为步骤执行的中间层，负责调度步骤的 `pre_run`、`run`、`post_run` 生命周期。

### 问题

- Executor 层增加了不必要的间接调用，降低了代码可维护性

- 调试时需要跨多层追踪调用链，增加认知负担

- 实际逻辑简单，Executor 只是透传调用

### 解决方案

在 `Pipeline._run_single()` 方法中直接内联步骤生命周期管理逻辑：

```python

if not step.pre_run():

return False

try:

success = step.run()

except Exception:

success = False

if success:

step.post_run()

self._state_tracker.mark_step_done(name)

```

### 反思

过度抽象是架构设计的常见陷阱。对于简单的顺序执行场景，直接的代码结构优于多层次抽象。保持 YAGNI（You Ain't Gonna Need It）原则，只为已知需求设计抽象层。

---

## 决策记录 #2：拆分 PipelineContext 为三个职责单一的组件

**日期**：2026-07-22

**Commit**：`cd551d4` - "拆分 PipelineContext"

### 背景

最初的 `PipelineContext` 是一个 God Object，同时管理配置读取、路径解析、状态追踪、日志记录等多种职责。

### 问题

- 职责不清，修改一个功能可能意外影响其他功能

- 单元测试困难，需要 mock 大量不相关的依赖

- 不符合单一职责原则

### 解决方案

拆分为三个独立组件：

| 组件 | 职责 | 关键方法 |
| --------------- | ---------- | ---------------------------------------- |
| `ConfigManager` | 配置读取和查询 | `get_step_config()`, `is_step_enabled()` |
| `PathResolver` | 路径解析和占位符替换 | `resolve()`, `get_step_output_dir()` |
| `StateTracker` | 断点续跑状态管理 | `is_step_done()`, `mark_step_done()` |

### 反思

拆分后每个组件职责清晰，易于独立测试和替换。这种"三个小而专"的设计比"一个大而全"的设计更优。Pipeline 类只需组合这三个组件即可。

---

## 决策记录 #3：增强器配置从 enabled 改为 weight: 0

**日期**：2026-07-24

**Commit**：`c0e1616` - "简化增强器配置方式 V1"

### 背景

最初增强器配置使用 `enabled: true/false` 控制启用状态，同时用 `weight` 控制选择概率。

```yaml

augmenters:

insert_filler:

enabled: true

weight: 1.0

asr_noise:

enabled: false

weight: 5.0

```

### 问题

- 两个参数容易不一致（`enabled: true, weight: 0` 或 `enabled: false, weight: 5`）

- 配置复杂度增加，用户需要理解两个参数的协同作用

### 解决方案

统一使用 `weight` 参数：`weight > 0` 启用，`weight <= 0` 禁用。

```yaml

augmenters:

insert_filler:

weight: 1.0

asr_noise:

weight: 0 # 禁用

```

### 反思

单一参数控制多个行为是更好的 API 设计。`weight` 参数同时表达了"是否启用"和"选中权重"两个语义，简化了配置，减少了出错可能。

---

## 决策记录 #4：快速失败原则在 pre_run 中的应用

**日期**：2026-07-24

**Commit**：`09082ec` - "修复_run_single的pre_run的静默通过"

### 背景

步骤的 `pre_run()` 方法用于检查前置条件（如输入文件是否存在）。原实现中 `pre_run()` 返回 `False` 时，`_run_single()` 方法返回 `True`（表示成功）。

### 问题

- 前置条件不满足但返回成功，导致流水线继续执行

- 下游步骤可能基于不存在的数据运行，产生难以排查的错误

- 无意义的重试循环，浪费计算资源

### 解决方案

改为快速失败：`pre_run()` 返回 `False` 时，`_run_single()` 返回 `False` 并记录错误日志。

```python

if not step.pre_run():

self.logger.error(f"❌ 步骤 {name} 前置条件不满足，终止执行")

return False # 终止流水线

```

### 反思

Fail-Fast 原则确保错误尽早暴露。静默通过会掩盖问题，让错误在下游以更隐蔽的方式显现，增加排查成本。

---

## 决策记录 #5：分词全局缓存优化

**日期**：2026-07-28

**Commit**：`7224e9d` - "perf(tokenize): 设置全局缓存"

### 背景

增强器（特别是 `AsrNoiseAugmenter` 和 `SynonymAugmenter`）在处理每条消息时都需要调用 jieba 分词。1w 条对话的增强过程中，大量重复分词调用相同文本。

### 问题

- 重复分词消耗大量 CPU 时间

- jieba 分词本身有开销，不适合频繁重复调用

### 解决方案

使用 `functools.lru_cache` 实现全局缓存：

```python

@lru_cache(maxsize=_TOKENIZE_CACHE_SIZE)

def tokenize_cached(text: str) -> tuple:

_ensure_jieba()

return tuple(jieba.cut(text))

```

缓存大小可通过环境变量 `TOKENIZE_CACHE_SIZE` 配置，默认 10000 条。

### 效果

- 1w 条对话 augment 耗时减少 19 秒

- 缓存命中率监控可通过 `log_tokenize_cache_stats()` 获取

### 反思

在批量处理场景下，缓存是最有效的优化手段之一。识别出可缓存的热点函数，并使用合适的缓存策略（LRU、TTL），能带来显著的性能提升。

---

## 决策记录 #6：增强器懒加载与注册冗余消除

**日期**：2026-07-28

**Commit**：`f37e0a0` - "消除categories.py和base.py的注册冗余"

### 背景

最初 `AugmenterRegistry.register()` 需要同时在 `base.py` 和 `categories.py` 中维护增强器元信息（分类、权重等）。

### 问题

- 两处信息容易不一致

- 新增增强器需要同时修改两个文件，违反 DRY 原则

### 解决方案

1. 将增强器元信息统一到 `categories.py` 的 `AUGMENTER_META` 字典

2. `base.py` 的注册方法不再需要 `category` 参数

3. 增强器实现懒加载：`initialize()` 方法在首次 `apply()` 调用前才执行资源加载

```python

def initialize(self):

if not self._initialized:

self._load_resources()

self._initialized = True

```

### 反思

单一数据源（Single Source of Truth）原则适用于所有配置管理场景。懒加载则优化了启动性能，避免在不必要时加载重量级资源（如 ASR 模型）。

---

## 决策记录 #7：合并 Clean 和 Finalize 步骤

**日期**：2026-07-23

**Commit**：`80e8239` - "合并clean和finalize"

### 背景

原设计中 `03_clean`（清洗）和 `04_finalize`（应用 loss 标记）是两个独立步骤。

### 问题

- 两个步骤紧密耦合：finalize 必须依赖 clean 的输出

- 中间数据传递增加了 IO 开销和出错可能

- 用户需要理解两个步骤的依赖关系

### 解决方案

将 finalize 的业务逻辑整合到 `CleanStep._run_finalize()` 方法中，在清洗完成后自动执行。用户配置中只需定义 `03_clean` 步骤。

```python

def run(self) -> bool:

# ... 清洗逻辑 ...

if not self._run_finalize(run_id):

return False

return True

```

### 反思

当两个步骤的输入输出存在强依赖时，合并为单个步骤是更好的选择。减少用户心智负担，同时降低 IO 中间环节的开销。

---

## 决策记录 #8：路径解析安全替换机制

**日期**：2026-07-22

**Commit**：`931a05c` - "统一路径解析模块"

### 背景

最初路径解析使用 Python 的 `str.format()` 方法，当配置中包含未定义占位符时会抛出 `KeyError`。

### 问题

- `KeyError` 异常信息不明确，难以定位问题

- 用户可能误写占位符（如 `{task_dir}` 写成 `{taskdir}`），导致运行时崩溃

### 解决方案

实现安全的占位符替换：只替换已知的 4 个占位符（`{task_dir}`、`{task_name}`、`{timestamp}`、`{intermediate_root}`、`{output_root}`），遇到未知占位符时保留原文本。

```python

placeholders = {

"{task_dir}": str(self._task_dir),

"{task_name}": self._task_name,

"{timestamp}": self._timestamp,

"{intermediate_root}": str(self._intermediate_root),

"{output_root}": str(self._output_root),

}

for placeholder, value in placeholders.items():

resolved = resolved.replace(placeholder, value)

```

### 反思

防御性编程原则在路径处理中尤为重要。配置文件由用户编写，必须假设可能存在各种格式错误。安全替换机制避免了因拼写错误导致的运行时崩溃。

---

## 决策记录 #9：Protected Words 机制

**日期**：2026-07-24

**Commit**：`7764e3b` - "随机删除方法使用保护词表"

### 背景

`RandomDeleteAugmenter` 随机删除文本中的字符，可能删除关键的极性词汇（如"嗯"、"是的"、"对的"等肯定词），导致语义反转。

### 问题

- 删除肯定词后，回答的确认语义丢失

- 生成的变体变成无效样本

### 解决方案

引入 `protected_words` 机制：在执行随机删除前，检查文本是否包含受保护词汇。如果删除受保护词汇会影响语义，则跳过该位置的删除操作。
同时，protected_words 也配置在 jieba 自定义词典中，确保这些词作为整体分词，不会被拆散。

### 反思

数据增强的核心挑战是在"增加多样性"和"保持语义正确性"之间取得平衡。受保护词表是一种轻量级的语义保护机制，适用于规则驱动的增强方法。

---

## 决策记录 #10：延迟拷贝与预缓存优化

**日期**：2026-07-27

**Commit**：`73d5dff` - "pref(asr_noise)：概率跳过之后再编码"

### 背景

在增强步骤的变体生成循环中，每条消息都执行 `deepcopy()` 和文本读取操作。

### 问题

- 不必要的 deepcopy：增强器判断不需要修改时，deepcopy 浪费内存

- 重复文本读取：同一条消息的文本在多个增强器中被反复访问

### 解决方案

1. **延迟 deepcopy**：在增强器确认文本发生变化后再执行 deepcopy

2. **预缓存文本**：在变体生成循环外预存所有可增强消息的文本内容

```python

# 预缓存文本

text_cache = {idx: messages[idx].get("content", "") for idx in enhanceable}

  

for variant_idx in range(num_variants):

new_messages = list(messages) # 轻量级列表复制

changed = False

for idx in enhanceable:

original_text = text_cache[idx] # 使用缓存

new_text = augmenter.apply(original_text, rng=rng)

if new_text != original_text:

if not changed:

new_dialogue = deepcopy(dialogue) # 延迟 deepcopy

changed = True

new_dialogue["messages"][idx]["content"] = new_text

```

### 反思

内存优化的关键是"按需分配"。延迟拷贝避免了不必要的内存分配，预缓存减少了重复的数据访问。这些优化在大规模数据处理场景下带来显著的性能提升。
