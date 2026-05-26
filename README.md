# 🎯 对话数据清洗与语义增强（模拟ASR转录错误）
本流水线专为**多轮对话数据**设计，实现从原始对话文档到高质量模型训练数据的全流程自动化处理。它集成了规则清洗、轮次分桶、Data‑Juicer 过滤、语义增强、模拟ASR转录错误等模块，支持配置驱动、断点续跑、任务隔离，能够显著提升数据质量并扩充训练样本。

## 📖 项目背景

在训练对话模型（如客服机器人、催收话术系统）时，原始对话数据通常存在以下问题：

-   对话历史不完整、轮次错乱
-   人工编写的对话数据不含有噪声（语气词、重复、无关信息）
-   样本分布不均（某些轮次过多或过少）
-   数据量不足，难以覆盖复杂场景
- 人工编写的对话数据无法模仿ASR转录模型的错误，人工添加错误效率太低。

本流水线旨在解决上述问题，提供一套端到端的解决方案，帮助用户快速构建高质量的训练数据集。

## 🧩 整体处理流程

    原始对话（.doc/.docx + case.txt）
        ↓ [00_dataset_process.py]
    raw_dialogues.json
        ↓ [01_split_dialogues.py]
    样本 JSONL（按批次）
        ↓ [02_split_into_buckets.py]
    分桶（按 turn 值）
        ↓ [03_clean_buckets_with_plots.py]
    Data‑Juicer 清洗 + 统计报告
        ↓ [04_apply_cleaned_loss_direct.py]
    应用 loss 标记（True/False）
        ↓ [05_main_augment_add.py]
    语义增强 + 变体生成
        ↓
    最终训练数据（原始+变体）

每个步骤均支持**参数配置**、**断点续跑**和**任务隔离**，可根据需求单独执行或整体运行。
### ASR 词表生成（独立）：

    原始 ASR 文本
        ↓ (ASR 错误分析流水线)
    prev_clean_summary.csv
        ↓ (precompute_asr_vectors.py)
    .pkl 文件（向量、拼音、前置词映射）
        ↓ (供 05 脚本加载)
    ASR 噪声增强器

## 🗂 项目脚本说明
### 1️⃣ **scripts/**
| scripts | 全流程脚本 |
|--|--|
| **common** | `封装的脚本` |
| asr_noise_augmenter.py | ASR 噪声增强器类（新增）。加载预处理生成的 `.pkl` 文件（异常词向量、拼音、前置词映射），提供 `find_best_abnormals` 方法：基于拼音相似度（编辑距离）和语义相似度（余弦）加权混合，返回与目标词最匹配的异常词候选。 |
| augment_utils_add.py | 增强函数库。包含所有基础增强函数（插入语气词、同音字替换、语序打乱、随机删除、同义词替换、词语重复等），以及 ASR 噪声增强函数  `apply_asr_noise`。提供 `multi_step_augment`（根据权重随机选择多个增强函数叠加）和 `augment_cell_multi`（处理带 `/` 分隔的多句子单元格）。 |
| **pipeline :** |  |
| 00_dataset_process.py | 解析原始对话文件（`.doc`/`.docx`）和案例提示文件（`.txt`），提取多轮对话并标注 `loss="True"`，生成 `raw_dialogues.json`。 |
| 01_split_dialogues.py | 将完整对话拆分为单轮样本（每条样本包含历史上下文），输出 JSONL 文件，并统计轮次分布。 |
| 02_split_into_buckets.py | 根据样本的 `turn` 值分桶（支持自动百分位或手动指定），以便不同桶使用不同的清洗规则。 |
| 03_clean_buckets_with_plots.py | 调用 Data‑Juicer 对每个桶进行清洗，生成清洗后样本、统计报告（含图表、保留率等）。 |
| 04_apply_cleaned_loss_direct.py | 根据清洗结果（保留的 `(id, turn)`），在原 `raw_dialogues.json` 中设置对应 assistant 消息的 `loss="True“`，其余为 `“False”`，输出 `training_data.json`。 |
| 05_main_augment_add.py | 语义增强主脚本（增强核心）。读取最终训练数据，对每个对话中的 `user` 消息（可配置）进行多步叠加增强。支持通过 `augment_weights` 控制每种增强操作的相对概率。集成了 ASR 噪声增强器，并输出原始+变体、仅变体两种格式。 |
| 06_replace_text.py | 紧急任务，修改prompt和对话实体 |
| precompute_asr_vectors.py | ASR 词表预处理脚本（新增）。读取清洗后的 `prev_clean_summary.csv`，按空格拆分异常词，使用 `sentence-transformers` 计算语义向量，同时用 `pypinyin` 计算拼音串。输出三个 `.pkl` 文件：`abnormal_vectors.pkl`（词→向量）、`abnormal_pinyin.pkl`（词→拼音）、`prev_to_abnormals.pkl`（前置词→异常词列表）。 |
|  **run_pipeline.py**| 流水线主控脚本，读取 `pipeline_config.yaml`，按步骤执行或单独执行某一步骤，支持断点续跑、任务隔离。 |
| homophone_formatting.py | 基础增强函数同音字词表规格化，符合nlpcda词表格式 |

### 2️⃣ **resources/**
| **resources/** | 说明 |
|--|--|
| **/prev_clean :** | 处理得到的ASR词表文件 |
| */prev_clean_summary.csv | 来自ASR_ERROR_Modeling项目 |
| */abnormal_pinyin.pkl |  |
| */abnormal_vectors.pkl |  |
| */prev_to_abnormals.pkl |  |
| **resources :** |  |
| bank.txt | 银行实体词表 |
| Homophone_tab.txt | 同音字词表（效果可能不如ASR词表） |
| synonyms.txt | 同义词词表（效果可能不如ASR词表） |

### 3️⃣ **Models/**
| Models/ |  |
|--|--|
| paraphrase-multilingual-MiniLM-L12-v2 | 用于 05 步骤 ASR noise 增强处理 |
|gpt2-chinese-cluecorpussmall | ASR_ERROR_Modeling项目产生词表 |
| Qwen2.5-1.5B-Instruct | ASR_ERROR_Modeling项目产生词表 |

###  5️⃣ **configs/**
| configs/ |  |
|--|--|
| ASR_test_config/ | 包含ASR增强的任务模板 |
|configs_qa/| 03步骤data-juicer清洗配置（text) |
| configs_q | 03步骤data-juicer清洗配置（user) |
| pipeline_config.yaml | pipeline流程配置模板 |

###  6️⃣ **test/**
| test/ | 测试脚本 |
|--|--|
| common/ | 同scripts |
| test_noise_augment.py | ASR 噪声增强器完整测试脚本（硬编码路径版） |


## 📂 配置文件说明
所有运行参数集中在 `pipeline_config.yaml` （配置模板）中，其他配置结构如下：
| 配置configs/ | 说明 |
|--|--|
| ASR_test_config/ | pipeline的全流程任务模板 |
|configs_qa/| 03步骤data-juicer清洗配置（text) |
| configs_q/ | 03步骤data-juicer清洗配置（user) |
| pipeline_config.yaml | pipeline流程配置模板 |


## 🚀 安装与运行
### 1. 环境准备 env_yaml/

| 项目环境（ data-juicer + sentence-transformers ) | data-juicer-sentence-transformers.yml |
|--|--|
纯净的data-juicer环境 

    env_yaml/data-juicer.yml
重建环境

    conda env create -f data-juicer-sentence-transformers.yml
重新命令环境

    conda env create -n 新环境名 -f data-juicer-sentence-transformers.yml

### 2. 准备原始数据
| 原始数据 | 说明 |
|--|--|
| `从00步骤开始：` |  |
| 对话文件（.doc 或 .docx） | 文件名格式建议包含案例编号（如 `案例123.docx`） |
| 对应的案例提示文件（.txt） | 放入 `data/cases_random/`，命名为 `case_{id}.txt` |
| `从01步骤开始：` |  |
|datas/raw_data/data-simulation-4809-modify.json | 样例（需修改01 input_json 和04 original_json 的路径） |


### 3. 修改配置文件

    复制 pipeline_config.yaml 并根据实际路径和需求调整参数。新任务务必修改 task_name 。

### 4. 运行完整流水线

    python path/to/run_pipeline.py --config path/to/pipeline_config.yaml

### 5. 单独运行某个步骤（调试）

    python path/to/run_pipeline.py --config path/to/pipeline_config.yaml --step 03_clean

有效步骤名：`00_generate_raw`, `01_split`, `02_bucket`, `03_clean`, `04_finalize`, `05_augment`。

## 📊 输出文件解读
| 文件/目录 | 说明 |
|--|--|
| `intermediate/{task_name}/raw_dialogues.json` | 原始标准化对话 |
|`intermediate/{task_name}/samples/`| 拆分后的单轮样本 JSONL |
| `intermediate/{task_name}/bucketed/` | 分桶后的样本 |
|`intermediate/{task_name}/cleaned_jsonl/{run_id}/`| 清洗后的样本 |
| `intermediate/{task_name}/final_training_data/{run_id}_final/training_data.json` | **最终训练数据**（loss 已标记） |
|`intermediate/{task_name}/cleaning_reports/{run_id}/`| 清洗报告（含图表、CSV） |
| `intermediate/{task_name}/output_augmented_data/{run_id}_augment_{tag}/` | 增强后的数据（原始+变体、仅变体） |
| `intermediate/{task_name}/logs/` | 各步骤日志 |
|  |  |

## 🔁 断点续跑与任务隔离

-   **断点续跑**：主控脚本在执行每个步骤前检查 `{task_dir}/.step_{step_name}_done` 文件。若文件存在且配置中 `resume: true`，则跳过该步骤。
    
-   **任务隔离**：每个任务拥有独立的 `work/{task_name}` 目录，不同任务间完全隔离，互不影响。
    
-   **时间戳输出**：清洗和增强步骤的输出目录自动添加时间戳，避免覆盖历史结果。
## 🛠 自定义与扩展

### 添加新的清洗配置

1.  在 `configs/configs_qa/` 下创建新的 Data‑Juicer YAML 文件。
    
2.  在 `pipeline_config.yaml` 的 `bucket_config_map` 中添加正则映射，将特定桶名指向新配置。
    

### 修改分桶策略

-   手动分桶：修改 `steps.02_bucket.manual_buckets` 列表。可以先运行01步骤，观察stats的轮次分布之后，自行决定分桶策略。
    
-   自动分桶：调整 `auto_params`（百分位点或最小桶大小）。
    
## 💡 数据增强模块说明
本流水线支持两种类型的文本增强：**基础规则增强**（轻量级，无需外部模型）和 **ASR 噪声增强**（模拟语音识别错误，需要预计算词表向量）。两者可通过配置文件灵活组合权重。
### 📌 一、基础增强函数（规则驱动）
这些增强函数不依赖任何外部模型，直接对句子进行字/词级别的变换，用于增加口语化、多样性和轻微噪声。
| 函数名 | 说明 | 示例 |
|--|--|--|
| `insert_filler` | 在句首或句中插入语气词（如“嗯”、“那个”） |“你好。” → “嗯，你好。”
| `stutter` | 重复句子中第一个汉字，模拟口吃 |“我明白了” → “我我明白了”
| `reorder` | 交换逗号前后内容或简单谓语前置 |“我吃饭了” → “吃饭了，我”
| `homophone` | 基于预定义词库替换同音字 |“还款” → “还宽”
| `random_delete` |随机删除部分字符（删除率 20%）  |“信用卡逾期” → “信用逾期”
| `random_entity_replace` | 替换实体词（如银行名称、公司名） |“华夏银行” → “招商银行”
| `similarword` | 基于同义词词库替换词语 |“还款” → “还钱”
| `word_repetition` | 随机重复句子中的一个多字词语 |“信用卡逾期” → “信用卡逾期逾期”

所有基础增强函数通过 `augment_weights` 配置相对权重（不要求总和为1），并在 `multi_step_augment` 中随机叠加使用。
### 📌  二、ASR 噪声增强（语义+拼音混合）
ASR 噪声增强模拟语音识别系统的典型错误（音近、形近、语义混淆），通过前置词匹配 + 混合相似度查找最合适的异常词进行替换或插入。
###  🧩2.1 前置要求
需要先通过 **ASR 错误分析流水线**生成清洗后的词表（`prev_clean_summary.csv`），然后运行预处理脚本生成三个 `.pkl` 文件：

    python scripts/precompute_asr_vectors.py --csv_path <词表路径> --model_path <本地模型路径>


| **resources/prev_clean :** | 生成文件 |
|--|--|
| `prev_clean_summary.csv` | 来自ASR_ERROR_Modeling项目 |
| `abnormal_pinyin.pkl` | 异常词的拼音字符串 |
| `abnormal_vectors.pkl` | 异常词的语义向量（384维） |
| `prev_to_abnormals.pkl` | 前置词 → 异常词列表映射 |

### 🧩2.2 增强逻辑
#### 1.  **分词与位置扫描**  
    使用 jieba 分词，找出所有满足前置词在 prev_to_abnormals 中的目标词位置。
    
#### 2.  **候选集限制**  
    若指定了前置词，则只在该前置词关联的异常词列表中搜索，否则使用全量异常词。
    
#### 3.  **混合相似度计算**  

| 对每个候选异常词 | 计算 |
|--|--|
| 拼音相似度 | 基于编辑距离归一化（`1 - edit_distance / max_len`） |
|语义相似度| 余弦相似度（使用预计算的语义向量） |
| 综合得分 | `α * 拼音相似度 + (1-α) * 语义相似度` （默认 `α = 0.7`，突出语音错误） |

#### 4.  **选择与操作**


-   取综合得分最高的 `top_k` 个候选（默认 `top_k=5`）
    
-   随机选择一个候选词
    
-   以 `INSERT_PROB` 概率执行**插入**（在前置词后、目标词前插入），否则执行**替换**（替换目标词）

        
#### 5.  **多位置批量操作**  

    一次扫描可同时修改多个互不相邻的位置（避免互相影响），最多修改 MAX_OPERATIONS 个词（默认 2）。
    
#### 6.  **极性保护**  

    预定义肯定词集合（如“是、能、可以”）和否定词集合（如“不、没、不能”）。若目标词属于其中之一，则只允许选择极性一致的候选词（或中性词），避免语义反转（如“能” → “不能”）。
### 🧩 2.3 配置示例

在 `pipeline_config.yaml` 的 `05_augment` 段中：
   

    05_augment:
      enabled: true
      target_roles: ["user"]
      augment_weights:
        asr_noise: 5.0          # 权重高，其他基础增强权重设为0以单独测试
        # 其他基础增强权重...
      asr_cache:
        vectors_path: "resources/prev_clean/abnormal_vectors.pkl"
        pinyin_path: "resources/prev_clean/abnormal_pinyin.pkl"
        prev_map_path: "resources/prev_clean/prev_to_abnormals.pkl"
        model_path: "Models/paraphrase-multilingual-MiniLM-L12-v2"
### 🧩 2.4 性能说明

-   **预处理阶段**：一次性计算所有异常词的向量和拼音，耗时取决于异常词数量（约 1 万词需 30 秒）。
    
-   **运行时**：仅需加载轻量级 `sentence-transformers` 模型（约 120MB），编码一个目标词耗时 < 10ms（CPU）。整个增强过程不依赖大模型（如 GPT/Qwen），可完全离线运行。
    

### 🧩 2.5 注意事项

-   前置词映射如果为空，增强器会退化为随机选择句子中的词（不推荐），请确保预处理脚本正确生成了 `prev_to_abnormals.pkl`。
    
-   极性保护依赖于预定义的词汇集合，可能需要根据实际任务扩充或调整。
    
-   插入操作可能破坏句子流畅性，建议 `INSERT_PROB` 设置 ≤ 0.3。


## 🧪 测试建议

-   先用小规模数据（如 100 个对话）测试全流程，验证配置无误。
    
-   通过 `sample_ratio` 参数（例如 0.1）加速迭代调试。
    
-   检查 `run_metadata.json` 和日志文件排查错误。
    
## 🗂 学习资料
[Data-Juicer](https://github.com/modelscope/data-juicer) 
----------

## 📄 许可证

本项目采用 [MIT 许可证](https://license/)。

----------

## 🤝 贡献与支持

欢迎提交 Issue 和 Pull Request。如有疑问，请联系项目维护者。

----------

**最后更新**：2026‑05‑22


