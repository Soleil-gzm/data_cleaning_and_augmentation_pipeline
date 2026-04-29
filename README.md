# 对话数据清洗与语义增强 
本项目提供一套完整的数据处理流水线，用于**多轮对话数据**的质量清洗与语义增强。 
主要分为两个阶段：
1. **清洗阶段**：基于 [Data-Juicer](https://github.com/modelscope/data-juicer) 对对话样本按轮次分桶，应用不同清洗规则（长度、语言质量等），最终打上 `loss` 标签生成高质量训练数据。
2. **增强阶段**：利用自定义的 `augment_utils_add` 工具包，对清洗后的对话进行多步叠加语义增强（同义词替换、回译、句式变换等），生成多样性变体，用于扩充训练集。
整个流程支持**时间戳隔离**、**任务元数据记录**和**断点续传**，便于实验追溯与结果对比。

## 项目结构（Git 跟踪）

```
├── .gitignore
├── README.md
├── data/                       # 原始数据
├── configs                     # 清洗配置文件
│   ├── configs_q
│   └── configs_qa
├── env_yaml                    # 环境配置文件
│   └── data-juicer.yml
├── resources                   # 增强工具依赖资源
│   ├── Homophone.txt
│   ├── Homophone_tab.txt
│   ├── bank.txt
│   └── synonyms.txt
└── scripts             
    ├── 00_dataset_process.py               # 原始文档 → JSON
    ├── 01_split_dialogues.py               # 对话拆分成样本
    ├── 02_split_into_buckets.py            # 按 turn 分桶
    ├── 03_clean_buckets_with_plots.py      # 调用 Data‑Juicer 清洗 + 报告
    ├── 04_apply_cleaned_loss_direct.py     # 根据清洗结果打 loss 标签( 初始json的loss全部为true )
    ├── 05_main_augment_add.py # 对话增强
    ├── common
    │   ├── __init__.py
    │   └── augment_utils_add.py            # 语义增强核心工具包
    └── homophone_formatting.py             # 同音字处理辅助脚本
```

## 环境准备

### 1. 创建 Data‑Juicer 环境（清洗阶段）

    bash
    conda env create -f env_yaml/data-juicer.yml
    conda activate data-juicer

### 2. 安装增强依赖（增强阶段）

    pip install pandas openpyxl
    如需回译等高级功能，请安装对应翻译库（如 googletrans, transformers 等）

 增强工具包 `augment_utils_add.py` 可能依赖 `resources/` 下的词表文件，请确保路径正确。

## 使用流程

### 步骤 0：生成原始 JSON

    bash
    python scripts/00_dataset_process.py
    
输入：`data/cases_random/*.txt + data/Yangqg_simulation_data/*.doc(x)`
输出：`intermediate/raw_dialogues.json`（所有 assistant 消息初始 loss=True）

### 步骤 1：拆分对话
    bash
    python scripts/01_split_dialogues.py

输出：intermediate/output_cleaning/samples/*.jsonl

### 步骤 2：按 turn 分桶
bash
    `python scripts/02_split_into_buckets.py`
输出：intermediate/output_cleaning/bucketed/bucket_*/*.jsonl

### 步骤 3：Data‑Juicer 清洗
bash
    `python scripts/03_clean_buckets_with_plots.py --tag <your_tag>`
参数 --tag：自定义清洗任务标签（如 default, strict_v2）

输出目录：intermediate/output_cleaning/cleaned_jsonl/{timestamp}_clean_{tag}/

同时生成清洗报告和图表（cleaning_reports/）

### 步骤 4：生成最终训练数据（打 loss 标签）
bash
    `python scripts/04_apply_cleaned_loss_direct.py --source_run_id <clean_run_id>`
若不指定 --source_run_id，自动使用最新的清洗结果。

输出：intermediate/output_cleaning/final_training_data/{clean_run_id}_final/training_data.json

### 步骤 5：语义增强（生成变体对话）
bash
    `python scripts/06_augment_dialogues.py --tag <augment_tag> [options]`
    
### 常用选项：

            参数	                        说明	                            默认值
        --source_run_id	                指定最终数据的                      run_id	自动取最新
        --tag	                        增强任务标签	                        default
        --num_variants	                每个原始对话生成的变体数	                 3
        --min_turns / --max_turns	    每个变体增强的轮次范围	                   1 / 2
        --target_roles	                增强的角色（user / assistant）	    user assistant
        --only_loss_true	            仅增强 loss=True 的 assistant	         False
        --adaptive_variants	            根据可增强轮次自动调整变体数	             False
        --seed	                        随机种子	                               42
示例：

bash
#### 只增强 loss=True 的 assistant，生成 5 个变体
`python scripts/06_augment_dialogues.py --tag lossOnly --only_loss_true --num_variants 5`

# 自适应变体数量，增强所有角色
`python scripts/06_augment_dialogues.py --tag adaptive_all --adaptive_variants`
输出目录：output/augmented_data/{timestamp}_augment_{tag}/

`augmented_data_{timestamp}.json` # 完整 JSON

`augmented_data_{timestamp}.jsonl` # JSONL 格式

`run_metadata.json` # 任务元数据

日志文件：intermediate/logs_augmentation/augment_{run_id}.log

重要路径约定
所有中间产物均位于 intermediate/ 目录，最终增强结果位于 output_augmented_data/。
每个任务（清洗、最终化、增强）都会生成一个 run_id，格式为：

text
{YYYYMMDD_HHMMSS}_{task}_{tag}
清洗：20250421_153022_clean_default

最终数据：20250421_153022_clean_default_final

增强：20250421_160000_augment_lossOnly

每个任务目录下均包含 run_metadata.json，记录输入来源、参数、统计信息等，方便追溯。

注意事项
Data‑Juicer 配置：configs/configs_qa/ 下的 YAML 文件必须与桶名匹配（参见 BUCKET_CONFIG_MAP）。请根据实际数据长度调整 text_length_filter 的 min_len / max_len。

语义增强工具包：请确保 common/augment_utils_add.py 中的依赖资源（如 resources/synonyms.txt）存在且路径正确。

路径空格：项目路径不能包含空格，否则 Data‑Juicer 会报错。

断点续传：01_split_dialogues.py 支持断点续传，进度记录在 intermediate/output_cleaning/progress.txt。

扩展与自定义
修改桶划分：编辑 02_split_into_buckets.py 中的 BUCKETS 字典。

调整清洗规则：修改 configs/configs_qa/ 下对应桶的 YAML 文件。

增加新的增强操作：在 augment_utils_add.py 中添加新函数，并在 augment_cell_multi 中调用。

许可证
本项目仅供内部研究使用，请勿泄露敏感数据。


modelscope download --model Qwen/Qwen3-1.7B-GGUF --local_dir /home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-1.7B-GGUF

