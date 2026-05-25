import os
import random
import re
import pandas as pd
import jieba
from nlpcda import Homophone, RandomDeleteChar, Randomword, Similarword

# 预先加载词典
jieba.initialize()

# ================= 可配置参数 =================
NUM_VARIANTS = 3                    # 每个原句生成几个变体（默认）

# 语气词库
FILLERS = ["嗯", "那个", "就是", "呃", "啊"]
TAILS = ["吧", "啊", "哦", "呗"]
TAIL_WORDS = set(["吧", "啊", "哦", "呗", "嗯", "啦", "呀", "嘛", "呐", "哈", "了", "吗", "呢"])

# 否定词集合（语序打乱时跳过）
NEGATION_WORDS = set(["不", "没", "无", "别", "不要", "不用", "未曾"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================= 随机同义词替换器初始化 =================
SIMILARWORD_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'synonyms.txt')
if os.path.exists(SIMILARWORD_DICT_PATH):
    try:
        _similarword_aug = Similarword(base_file=SIMILARWORD_DICT_PATH, create_num=3, change_rate=0.2, seed=42)
        print(f"[INFO] 成功加载自定义同义词词库: {SIMILARWORD_DICT_PATH}")
    except Exception as e:
        print(f"[ERROR] 加载自定义同义词词库失败: {e}，使用默认词库")
        _similarword_aug = Similarword(create_num=3, change_rate=0.2, seed=42)
else:
    print(f"[WARN] 自定义同义词词库不存在，使用默认词库")
    _similarword_aug = Similarword(create_num=3, change_rate=0.2, seed=42)

# ================= 同音字替换器初始化 =================
HOMOPHONE_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'Homophone_tab.txt')
if not os.path.exists(HOMOPHONE_DICT_PATH):
    print(f"[WARN] 同音词词库文件不存在，将使用默认词库（可能产生生僻字）")
    _homophone_aug = Homophone(create_num=3, change_rate=0.3, seed=42)
else:
    try:
        _homophone_aug = Homophone(base_file=HOMOPHONE_DICT_PATH, create_num=3, change_rate=0.3, seed=42)
        print(f"[INFO] 成功加载同音词自定义词库: {HOMOPHONE_DICT_PATH}")
    except Exception as e:
        print(f"[ERROR] 加载同音词自定义词库失败: {e}，使用默认词库")
        _homophone_aug = Homophone(create_num=3, change_rate=0.3, seed=42)

# ================= 实体词替换器初始化 =================
ENTITY_FILE = os.path.join(BASE_DIR, 'resources', 'bank.txt')
if os.path.exists(ENTITY_FILE):
    try:
        _random_entity_aug = Randomword(base_file=ENTITY_FILE, create_num=3, change_rate=0.2, seed=42)
        print(f"[INFO] 成功加载自定义实体词库: {ENTITY_FILE}")
    except Exception as e:
        print(f"[ERROR] 加载自定义实体词库失败: {e}，使用默认词库")
        _random_entity_aug = Randomword(create_num=3, change_rate=0.2, seed=42)
else:
    print(f"[WARN] 自定义实体词库不存在，使用默认词库")
    _random_entity_aug = Randomword(create_num=3, change_rate=0.2, seed=42)

# ================= 随机删除增强器 =================
_random_delete_aug = RandomDeleteChar(create_num=3, change_rate=0.2, seed=42)

# ================= 独立增强函数（可叠加） =================

def apply_insert_filler(sentence: str) -> str:
    """插入语气词（句首或句中）"""
    filler = random.choice(FILLERS)
    if random.random() < 0.6:
        return f"{filler}，{sentence}"
    else:
        match = re.search(r'[，,。？!]', sentence)
        if match:
            pos = match.end()
            if pos > len(sentence) * 0.6:
                return f"{filler}，{sentence}"
            return sentence[:pos] + filler + "，" + sentence[pos:]
        else:
            words = sentence.split()
            if len(words) >= 2:
                return words[0] + filler + "，" + " ".join(words[1:])
            else:
                return f"{filler}，{sentence}"

def apply_stutter(sentence: str) -> str:
    """结巴模拟（重复第一个汉字）"""
    if len(sentence) < 2:
        return sentence
    match = re.search(r'[\u4e00-\u9fa5]', sentence)
    if not match:
        if len(sentence) > 1:
            return sentence[0] * 2 + sentence[1:]
        return sentence
    char = match.group()
    repeat_count = random.randint(1, 2)
    stuttered_char = char * (repeat_count + 1)
    start, end = match.start(), match.end()
    return sentence[:start] + stuttered_char + sentence[end:]

def reorder_sentence(sentence: str) -> str:
    """语序打乱（交换逗号前后，或简单谓语前置）"""
    if len(sentence) < 5:
        return sentence
    if any(neg in sentence for neg in NEGATION_WORDS):
        return sentence

    end_punct = ''
    if sentence and sentence[-1] in '。！？!?':
        end_punct = sentence[-1]
        sentence = sentence[:-1].rstrip()

    # 模式1：交换逗号前后
    if '，' in sentence:
        parts = sentence.split('，', 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
            return new_sent + end_punct
    if ',' in sentence:
        parts = sentence.split(',', 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
            return new_sent + end_punct

    # 模式2：简单谓语前置
    match = re.match(r'^(我|你)(已经|也|就|都)?(\w+?)(了|过)?(.*)$', sentence)
    if match:
        subject = match.group(1)
        adverb = match.group(2) or ''
        verb = match.group(3)
        aspect = match.group(4) or ''
        rest = match.group(5).strip()
        if verb and len(verb) >= 1:
            new_sent = f"{verb}{aspect}{rest}，{subject}{adverb}"
            new_sent = re.sub(r'\s+', '', new_sent)
            return new_sent + end_punct

    return sentence + end_punct

def apply_reorder(sentence: str) -> str:
    """语序打乱"""
    return reorder_sentence(sentence)

def homophone_augment(sentence: str) -> str:
    """同音字替换"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _homophone_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"同音字替换出错: {e}")
        return sentence

def apply_homophone(sentence: str) -> str:
    """同音字替换（别名）"""
    return homophone_augment(sentence)

def random_delete_augment(sentence: str) -> str:
    """随机删除字符"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _random_delete_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"随机删除字符出错: {e}")
        return sentence

def apply_random_delete(sentence: str) -> str:
    """随机删除字符（别名）"""
    return random_delete_augment(sentence)

def apply_random_entity_replace(sentence: str) -> str:
    """随机替换句子中的实体（公司/机构名称）"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _random_entity_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"随机实体替换出错: {e}")
        return sentence

def apply_similarword(sentence: str) -> str:
    """使用同义词替换进行增强（替换词语为同义词）"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _similarword_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"同义词替换出错: {e}")
        return sentence

def apply_word_repetition(sentence: str) -> str:
    """使用 jieba 分词后，随机重复句子中的一个多字词语,（长度≥2），避免与 stutter 功能重叠"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence

    words = jieba.lcut(sentence)
    candidates = [w for w in words if len(w) >= 2 and re.match(r'[\u4e00-\u9fa5]+', w)]
    if not candidates:
        return sentence

    chosen = random.choice(candidates)
    new_sentence = sentence.replace(chosen, chosen + chosen, 1)
    return new_sentence

# ================= 新增：ASR 噪声增强（集成到 pipeline）=================
import random as _random
import jieba as _jieba
import re as _re

_asr_augmenter = None

def set_asr_augmenter(augmenter):
    """设置全局 ASR 增强器实例（由主脚本调用）"""
    global _asr_augmenter
    _asr_augmenter = augmenter

def get_asr_augmenter():
    """获取 ASR 增强器实例"""
    return _asr_augmenter

# def apply_asr_noise(sentence: str) -> str:
#     """
#     对句子应用 ASR 噪声增强（多位置、替换/插入、前置词匹配）
#     如果未设置增强器或句子无效，返回原句
#     增加极性检测，避免肯定/否定词被翻转
#     """

#     print(f"[ASR_DEBUG] 函数被调用，句子: {sentence[:30]}...")
#     augmenter = get_asr_augmenter()
#     print(f"[ASR_DEBUG] augmenter 是否为 None: {augmenter is None}")
#     if augmenter is not None:
#         print(f"[ASR_DEBUG] 前置词映射大小: {len(augmenter.prev_to_abnormals)}")
#     else:
#         print("[ASR_DEBUG] augmenter 为 None，直接返回原句")
#         return sentence

#     augmenter = get_asr_augmenter()
#     if augmenter is None:
#         return sentence
#     if not sentence or not sentence.strip():
#         return sentence

#     MAX_OPERATIONS = 2
#     INSERT_PROB = 0.2
#     ALPHA = 0.7
#     RETRY_TIMES = 3

#     def enhance_once(sent):
#         words = _jieba.lcut(sent)
#         if len(words) < 2:
#             return sent

#         candidate_indices = []
#         for i in range(1, len(words)):
#             if words[i-1] in augmenter.prev_to_abnormals:
#                 candidate_indices.append(i)
#         if not candidate_indices:
#             return sent

#         max_ops = min(MAX_OPERATIONS, len(candidate_indices))
#         selected = []
#         shuffled = _random.sample(candidate_indices, len(candidate_indices))
#         for idx in shuffled:
#             if not selected or all(abs(idx - x) >= 2 for x in selected):
#                 selected.append(idx)
#                 if len(selected) >= max_ops:
#                     break

#         operations = []
#         for pos in selected:
#             prev_word = words[pos-1]
#             target_word = words[pos]
#             candidates = augmenter.find_best_abnormals(
#                 target_word,
#                 prev_word=prev_word,
#                 top_k=5,
#                 alpha=ALPHA
#             )
#             if not candidates:
#                 continue
#             chosen = _random.choice(candidates)
#             if _random.random() < INSERT_PROB:
#                 operations.append((pos, chosen, True))
#             else:
#                 operations.append((pos, chosen, False))

#         if not operations:
#             return sent

#         new_words = words[:]
#         for pos, new_word, is_insert in sorted(operations, key=lambda x: x[0], reverse=True):
#             if is_insert:
#                 new_words.insert(pos, new_word)
#             else:
#                 new_words[pos] = new_word
#         return ''.join(new_words)

#     original = sentence
#     for _ in range(RETRY_TIMES):
#         result = enhance_once(original)
#         if result != original:
#             return result
#     return original

def apply_asr_noise(sentence: str) -> str:
    """
    对句子应用 ASR 噪声增强（多位置、替换/插入、前置词匹配）
    增加极性检测，避免肯定/否定词被翻转
    """
    augmenter = get_asr_augmenter()
    if augmenter is None:
        return sentence
    if not sentence or not sentence.strip():
        return sentence

    MAX_OPERATIONS = 2
    INSERT_PROB = 0.2
    ALPHA = 0.7
    RETRY_TIMES = 3

    # 定义肯定词和否定词集合（可根据实际情况扩充）
    AFFIRMATIVE_WORDS = {"是", "有", "能", "可以", "行", "好", "对", "是的", "没错", "肯定", "必须", "需要", "会", "应该"}
    NEGATIVE_WORDS = {"不", "没", "无", "别", "不要", "不用", "不行", "不是", "没有", "不能", "不可以", "否定", "不会", "不该"}

    def enhance_once(sent):
        words = _jieba.lcut(sent)
        if len(words) < 2:
            return sent

        candidate_indices = []
        for i in range(1, len(words)):
            if words[i-1] in augmenter.prev_to_abnormals:
                candidate_indices.append(i)
        if not candidate_indices:
            return sent

        max_ops = min(MAX_OPERATIONS, len(candidate_indices))
        selected = []
        shuffled = _random.sample(candidate_indices, len(candidate_indices))
        for idx in shuffled:
            if not selected or all(abs(idx - x) >= 2 for x in selected):
                selected.append(idx)
                if len(selected) >= max_ops:
                    break

        operations = []
        for pos in selected:
            prev_word = words[pos-1]
            target_word = words[pos]
            candidates = augmenter.find_best_abnormals(
                target_word,
                prev_word=prev_word,
                top_k=5,
                alpha=ALPHA
            )
            if not candidates:
                continue

            # 极性检测：寻找与目标词极性一致的候选词
            chosen = None
            # 确定目标词的极性（如果既不是肯定也不是否定，则极性为 None）
            target_polarity = None
            if target_word in AFFIRMATIVE_WORDS:
                target_polarity = "affirmative"
            elif target_word in NEGATIVE_WORDS:
                target_polarity = "negative"

            # 尝试最多5次选择一个不会翻转极性的候选
            for _ in range(5):
                cand = _random.choice(candidates)
                # 如果目标词没有极性要求，直接接受
                if target_polarity is None:
                    chosen = cand
                    break
                # 检查候选词的极性
                if cand in AFFIRMATIVE_WORDS:
                    cand_polarity = "affirmative"
                elif cand in NEGATIVE_WORDS:
                    cand_polarity = "negative"
                else:
                    cand_polarity = None
                # 如果极性一致（或候选无极性），则接受
                if cand_polarity == target_polarity or cand_polarity is None:
                    chosen = cand
                    break
                # 否则继续尝试下一个

            if chosen is None:
                # 找不到合适的候选，跳过这个位置
                continue

            if _random.random() < INSERT_PROB:
                operations.append((pos, chosen, True))
            else:
                operations.append((pos, chosen, False))

        if not operations:
            return sent

        new_words = words[:]
        for pos, new_word, is_insert in sorted(operations, key=lambda x: x[0], reverse=True):
            if is_insert:
                new_words.insert(pos, new_word)
            else:
                new_words[pos] = new_word
        return ''.join(new_words)

    original = sentence
    for _ in range(RETRY_TIMES):
        result = enhance_once(original)
        if result != original:
            return result
    return original

# ================= 增强函数映射表 =================
AUGMENT_FUNC_MAP = {
    "insert_filler": apply_insert_filler,
    "stutter": apply_stutter,
    "reorder": apply_reorder,
    "homophone": apply_homophone,
    "random_delete": apply_random_delete,
    "random_entity_replace": apply_random_entity_replace,
    "similarword": apply_similarword,
    "word_repetition": apply_word_repetition,
    "asr_noise": apply_asr_noise,        # 新增强化
}

# ================= 多步叠加增强函数 =================

def multi_step_augment(sentence: str, min_steps=1, max_steps=3, weights=None) -> str:
    """
    对句子应用多次随机增强（可重复）
    :param sentence: 原始句子
    :param min_steps: 最少叠加次数
    :param max_steps: 最多叠加次数
    :param weights: 可选，各增强操作的权重字典，格式如 {"insert_filler":2, "stutter":1, ...}
                    若为 None，则使用均匀分布（所有操作等概率）
    :return: 增强后的句子
    """
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence

    # 构建 population 和对应的权重列表
    if weights is not None:
        valid_ops = [(name, w) for name, w in weights.items() if w > 0 and name in AUGMENT_FUNC_MAP]
        if not valid_ops:
            population = list(AUGMENT_FUNC_MAP.values())
            weight_list = None
        else:
            population = [AUGMENT_FUNC_MAP[name] for name, _ in valid_ops]
            weight_list = [w for _, w in valid_ops]
    else:
        population = list(AUGMENT_FUNC_MAP.values())
        weight_list = None

    steps = random.randint(min_steps, max_steps)
    result = sentence
    for _ in range(steps):
        if weight_list is not None:
            func = random.choices(population, weights=weight_list, k=1)[0]
        else:
            func = random.choice(population)
        result = func(result)

    # 如果结果未变且句子不空，重试最多2次
    if result == sentence and len(sentence) > 1:
        for _ in range(2):
            new_result = sentence
            for _ in range(steps):
                if weight_list is not None:
                    func = random.choices(population, weights=weight_list, k=1)[0]
                else:
                    func = random.choice(population)
                new_result = func(new_result)
            if new_result != sentence:
                result = new_result
                break
    return result

def augment_cell_multi(cell_value, num_variants=NUM_VARIANTS, min_steps=1, max_steps=3, augment_weights=None):
    """
    处理一个单元格（可能含 '/' 分隔的多条句子），对每条句子生成 num_variants 个变体。
    返回平铺的变体列表，长度为 (句子数 × num_variants)。
    
    参数：
        cell_value: 输入字符串，可能包含 '/' 分隔的多句话
        num_variants: 每句话生成的变体数量
        min_steps: 每个变体最少增强步数
        max_steps: 每个变体最多增强步数
        augment_weights: 权重字典，传递给 multi_step_augment
    """
    if pd.isna(cell_value):
        return []
    
    raw_sentences = [s.strip() for s in str(cell_value).split('/') if s.strip()]
    if not raw_sentences:
        return []
    
    all_variants = []
    for sent in raw_sentences:
        for _ in range(num_variants):
            variant = multi_step_augment(sent, min_steps, max_steps, weights=augment_weights)
            all_variants.append(variant)
    return all_variants

# ================= 辅助函数 =================
def move_column_to_right(df, col_name, new_col_name):
    """将新列移动到原列右侧"""
    cols = df.columns.tolist()
    if new_col_name not in cols:
        return df
    idx = cols.index(col_name)
    cols.remove(new_col_name)
    cols.insert(idx + 1, new_col_name)
    return df[cols]