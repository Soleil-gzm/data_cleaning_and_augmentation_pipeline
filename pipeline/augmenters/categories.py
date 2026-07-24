"""
增强方法分类元信息
==================
- lexical  词法/替换级：insert_filler、stutter、homophone、random_delete、
                       synonym_replace、word_repetition
- order    语序/重排级：reorder
- model    模型级（需要加载语义/ASR 模型）：asr_noise
"""

CATEGORY_LEXICAL = "lexical"
CATEGORY_ORDER = "order"
CATEGORY_MODEL = "model"

CATEGORY_LABELS = {
    CATEGORY_LEXICAL: "词法/替换",
    CATEGORY_ORDER: "语序/重排",
    CATEGORY_MODEL: "模型/ASR",
}

CATEGORY_NAMES = (CATEGORY_LEXICAL, CATEGORY_ORDER, CATEGORY_MODEL)

AUGMENTER_META = {
    "insert_filler":         {"category": CATEGORY_LEXICAL, "default_weight": 1.0,  "requires_model": False},
    "stutter":               {"category": CATEGORY_LEXICAL, "default_weight": 1.0,  "requires_model": False},
    "homophone":             {"category": CATEGORY_LEXICAL, "default_weight": 1.0,  "requires_model": False},
    "random_delete":         {"category": CATEGORY_LEXICAL, "default_weight": 1.0,  "requires_model": False},
    "synonym_replace":       {"category": CATEGORY_LEXICAL, "default_weight": 1.0,  "requires_model": False},
    "word_repetition":       {"category": CATEGORY_LEXICAL, "default_weight": 2.0,  "requires_model": False},
    "reorder":               {"category": CATEGORY_ORDER,   "default_weight": 1.0,  "requires_model": False},
    "asr_noise":             {"category": CATEGORY_MODEL,    "default_weight": 1.0,  "requires_model": True},
}


def get_category(name: str) -> str:
    """返回指定增强器的 category，未注册则返回 lexical 作为安全默认"""
    meta = AUGMENTER_META.get(name, {})
    return meta.get("category", CATEGORY_LEXICAL)


def requires_model(name: str) -> bool:
    meta = AUGMENTER_META.get(name, {})
    return bool(meta.get("requires_model", False))


def default_weight(name: str) -> float:
    meta = AUGMENTER_META.get(name, {})
    return float(meta.get("default_weight", 1.0))
