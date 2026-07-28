"""
增强器模块入口：注册所有增强器（含别名以兼容旧配置）
按分类组织：
    methods/lexical  - 词法/替换级
    methods/order    - 语序/重排级
    methods/model    - 模型/ASR 级
"""
from .base import BaseAugmenter, AugmenterRegistry
from .composite import CompositeAugmenter
from .categories import (
    CATEGORY_LEXICAL,
    CATEGORY_ORDER,
    CATEGORY_MODEL,
    CATEGORY_LABELS,
    AUGMENTER_META,
    get_category,
    requires_model,
    default_weight,
)

from .methods.lexical import (
    InsertFillerAugmenter,
    StutterAugmenter,
    HomophoneAugmenter,
    RandomDeleteAugmenter,
    SynonymAugmenter,
    WordRepetitionAugmenter,
)
from .methods.other import ReorderAugmenter
from .methods.model import AsrNoiseAugmenter


# ---------- 注册（分类信息已统一由 categories.AUGMENTER_META 管理）----------
AugmenterRegistry.register("insert_filler", InsertFillerAugmenter)
AugmenterRegistry.register("stutter", StutterAugmenter)
AugmenterRegistry.register("homophone", HomophoneAugmenter)
AugmenterRegistry.register("random_delete", RandomDeleteAugmenter)
AugmenterRegistry.register(
    "synonym_replace", SynonymAugmenter,
    aliases=("similarword", "synonym")
)
AugmenterRegistry.register("word_repetition", WordRepetitionAugmenter)
AugmenterRegistry.register("reorder", ReorderAugmenter)
AugmenterRegistry.register("asr_noise", AsrNoiseAugmenter)


__all__ = [
    "BaseAugmenter",
    "AugmenterRegistry",
    "CompositeAugmenter",
    "CATEGORY_LEXICAL",
    "CATEGORY_ORDER",
    "CATEGORY_MODEL",
    "CATEGORY_LABELS",
    "AUGMENTER_META",
    "get_category",
    "requires_model",
    "default_weight",
    "InsertFillerAugmenter",
    "StutterAugmenter",
    "HomophoneAugmenter",
    "RandomDeleteAugmenter",
    "SynonymAugmenter",
    "WordRepetitionAugmenter",
    "ReorderAugmenter",
    "AsrNoiseAugmenter",
]
