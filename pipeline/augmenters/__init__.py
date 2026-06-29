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
    RandomEntityReplaceAugmenter,
    SynonymAugmenter,
    WordRepetitionAugmenter,
)
from .methods.other import ReorderAugmenter
from .methods.model import AsrNoiseAugmenter


# ---------- 注册（含别名以兼容旧配置里的 similarword / entity_replace 等名字）----------
AugmenterRegistry.register(
    "insert_filler", InsertFillerAugmenter, category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "stutter", StutterAugmenter, category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "homophone", HomophoneAugmenter, category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "random_delete", RandomDeleteAugmenter, category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "random_entity_replace", RandomEntityReplaceAugmenter,
    aliases=("entity_replace",), category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "synonym_replace", SynonymAugmenter,
    aliases=("similarword", "synonym"), category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "word_repetition", WordRepetitionAugmenter, category=CATEGORY_LEXICAL
)
AugmenterRegistry.register(
    "reorder", ReorderAugmenter, category=CATEGORY_ORDER
)
AugmenterRegistry.register(
    "asr_noise", AsrNoiseAugmenter, category=CATEGORY_MODEL
)


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
    "RandomEntityReplaceAugmenter",
    "SynonymAugmenter",
    "WordRepetitionAugmenter",
    "ReorderAugmenter",
    "AsrNoiseAugmenter",
]
