"""
增强器模块入口：注册所有增强器
"""
from .base import BaseAugmenter, AugmenterRegistry
from .composite import CompositeAugmenter
from .synonym import SynonymAugmenter
from .stutter import StutterAugmenter
from .reorder import ReorderAugmenter
from .word_repetition import WordRepetitionAugmenter
from .insert_filler import InsertFillerAugmenter
from .homophone import HomophoneAugmenter
from .random_delete import RandomDeleteAugmenter
from .random_entity_replace import RandomEntityReplaceAugmenter
from .asr_noise import AsrNoiseAugmenter

# 注册所有增强器
AugmenterRegistry.register("synonym_replace", SynonymAugmenter)
AugmenterRegistry.register("stutter", StutterAugmenter)
AugmenterRegistry.register("reorder", ReorderAugmenter)
AugmenterRegistry.register("word_repetition", WordRepetitionAugmenter)
AugmenterRegistry.register("insert_filler", InsertFillerAugmenter)
AugmenterRegistry.register("homophone", HomophoneAugmenter)
AugmenterRegistry.register("random_delete", RandomDeleteAugmenter)
AugmenterRegistry.register("random_entity_replace", RandomEntityReplaceAugmenter)
AugmenterRegistry.register("asr_noise", AsrNoiseAugmenter)

__all__ = [
    "BaseAugmenter",
    "AugmenterRegistry",
    "CompositeAugmenter",
    "SynonymAugmenter",
    "StutterAugmenter",
    "ReorderAugmenter",
    "WordRepetitionAugmenter",
    "InsertFillerAugmenter",
    "HomophoneAugmenter",
    "RandomDeleteAugmenter",
    "RandomEntityReplaceAugmenter",
    "AsrNoiseAugmenter",
]