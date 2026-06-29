"""
增强方法集合（按类别分组）
========================
- lexical:  词法/替换级增强
- order:    语序/重排级增强
- model:    需要模型/预训练能力的增强
"""
from . import lexical as _lexical
from . import other as _order
from . import model as _model  # noqa: F401

LEXICAL = _lexical
ORDER = _order
MODEL = _model

__all__ = ["LEXICAL", "ORDER", "MODEL"]
