"""
增强器注册表（从 base 分离，避免循环导入）
"""
from .base import AugmenterRegistry

__all__ = ["AugmenterRegistry"]