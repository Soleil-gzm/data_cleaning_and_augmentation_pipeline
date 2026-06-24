"""
进度条工具：支持 tqdm，若无则降级为简单迭代
"""
import sys
from typing import Iterable, Optional, Any, Callable
from tqdm import tqdm as _tqdm

# 全局开关，由上下文注入
_SHOW_PROGRESS = True


def set_progress_global(show: bool):
    global _SHOW_PROGRESS
    _SHOW_PROGRESS = show


def get_progress_bar(
    iterable: Iterable,
    desc: str = "Processing",
    unit: str = "it",
    total: Optional[int] = None,
    show: Optional[bool] = None,
    **kwargs
):
    """
    获取进度条迭代器
    """
    if show is None:
        show = _SHOW_PROGRESS

    if not show:
        return iterable

    return _tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        file=sys.stdout,
        **kwargs
    )


def progress_wrapper(func: Callable, desc: str, show: Optional[bool] = None):
    """
    装饰器：为函数调用显示单次进度（例如长时间运行的函数）
    """
    if show is None:
        show = _SHOW_PROGRESS

    if not show:
        return func()

    # 简单的脉冲进度条
    with _tqdm(total=1, desc=desc, unit="task", file=sys.stdout) as pbar:
        result = func()
        pbar.update(1)
        return result