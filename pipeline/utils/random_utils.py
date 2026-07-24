"""
统一随机数工具模块
==================
提供统一的随机数接口，支持全局随机和指定随机数生成器。
所有函数都接受可选的 rng 参数，若为 None 则使用全局 random 模块。

使用方式：
    from ..utils.random_utils import rand, choice, choices, randint, sample, RandomGenerator

    # 使用全局随机
    val = rand()
    item = choice([1, 2, 3])

    # 使用指定随机数生成器
    rng = RandomGenerator(42)
    val = rand(rng=rng)
    item = choice([1, 2, 3], rng=rng)
"""

import random
from typing import Optional, Sequence, List, Any


def rand(rng: Optional[random.Random] = None) -> float:
    """返回 [0.0, 1.0) 范围内的随机浮点数"""
    if rng is None:
        return random.random()
    return rng.random()


def choice(seq: Sequence[Any], rng: Optional[random.Random] = None) -> Any:
    """从序列中随机选择一个元素"""
    if rng is None:
        return random.choice(seq)
    return rng.choice(seq)


def choices(
    population: Sequence[Any],
    weights: Optional[Sequence[float]] = None,
    rng: Optional[random.Random] = None,
    k: int = 1,
) -> List[Any]:
    """从总体中随机选择 k 个元素，可指定权重"""
    if rng is None:
        return random.choices(population, weights=weights, k=k)
    return rng.choices(population, weights=weights, k=k)


def randint(a: int, b: int, rng: Optional[random.Random] = None) -> int:
    """返回 [a, b] 范围内的随机整数"""
    if rng is None:
        return random.randint(a, b)
    return rng.randint(a, b)


def sample(
    population: Sequence[Any],
    k: int,
    rng: Optional[random.Random] = None,
) -> List[Any]:
    """从总体中随机选择 k 个不重复的元素"""
    if rng is None:
        return random.sample(population, k)
    return rng.sample(population, k)


def shuffle(seq: List[Any], rng: Optional[random.Random] = None):
    """随机打乱序列（原地修改）"""
    if rng is None:
        random.shuffle(seq)
    else:
        rng.shuffle(seq)


class RandomGenerator:
    """
    随机数生成器封装类
    提供与 random.Random 相同的接口，便于统一管理随机种子
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    @property
    def rng(self) -> random.Random:
        return self._rng

    def rand(self) -> float:
        return self._rng.random()

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._rng.choice(seq)

    def choices(
        self,
        population: Sequence[Any],
        weights: Optional[Sequence[float]] = None,
        k: int = 1,
    ) -> List[Any]:
        return self._rng.choices(population, weights=weights, k=k)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def sample(self, population: Sequence[Any], k: int) -> List[Any]:
        return self._rng.sample(population, k)

    def shuffle(self, seq: List[Any]):
        self._rng.shuffle(seq)

    def random(self) -> float:
        return self._rng.random()


__all__ = [
    "rand",
    "choice",
    "choices",
    "randint",
    "sample",
    "shuffle",
    "RandomGenerator",
]