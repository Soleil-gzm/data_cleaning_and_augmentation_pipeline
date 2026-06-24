"""
统一日志配置
"""

import logging
from pathlib import Path
from typing import Union


def setup_task_logger(
    task_name: str,
    log_dir: Union[str, Path],
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """
    为任务配置日志：控制台 INFO，文件 DEBUG。
    返回配置好的 Logger 对象。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{task_name}.log"

    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.DEBUG)
    # 清除已有 handlers，避免重复
    if logger.handlers:
        logger.handlers.clear()

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
