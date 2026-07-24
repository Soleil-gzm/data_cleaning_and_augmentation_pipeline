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
    
    配置根 logger，所有子 logger（包括步骤、分析器的）都会继承 handlers。
    这样控制台和文件都会输出完整的日志。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{task_name}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # 清除已有 handlers，避免重复
    if root_logger.handlers:
        root_logger.handlers.clear()

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

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    # 过滤第三方库的 DEBUG 日志，避免日志文件过大
    _suppress_third_party_logs()

    # 返回 Pipeline logger 供调用方使用
    return logging.getLogger("Pipeline")


def _suppress_third_party_logs():
    """
    抑制第三方库的 DEBUG 日志，避免日志文件过大。
    """
    third_party_loggers = [
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "torch",
        "transformers",
        "sentence_transformers",
        "data_juicer",
        "tqdm",
        "paramiko",
        "botocore",
        "urllib3",
        "requests",
        "shap",
        "sklearn",
        "numpy",
    ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
