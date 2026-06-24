# common/logger.py
import logging
from pathlib import Path
from typing import Optional

def setup_logger(
    logger_name: str,
    log_dir: Path,
    log_filename: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    formatter: Optional[logging.Formatter] = None,
) -> logging.Logger:
    """
    配置日志：控制台输出指定级别，文件输出更详细级别。
    
    Args:
        logger_name: 日志器名称（如 "Pipeline", "Augment"）
        log_dir: 日志文件目录
        log_filename: 日志文件名（若为 None，则自动生成如 f"{logger_name}.log"）
        console_level: 控制台日志级别，默认 INFO
        file_level: 文件日志级别，默认 DEBUG
        formatter: 自定义格式，若为 None 则使用默认格式
        
    Returns:
        配置好的 Logger 实例
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_filename is None:
        log_filename = f"{logger_name}.log"
    log_file = log_dir / log_filename
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)   # 全局最低级别，由 handler 控制过滤
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件 handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(file_level)
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    # 格式器
    if formatter is None:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger