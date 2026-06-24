"""
子进程执行工具
"""

import subprocess
import logging
from typing import List, Union, Optional
from pathlib import Path


def run_subprocess(
    cmd: List[Union[str, Path]],
    logger: Optional[logging.Logger] = None,
    capture_output: bool = True,
    check: bool = False,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """
    执行子进程，并记录日志。返回 CompletedProcess 对象。
    """
    cmd_str = " ".join(str(c) for c in cmd)
    if logger:
        logger.debug(f"执行命令: {cmd_str}")

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd=cwd,
        env=env,
    )

    if logger:
        if result.returncode != 0:
            logger.error(f"命令执行失败，返回码 {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[:1000]}")
        else:
            logger.debug("命令执行成功")

    return result
