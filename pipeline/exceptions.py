"""
统一异常体系
============
定义项目中使用的自定义异常，便于统一捕获和处理
"""


class PipelineError(Exception):
    """管道执行异常基类"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class PipelineIOError(PipelineError):
    """IO 操作异常"""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (原因: {self.cause})"
        return self.message


class PipelineConfigError(PipelineError):
    """配置错误异常"""

    pass


class PipelineValidationError(PipelineError):
    """数据验证异常"""

    pass


class PipelineStepError(PipelineError):
    """步骤执行异常"""

    def __init__(self, message: str, step_name: str = None):
        super().__init__(message)
        self.step_name = step_name

    def __str__(self) -> str:
        if self.step_name:
            return f"步骤 [{self.step_name}] 执行失败: {self.message}"
        return f"步骤执行失败: {self.message}"


class PipelineRuntimeError(PipelineError):
    """运行时异常"""

    pass


__all__ = [
    "PipelineError",
    "PipelineIOError",
    "PipelineConfigError",
    "PipelineValidationError",
    "PipelineStepError",
    "PipelineRuntimeError",
]