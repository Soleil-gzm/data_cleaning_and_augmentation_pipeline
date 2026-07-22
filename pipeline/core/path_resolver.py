"""
路径解析器：统一管理项目中所有路径的解析逻辑

支持三种路径格式：
1. 绝对路径：以 '/' 开头或包含驱动器（Windows），直接返回
2. 占位符路径：包含 '{task_dir}'、'{task_name}'、'{timestamp}' 等占位符
3. 相对路径：相对于项目根目录

安全特性：
- 占位符替换使用安全字典替换，避免 KeyError
- 初始化时不创建任何目录，避免权限问题
- project_root 从配置中显式获取，避免推断错误

使用示例：
    resolver = PathResolver(config)
    path = resolver.resolve("{task_dir}/samples")  # -> /path/to/intermediate/task_name/samples
    path = resolver.resolve("configs/config.yaml")  # -> /path/to/project_root/configs/config.yaml
    path = resolver.resolve("/absolute/path/file.json")  # -> /absolute/path/file.json
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class PathResolver:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._task_name = config.get("task_name", "default_task")
        self._intermediate_root = Path(
            config.get("paths", {}).get("intermediate", "./intermediate")
        )
        self._output_root = Path(config.get("paths", {}).get("output", "./output"))
        self._task_dir = self._intermediate_root / self._task_name
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._project_root = self._resolve_project_root(config)

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    @property
    def intermediate_root(self) -> Path:
        return self._intermediate_root

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _resolve_project_root(self, config: Dict[str, Any]) -> Path:
        """
        解析项目根目录：
        1. 如果配置中有显式的 project_root，直接使用
        2. 如果 intermediate_root 是相对路径，其父目录为项目根
        3. 如果 intermediate_root 是绝对路径，返回当前工作目录
        """
        explicit_root = config.get("paths", {}).get("project_root")
        if explicit_root:
            return Path(explicit_root).resolve()

        if self._intermediate_root.is_absolute():
            return Path.cwd()

        return (self._intermediate_root.parent).resolve()

    def resolve(self, path_str: str) -> Path:
        """
        解析路径：
        - 若以 '/' 开头或包含驱动器（Windows），视为绝对路径，直接返回。
        - 若包含占位符（{task_dir}, {task_name}, {timestamp}），安全替换后返回。
        - 否则，视为相对于项目根目录。

        安全特性：只替换已知的占位符，遇到未知占位符不会抛出异常。
        """
        p = Path(path_str)
        if p.is_absolute():
            return p

        placeholders = {
            "{task_dir}": str(self._task_dir),
            "{task_name}": self._task_name,
            "{timestamp}": self._timestamp,
            "{intermediate_root}": str(self._intermediate_root),
            "{output_root}": str(self._output_root),
        }

        resolved = path_str
        for placeholder, value in placeholders.items():
            resolved = resolved.replace(placeholder, value)

        p_resolved = Path(resolved)
        if p_resolved.is_absolute():
            return p_resolved

        return self._project_root / p_resolved

    def get_step_output_dir(self, step_name: str, default_subdir: str = None) -> Path:
        """
        获取步骤输出目录：
        1. 优先从步骤配置中读取 output_dir
        2. 其次使用 default_subdir
        3. 最后使用步骤名作为子目录
        """
        steps = self._config.get("steps", {})
        step_cfg = steps.get(step_name, {})
        out_dir = step_cfg.get("output_dir")
        if out_dir:
            return self.resolve(out_dir)
        if default_subdir:
            return self._task_dir / default_subdir
        return self._task_dir / step_name

    def get_path(self, path_key: str) -> Path:
        """
        从配置的 data_paths 中获取预定义路径
        """
        paths = self._config.get("data_paths", {})
        path_str = paths.get(path_key)
        if path_str:
            return self.resolve(path_str)
        raise KeyError(f"未定义路径: {path_key}")

    def ensure_dir(self, path: Path) -> Path:
        """确保目录存在，不存在则创建"""
        path.mkdir(parents=True, exist_ok=True)
        return path