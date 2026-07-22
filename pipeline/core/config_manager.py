"""
配置管理器：管理流水线的所有配置读取和查询

职责：
- 统一管理配置字典
- 提供步骤配置查询接口
- 提供步骤启用状态判断

使用示例：
    config_manager = ConfigManager(config_dict)
    step_cfg = config_manager.get_step_config("01_split")
    is_enabled = config_manager.is_step_enabled("01_split")
"""

from typing import Dict, Any


class ConfigManager:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._task_name = config.get("task_name", "default_task")
        self._resume = config.get("resume", False)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def resume(self) -> bool:
        return self._resume

    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """获取步骤配置"""
        steps = self._config.get("steps", {})
        return steps.get(step_name, {})

    def is_step_enabled(self, step_name: str) -> bool:
        """判断步骤是否启用"""
        step_cfg = self.get_step_config(step_name)
        return step_cfg.get("enabled", True)