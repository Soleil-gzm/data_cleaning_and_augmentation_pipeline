"""
状态追踪器：管理流水线的断点续跑状态

职责：
- 记录步骤执行状态（完成/未完成）
- 提供状态查询接口
- 提供状态更新接口

使用示例：
    state_tracker = StateTracker(task_dir)
    if state_tracker.is_step_done("01_split"):
        print("步骤已完成")
    state_tracker.mark_step_done("01_split")
"""

from pathlib import Path


class StateTracker:
    def __init__(self, task_dir: Path):
        self._task_dir = task_dir

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    def is_step_done(self, step_name: str) -> bool:
        """判断步骤是否已完成"""
        flag_path = self._task_dir / f".step_{step_name}_done"
        return flag_path.exists()

    def mark_step_done(self, step_name: str):
        """标记步骤为已完成"""
        flag_path = self._task_dir / f".step_{step_name}_done"
        flag_path.touch()

    def clear_step_done(self, step_name: str):
        """清除步骤完成标记"""
        flag_path = self._task_dir / f".step_{step_name}_done"
        if flag_path.exists():
            flag_path.unlink()

    def clear_all_done(self):
        """清除所有步骤完成标记"""
        for flag_file in self._task_dir.glob(".step_*_done"):
            flag_file.unlink()

    def get_completed_steps(self) -> list:
        """获取所有已完成的步骤名"""
        completed = []
        for flag_file in self._task_dir.glob(".step_*_done"):
            step_name = flag_file.name.replace(".step_", "").replace("_done", "")
            completed.append(step_name)
        return completed