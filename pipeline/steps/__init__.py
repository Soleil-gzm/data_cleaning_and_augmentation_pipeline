"""
导入所有步骤以自动注册
"""

from ..core.step_registry import StepRegistry
from .split_dialogues import SplitDialoguesStep
from .bucket import BucketStep
from .clean import CleanStep
from .finalize import FinalizeStep
from .augment import AugmentStep

# 注册
StepRegistry.register("01_split", SplitDialoguesStep)
StepRegistry.register("02_bucket", BucketStep)
StepRegistry.register("03_clean", CleanStep)
StepRegistry.register("04_finalize", FinalizeStep)
StepRegistry.register("05_augment", AugmentStep)