"""
导入所有步骤以自动注册
"""
from ..core.step_registry import StepRegistry
from .generate_raw import GenerateRawStep
from .split_dialogues import SplitDialoguesStep
from .bucket import BucketStep
from .clean import CleanStep
from .finalize import FinalizeStep
from .augment import AugmentStep
from .replace_text import ReplaceTextStep

# 注册
StepRegistry.register("00_generate_raw", GenerateRawStep)
StepRegistry.register("01_split", SplitDialoguesStep)
StepRegistry.register("02_bucket", BucketStep)
StepRegistry.register("03_clean", CleanStep)
StepRegistry.register("04_finalize", FinalizeStep)
StepRegistry.register("05_augment", AugmentStep)
StepRegistry.register("06_replace_text", ReplaceTextStep)