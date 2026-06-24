from .registry import AnalyzerRegistry
from .retention import RetentionAnalyzer
from .turn_distribution import TurnDistributionAnalyzer

AnalyzerRegistry.register("RetentionAnalyzer", RetentionAnalyzer)
AnalyzerRegistry.register("TurnDistributionAnalyzer", TurnDistributionAnalyzer)