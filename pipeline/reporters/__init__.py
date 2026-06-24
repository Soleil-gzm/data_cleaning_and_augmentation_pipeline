from .registry import ReporterRegistry
from .json_reporter import JsonReporter
from .csv_reporter import CsvReporter
from .matplotlib_reporter import MatplotlibReporter

ReporterRegistry.register("json", JsonReporter)
ReporterRegistry.register("csv", CsvReporter)
ReporterRegistry.register("matplotlib", MatplotlibReporter)