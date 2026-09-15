import importlib
import logging
import os
import inspect
from pathlib import Path
from ._base import Metric as MetricABC

_logger = logging.getLogger(__name__)


_modules = os.listdir(os.path.dirname(__file__))
_all = filter(lambda x: not x.startswith(('_', '.')),_modules)
__all__ = []
for file in _all:
    module = Path(file).stem
    try:
        mod = importlib.import_module(f'.{module}', package=__name__)
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__:
                globals()[name] = cls
                if issubclass(cls, MetricABC):
                    __all__.append(cls)
    except ImportError as e:
        _logger.warning(f'Directory "{module}" could not be imported; Ignoring directory')
