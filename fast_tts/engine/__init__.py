# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:04
# Author    :Hui Huang
from ..import_utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "base_engine": [
        "BaseEngine"
    ],
    "spark_engine": ["AsyncSparkEngine", "SparkAcousticTokens"],
    "orpheus_engine": ["AsyncOrpheusEngine"]
}

if TYPE_CHECKING:
    from .base_engine import BaseEngine
    from .spark_engine import AsyncSparkEngine, SparkAcousticTokens
    from .orpheus_engine import AsyncOrpheusEngine
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
