"""
DiT 模型量化模块

提供 FP8 量化的两种实现：
- fp8_cast: 通用方法，适用于所有支持 FP8 的显卡
- fp8_scaled_mm: Hopper 显卡专用，需要 TensorRT-LLM
"""

from .base import ContentMatching, KeyValueOperationResult, ModuleOps, SDOps
from .fp8_cast import (
    TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    UpcastWithStochasticRounding,
)
from .policy import QuantizationPolicy

__all__ = [
    # 基础设施
    "SDOps",
    "ModuleOps",
    "KeyValueOperationResult",
    "ContentMatching",
    # FP8 Cast
    "TRANSFORMER_LINEAR_DOWNCAST_MAP",
    "UPCAST_DURING_INFERENCE",
    "UpcastWithStochasticRounding",
    # 量化策略
    "QuantizationPolicy",
]

# 尝试导入 FP8 Scaled MM（需要 TensorRT-LLM）
try:
    from .fp8_scaled_mm import (
        EXCLUDED_LAYER_SUBSTRINGS,
        FP8_PREPARE_MODULE_OPS,
        FP8_TRANSPOSE_SD_OPS,
        FP8Linear,
        quantize_weight_to_fp8_per_tensor,
    )

    __all__.extend(
        [
            "FP8Linear",
            "FP8_TRANSPOSE_SD_OPS",
            "FP8_PREPARE_MODULE_OPS",
            "EXCLUDED_LAYER_SUBSTRINGS",
            "quantize_weight_to_fp8_per_tensor",
        ]
    )
except ImportError:
    pass  # TensorRT-LLM not installed
