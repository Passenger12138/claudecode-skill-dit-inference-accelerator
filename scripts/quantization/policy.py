"""
量化策略配置

提供统一的量化策略接口，支持：
- fp8_cast: 通用 FP8 量化
- fp8_scaled_mm: Hopper 显卡专用 FP8 量化
"""

from dataclasses import dataclass

from .base import ModuleOps, SDOps
from .fp8_cast import TRANSFORMER_LINEAR_DOWNCAST_MAP, UPCAST_DURING_INFERENCE


@dataclass(frozen=True)
class QuantizationPolicy:
    """
    模型量化配置

    Attributes:
        sd_ops: 状态字典操作，用于加载时转换权重
        module_ops: 模块操作，用于加载后变换模块
    """

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()

    @classmethod
    def none(cls) -> "QuantizationPolicy":
        """不使用量化"""
        return cls()

    @classmethod
    def fp8_cast(cls) -> "QuantizationPolicy":
        """
        FP8 Cast 量化策略（通用）

        适用于所有支持 FP8 的显卡（RTX 30/40 系列、H100 等）。
        原理：加载时将权重 cast 到 FP8，推理时 upcast 回原精度计算。
        """
        return cls(
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(UPCAST_DURING_INFERENCE,),
        )

    @classmethod
    def fp8_scaled_mm(cls) -> "QuantizationPolicy":
        """
        FP8 Scaled MM 量化策略（Hopper 显卡专用）

        需要 TensorRT-LLM 支持。
        原理：使用 cuBLAS 的 scaled matrix multiplication 进行高效 FP8 计算。
        """
        try:
            import tensorrt_llm  # noqa: F401, PLC0415
        except ImportError as e:
            raise ImportError(
                "tensorrt_llm 未安装，无法使用 fp8_scaled_mm。\n"
                "请安装 TensorRT-LLM 或使用 fp8_cast 作为替代。"
            ) from e

        # 延迟导入，避免在 TensorRT-LLM 未安装时报错
        from .fp8_scaled_mm import FP8_PREPARE_MODULE_OPS, FP8_TRANSPOSE_SD_OPS

        return cls(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    def is_quantized(self) -> bool:
        """是否启用了量化"""
        return self.sd_ops is not None or len(self.module_ops) > 0
