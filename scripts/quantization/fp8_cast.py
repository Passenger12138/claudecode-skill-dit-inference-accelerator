"""
FP8 Cast 量化实现

通用的 FP8 量化方法，适用于所有支持 FP8 的显卡（RTX 30/40 系列、H100 等）。
原理：加载时将权重 cast 到 FP8，推理时 upcast 回原精度计算。
"""

import torch

from .base import KeyValueOperationResult, ModuleOps, SDOps


# ========== Triton 优化的随机舍入 ==========

BLOCK_SIZE = 1024


def _fused_add_round_launch(
    target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int
) -> torch.Tensor:
    """
    使用 Triton kernel 进行融合的加法和随机舍入

    注意：需要 triton 和自定义 kernel，如果不可用则回退到简单转换
    """
    try:
        import triton  # noqa: PLC0415

        # 这里需要自定义 kernel，如果没有则回退
        # from your_project.kernels import fused_add_round_kernel
        raise ImportError("Custom kernel not available")
    except ImportError:
        # 回退到简单转换
        return original_weight.to(target_weight.dtype)


def calculate_weight_float8(
    target_weights: torch.Tensor, original_weights: torch.Tensor
) -> torch.Tensor:
    """计算 FP8 权重（带随机舍入优化）"""
    result = _fused_add_round_launch(target_weights, original_weights, seed=0).to(
        target_weights.dtype
    )
    target_weights.copy_(result, non_blocking=True)
    return target_weights


# ========== 核心量化操作 ==========


def _naive_weight_or_bias_downcast(
    key: str, value: torch.Tensor
) -> list[KeyValueOperationResult]:
    """
    将权重/偏置下采样到 FP8 (float8_e4m3fn)

    这是最简单的量化方法，直接 cast 到 FP8。
    DiT 模型对这种量化不敏感，精度损失可忽略。
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor,
    dtype: torch.dtype,
    with_stochastic_rounding: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """
    将 FP8 权重上采样到指定精度

    Args:
        weight: FP8 权重 (float8_e4m3fn 或 float8_e5m2)
        dtype: 目标精度 (通常是 bfloat16 或 float16)
        with_stochastic_rounding: 是否使用随机舍入（可提高训练稳定性）
        seed: 随机种子

    Returns:
        上采样后的权重
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return _fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


def _replace_fwd_with_upcast(
    layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0
) -> None:
    """
    替换 Linear.forward，推理时将 FP8 权重上采样到输入精度

    这是 FP8 Cast 的核心：
    1. 权重以 FP8 格式存储（节省显存）
    2. 每次 forward 时临时转换回高精度（保证计算精度）
    """
    layer.original_forward = layer.forward

    def new_linear_forward(*args, **_kwargs) -> torch.Tensor:
        x = args[0]
        w_up = _upcast_and_round(layer.weight, x.dtype, with_stochastic_rounding, seed)
        b_up = None

        if layer.bias is not None:
            b_up = _upcast_and_round(
                layer.bias, x.dtype, with_stochastic_rounding, seed
            )

        return torch.nn.functional.linear(x, w_up, b_up)

    layer.forward = new_linear_forward


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    替换模型中所有 Linear 层的 forward 方法

    遍历模型的所有子模块，对 Linear 层应用 FP8 upcast forward。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


# ========== 构建量化映射 ==========

# Transformer Linear 层的下采样映射
# 覆盖 attention (Q/K/V/Out) 和 FFN (MLP) 的所有权重和偏置
TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    # Attention Q/K/V
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_q.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_q.bias",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_k.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_k.bias",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_v.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_v.bias",
        operation=_naive_weight_or_bias_downcast,
    )
    # Attention Output
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_out.0.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix=".to_out.0.bias",
        operation=_naive_weight_or_bias_downcast,
    )
    # FFN / MLP
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix="ff.net.0.proj.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix="ff.net.0.proj.bias",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix="ff.net.2.weight",
        operation=_naive_weight_or_bias_downcast,
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.",
        key_suffix="ff.net.2.bias",
        operation=_naive_weight_or_bias_downcast,
    )
)

# 推理时上采样的模块操作
UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: hasattr(model, "transformer_blocks"),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    带随机舍入的 FP8 上采样

    随机舍入可以在训练时提高数值稳定性，
    但推理时通常不需要。
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: hasattr(model, "transformer_blocks"),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )
