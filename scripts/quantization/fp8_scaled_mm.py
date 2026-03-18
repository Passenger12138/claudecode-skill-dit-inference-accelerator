"""
FP8 Scaled MM 量化实现

Hopper 显卡（H100、H200）专用的 FP8 量化方法。
使用 TensorRT-LLM 的 cublas_scaled_mm 进行高效的 FP8 矩阵乘法。
"""

from typing import Callable

import torch
from torch import nn

from .base import KeyValueOperationResult, ModuleOps, SDOps


class FP8Linear(nn.Module):
    """
    FP8 Linear 层，使用缩放矩阵乘法

    与普通 Linear 的区别：
    1. 权重以 FP8 格式存储（转置以适配 cuBLAS）
    2. 使用 per-tensor 缩放因子进行量化/反量化
    3. 使用 TensorRT-LLM 的 cublas_scaled_mm 进行高效计算
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 权重（转置存储以适配 cuBLAS: shape = [in, out]）
        fp8_shape = (in_features, out_features)
        self.weight = nn.Parameter(
            torch.empty(fp8_shape, dtype=torch.float8_e4m3fn, device=device)
        )

        # 权重缩放因子（用于反量化）
        self.weight_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32, device=device)
        )

        # 输入缩放因子（静态量化）
        self.input_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32, device=device)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        # 静态量化输入
        qinput, cur_input_scale = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, self.input_scale
        )

        # 展平为 2D 进行矩阵乘法
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])

        # FP8 缩放矩阵乘法
        output = torch.ops.trtllm.cublas_scaled_mm(
            qinput,
            self.weight,
            scale_a=cur_input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=x.dtype,
        )

        # 添加偏置
        if self.bias is not None:
            bias = self.bias
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        # 恢复原始形状
        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        return output


def quantize_weight_to_fp8_per_tensor(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将权重量化到 FP8，使用 per-tensor 缩放

    Args:
        weight: 原始权重张量（任意 dtype，会转为 float32 处理）

    Returns:
        (quantized_weight, weight_scale):
        - quantized_weight: FP8 权重，已转置适配 cublas_scaled_mm
        - weight_scale: per-tensor 缩放因子（量化缩放的倒数）
    """
    weight_fp32 = weight.to(torch.float32)

    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # 计算缩放因子：将权重范围映射到 FP8 范围
    max_abs = torch.amax(torch.abs(weight_fp32))
    scale = fp8_max / max_abs

    @torch.compiler.disable
    def _quantize(
        weight_fp32: torch.Tensor,
        scale: torch.Tensor,
        fp8_min: float,
        fp8_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_weight = torch.clamp(weight_fp32 * scale, min=fp8_min, max=fp8_max).to(
            torch.float8_e4m3fn
        )
        # 转置以适配 cuBLAS 布局
        quantized_weight = quantized_weight.t()
        weight_scale = scale.reciprocal()
        return quantized_weight, weight_scale

    quantized_weight, weight_scale = _quantize(weight_fp32, scale, fp8_min, fp8_max)
    return quantized_weight, weight_scale


def _should_skip_layer(layer_name: str, excluded_layer_substrings: tuple[str, ...]) -> bool:
    """判断是否应该跳过该层（不量化）"""
    return any(substring in layer_name for substring in excluded_layer_substrings)


# 需要排除的层（保持高精度）
# 这些层对量化敏感，保持原精度可提高生成质量
EXCLUDED_LAYER_SUBSTRINGS = (
    # 输入/输出投影层
    "patchify_proj",
    "proj_out",
    # AdaLN 层（自适应层归一化）
    "adaln_single",
    "caption_projection",
    # 音视频相关
    "audio_patchify_proj",
    "audio_adaln_single",
    "audio_caption_projection",
    "audio_proj_out",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    # 第一层（输入层敏感）
    "transformer_blocks.0.",
    # 最后几层（输出层敏感）
    *[f"transformer_blocks.{i}." for i in range(43, 48)],
)


def _linear_to_fp8linear(layer: nn.Linear) -> FP8Linear:
    """
    将 nn.Linear 转换为 FP8Linear

    Args:
        layer: 原始 Linear 层（通常在 meta device 上）

    Returns:
        新的 FP8Linear，具有相同配置
    """
    return FP8Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
    )


def _apply_fp8_prepare_to_model(
    model: nn.Module, excluded_layer_substrings: tuple[str, ...]
) -> nn.Module:
    """
    将模型中的 Linear 层替换为 FP8Linear

    遍历模型树，将符合条件的 Linear 层替换为 FP8Linear。
    排除列表中的层保持原样。
    """
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue

        if _should_skip_layer(name, excluded_layer_substrings):
            continue

        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, module))

    for parent, attr_name, linear in replacements:
        setattr(parent, attr_name, _linear_to_fp8linear(linear))

    return model


def _create_transpose_kv_operation(
    excluded_layer_substrings: tuple[str, ...],
) -> Callable[[str, torch.Tensor], list[KeyValueOperationResult]]:
    """
    创建转置权重的 kv 操作

    FP8 Scaled MM 需要权重以转置形式存储以适配 cuBLAS 布局。
    """

    def transpose_if_matches(
        key: str, value: torch.Tensor
    ) -> list[KeyValueOperationResult]:
        # 只处理 .weight keys
        if not key.endswith(".weight"):
            return [KeyValueOperationResult(key, value)]

        # 只转置 2D FP8 张量（Linear 权重）
        if value.dim() != 2 or value.dtype != torch.float8_e4m3fn:
            return [KeyValueOperationResult(key, value)]

        # 检查是否在排除列表中
        layer_name = key.rsplit(".weight", 1)[0]
        if _should_skip_layer(layer_name, excluded_layer_substrings):
            return [KeyValueOperationResult(key, value)]

        # 转置以适配 cuBLAS 布局 (out, in) -> (in, out)
        transposed_weight = value.t()
        return [KeyValueOperationResult(key, transposed_weight)]

    return transpose_if_matches


# FP8 Scaled MM 的状态字典操作
FP8_TRANSPOSE_SD_OPS = SDOps("fp8_transpose_weights").with_kv_operation(
    _create_transpose_kv_operation(EXCLUDED_LAYER_SUBSTRINGS),
    key_prefix="transformer_blocks.",
    key_suffix=".weight",
)

# FP8 Scaled MM 的模块操作
FP8_PREPARE_MODULE_OPS = ModuleOps(
    name="fp8_prepare_for_loading",
    matcher=lambda model: hasattr(model, "transformer_blocks"),
    mutator=lambda model: _apply_fp8_prepare_to_model(model, EXCLUDED_LAYER_SUBSTRINGS),
)
