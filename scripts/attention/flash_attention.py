"""
Flash Attention 实现

提供 Flash Attention 2 和 Flash Attention 3 的封装。
"""

from typing import Optional

import torch


def is_flash_attn_available() -> bool:
    """检查 Flash Attention 2 是否可用"""
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        return True
    except ImportError:
        return False


def is_flash_attn3_available() -> bool:
    """检查 Flash Attention 3 是否可用（需要 Hopper 显卡）"""
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_v3  # noqa: F401

        # 检查是否是 Hopper 架构
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.major >= 9
        return False
    except ImportError:
        return False


def attention_with_flash2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    使用 Flash Attention 2 计算注意力

    Args:
        query: Query 张量 [B, S, H*D] 或 [B, S, H, D]
        key: Key 张量 [B, S, H*D] 或 [B, S, H, D]
        value: Value 张量 [B, S, H*D] 或 [B, S, H, D]
        num_heads: 注意力头数
        dropout_p: Dropout 概率（推理时应为 0）
        causal: 是否使用因果注意力掩码
        softmax_scale: Softmax 缩放因子（默认 1/sqrt(dim_head)）

    Returns:
        注意力输出 [B, S, H*D]
    """
    from flash_attn import flash_attn_func

    batch_size = query.shape[0]
    seq_len = query.shape[1]

    # 如果是 3D 张量，重塑为 [B, S, H, D]
    if query.dim() == 3:
        dim_head = query.shape[-1] // num_heads
        query = query.view(batch_size, seq_len, num_heads, dim_head)
        key = key.view(batch_size, key.shape[1], num_heads, dim_head)
        value = value.view(batch_size, value.shape[1], num_heads, dim_head)

    # Flash Attention 2 计算
    output = flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    # 重塑回 [B, S, H*D]
    return output.reshape(batch_size, seq_len, -1)


def attention_with_flash3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    使用 Flash Attention 3 计算注意力（Hopper 显卡专用）

    Flash Attention 3 针对 Hopper 架构进行了深度优化，
    利用 TMA（Tensor Memory Accelerator）和异步流水线。

    Args:
        query: Query 张量 [B, S, H*D] 或 [B, S, H, D]
        key: Key 张量 [B, S, H*D] 或 [B, S, H, D]
        value: Value 张量 [B, S, H*D] 或 [B, S, H, D]
        num_heads: 注意力头数
        dropout_p: Dropout 概率（推理时应为 0）
        causal: 是否使用因果注意力掩码
        softmax_scale: Softmax 缩放因子（默认 1/sqrt(dim_head)）

    Returns:
        注意力输出 [B, S, H*D]
    """
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    batch_size = query.shape[0]
    seq_len = query.shape[1]

    # 如果是 3D 张量，重塑为 [B, S, H, D]
    if query.dim() == 3:
        dim_head = query.shape[-1] // num_heads
        query = query.view(batch_size, seq_len, num_heads, dim_head)
        key = key.view(batch_size, key.shape[1], num_heads, dim_head)
        value = value.view(batch_size, value.shape[1], num_heads, dim_head)

    # Flash Attention 3 计算
    output = flash_attn_func_v3(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    # 处理返回值（FA3 可能返回元组）
    if isinstance(output, tuple):
        output = output[0]

    # 重塑回 [B, S, H*D]
    return output.reshape(batch_size, seq_len, -1)


def attention_with_flash_auto(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    自动选择最佳 Flash Attention 版本

    优先使用 FA3（如果 Hopper 显卡可用），否则使用 FA2。
    """
    if is_flash_attn3_available():
        return attention_with_flash3(
            query, key, value, num_heads, dropout_p, causal, softmax_scale
        )
    elif is_flash_attn_available():
        return attention_with_flash2(
            query, key, value, num_heads, dropout_p, causal, softmax_scale
        )
    else:
        raise ImportError(
            "Flash Attention 未安装。\n"
            "请运行: pip install flash-attn --no-build-isolation"
        )
