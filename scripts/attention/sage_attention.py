"""
Sage Attention 实现

Sage Attention 是一种有损但高效的注意力优化方法。
使用 INT8 量化和近似计算来加速注意力。
"""

from typing import Optional

import torch


def is_sage_attn_available() -> bool:
    """检查 Sage Attention 是否可用"""
    try:
        from sageattention import sageattn  # noqa: F401

        return True
    except ImportError:
        return False


def attention_with_sage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    causal: bool = False,
    smooth_k: bool = True,
    quantization_backend: str = "triton",
) -> torch.Tensor:
    """
    使用 Sage Attention 计算注意力

    Sage Attention 使用 INT8 量化来加速计算，有微小精度损失，
    但对于视频生成等任务通常可以接受。

    Args:
        query: Query 张量 [B, S, H*D] 或 [B, S, H, D]
        key: Key 张量 [B, S, H*D] 或 [B, S, H, D]
        value: Value 张量 [B, S, H*D] 或 [B, S, H, D]
        num_heads: 注意力头数
        causal: 是否使用因果注意力掩码
        smooth_k: 是否平滑 K 张量（推荐开启）
        quantization_backend: 量化后端 ("triton" 或 "cuda")

    Returns:
        注意力输出 [B, S, H*D]
    """
    from sageattention import sageattn

    batch_size = query.shape[0]
    seq_len = query.shape[1]

    # 如果是 3D 张量，重塑为 [B, S, H, D]
    if query.dim() == 3:
        dim_head = query.shape[-1] // num_heads
        query = query.view(batch_size, seq_len, num_heads, dim_head)
        key = key.view(batch_size, key.shape[1], num_heads, dim_head)
        value = value.view(batch_size, value.shape[1], num_heads, dim_head)

    # 转换为 [B, H, S, D] 格式（Sage Attention 需要）
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Sage Attention 计算
    output = sageattn(
        query,
        key,
        value,
        is_causal=causal,
        smooth_k=smooth_k,
    )

    # 转换回 [B, S, H, D]
    output = output.transpose(1, 2)

    # 重塑回 [B, S, H*D]
    return output.reshape(batch_size, seq_len, -1)


def attention_with_sage_2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    使用 Sage Attention 2 计算注意力

    Sage Attention 2 支持更多数据类型和更好的精度。
    """
    try:
        from sageattention import sageattn_varlen
    except ImportError:
        # 回退到 sageattn
        return attention_with_sage(query, key, value, num_heads, causal)

    batch_size = query.shape[0]
    seq_len = query.shape[1]

    # 如果是 3D 张量，重塑为 [B, S, H, D]
    if query.dim() == 3:
        dim_head = query.shape[-1] // num_heads
        query = query.view(batch_size, seq_len, num_heads, dim_head)
        key = key.view(batch_size, key.shape[1], num_heads, dim_head)
        value = value.view(batch_size, value.shape[1], num_heads, dim_head)

    # 转换为 [B, H, S, D] 格式
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Sage Attention 2 计算
    output = sageattn_varlen(
        query,
        key,
        value,
        is_causal=causal,
    )

    # 转换回 [B, S, H, D]
    output = output.transpose(1, 2)

    # 重塑回 [B, S, H*D]
    return output.reshape(batch_size, seq_len, -1)
