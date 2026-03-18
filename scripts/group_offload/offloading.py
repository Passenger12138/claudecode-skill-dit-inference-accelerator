"""
流式轮转实现

基于 diffusers 的 group offloading 机制，提供简化的 API。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch import nn


class OffloadType(str, Enum):
    """Offload 类型"""

    BLOCK_LEVEL = "block_level"  # 按 transformer block 分组
    LEAF_LEVEL = "leaf_level"  # 最细粒度，每个子模块单独 offload


@dataclass
class GroupOffloadingConfig:
    """
    Group Offloading 配置

    Attributes:
        onload_device: 计算设备（通常是 GPU）
        offload_device: 存储设备（通常是 CPU）
        offload_type: Offload 类型
        use_stream: 是否使用 CUDA stream 加速传输
        num_blocks_per_group: 每组的 block 数量（仅 block_level 有效）
    """

    onload_device: torch.device = torch.device("cuda")
    offload_device: torch.device = torch.device("cpu")
    offload_type: OffloadType = OffloadType.LEAF_LEVEL
    use_stream: bool = True
    num_blocks_per_group: int = 1


def apply_group_offloading(
    model: nn.Module,
    onload_device: torch.device = torch.device("cuda"),
    offload_device: torch.device = torch.device("cpu"),
    offload_type: str = "leaf_level",
    use_stream: bool = True,
    num_blocks_per_group: int = 1,
) -> nn.Module:
    """
    应用 Group Offloading 到模型

    使用 diffusers 提供的 apply_group_offloading 函数。

    Args:
        model: 要 offload 的模型
        onload_device: 计算设备
        offload_device: 存储设备
        offload_type: "block_level" 或 "leaf_level"
        use_stream: 是否使用 CUDA stream
        num_blocks_per_group: 每组的 block 数（仅 block_level 有效）

    Returns:
        应用了 offloading 的模型
    """
    try:
        from diffusers.hooks import apply_group_offloading as diffusers_apply_group_offloading
    except ImportError:
        raise ImportError(
            "diffusers 未安装或版本过低。\n"
            "请运行: pip install diffusers>=0.32.0"
        )

    # 确保模型在 CPU 上
    model.to(offload_device)

    # 应用 group offloading
    diffusers_apply_group_offloading(
        model,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type=offload_type,
        use_stream=use_stream,
        num_blocks_per_group=num_blocks_per_group if offload_type == "block_level" else None,
    )

    return model


def apply_group_offloading_with_config(
    model: nn.Module,
    config: GroupOffloadingConfig,
) -> nn.Module:
    """
    使用配置对象应用 Group Offloading

    Args:
        model: 要 offload 的模型
        config: Offloading 配置

    Returns:
        应用了 offloading 的模型
    """
    return apply_group_offloading(
        model,
        onload_device=config.onload_device,
        offload_device=config.offload_device,
        offload_type=config.offload_type.value,
        use_stream=config.use_stream,
        num_blocks_per_group=config.num_blocks_per_group,
    )


def estimate_memory_savings(
    model: nn.Module,
    offload_type: str = "leaf_level",
) -> dict:
    """
    估算 Group Offloading 的显存节省

    Args:
        model: 模型
        offload_type: Offload 类型

    Returns:
        显存估算信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # 估算不同 offload 类型的显存占用
    if offload_type == "leaf_level":
        # 最细粒度：只需要最大单层的显存
        max_layer_bytes = 0
        for module in model.modules():
            layer_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            max_layer_bytes = max(max_layer_bytes, layer_bytes)
        estimated_gpu_bytes = max_layer_bytes
    else:
        # block_level：需要最大单个 block 的显存
        max_block_bytes = 0
        for name, module in model.named_children():
            block_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            max_block_bytes = max(max_block_bytes, block_bytes)
        estimated_gpu_bytes = max_block_bytes

    return {
        "total_params": total_params,
        "total_memory_gb": total_bytes / (1024 ** 3),
        "estimated_gpu_memory_gb": estimated_gpu_bytes / (1024 ** 3),
        "memory_savings_percent": (1 - estimated_gpu_bytes / total_bytes) * 100,
    }


def get_recommended_offload_config(
    model: nn.Module,
    gpu_memory_gb: float,
) -> GroupOffloadingConfig:
    """
    根据模型大小和显存推荐 Offload 配置

    Args:
        model: 模型
        gpu_memory_gb: GPU 显存大小（GB）

    Returns:
        推荐的配置
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_gb = total_bytes / (1024 ** 3)

    # 预留 30% 给激活值
    available_gb = gpu_memory_gb * 0.7

    if total_gb <= available_gb:
        # 需要 offloading 使用 block_level 加速更快
        return GroupOffloadingConfig(
            offload_type=OffloadType.BLOCK_LEVEL,
            use_stream=True,
        )
    else:
        # 需要 offloading，使用 leaf_level 最大化节省
        return GroupOffloadingConfig(
            offload_type=OffloadType.LEAF_LEVEL,
            use_stream=True,
        )
