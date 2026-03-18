"""
流式轮转 (Group Offloading) 模块

将模型权重存储在 CPU 内存，推理时按需流式传输到 GPU。
适用于显存不足以容纳整个模型的场景。
"""

from .offloading import (
    apply_group_offloading,
    apply_group_offloading_with_config,
    GroupOffloadingConfig,
    OffloadType,
)

__all__ = [
    "apply_group_offloading",
    "apply_group_offloading_with_config",
    "GroupOffloadingConfig",
    "OffloadType",
]
