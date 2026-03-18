"""
DiT 模型 Attention 优化模块

提供多种 Attention 优化实现：
- Flash Attention 2: 通用优化，~1.3x 加速
- Flash Attention 3: Hopper 显卡优化，~1.5x 加速
- Sage Attention: 有损优化，~1.6x 加速
"""

from .flash_attention import (
    attention_with_flash2,
    attention_with_flash3,
    is_flash_attn_available,
    is_flash_attn3_available,
)
from .sage_attention import attention_with_sage, is_sage_attn_available
from .processor import (
    create_attention_processor,
    AttentionProcessorType,
    replace_attention_processor,
)

__all__ = [
    # Flash Attention
    "attention_with_flash2",
    "attention_with_flash3",
    "is_flash_attn_available",
    "is_flash_attn3_available",
    # Sage Attention
    "attention_with_sage",
    "is_sage_attn_available",
    # Processor
    "create_attention_processor",
    "AttentionProcessorType",
    "replace_attention_processor",
]
