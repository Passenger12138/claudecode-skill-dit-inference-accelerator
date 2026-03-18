"""
Attention Processor 替换工具

提供替换 DiT 模型 Attention Processor 的工具函数。
"""

from enum import Enum
from typing import Callable, Optional

import torch
from torch import nn

from .flash_attention import (
    attention_with_flash2,
    attention_with_flash3,
    is_flash_attn3_available,
    is_flash_attn_available,
)
from .sage_attention import attention_with_sage, is_sage_attn_available


class AttentionProcessorType(Enum):
    """Attention Processor 类型"""

    PYTORCH_SDPA = "pytorch_sdpa"  # PyTorch 原生 scaled_dot_product_attention
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"
    SAGE_ATTENTION = "sage_attention"
    AUTO = "auto"  # 自动选择最佳


class FlashAttentionProcessor:
    """Flash Attention Processor"""

    def __init__(self, use_fa3: bool = False):
        self.use_fa3 = use_fa3 and is_flash_attn3_available()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 选择 Flash Attention 版本
        if self.use_fa3:
            hidden_states = attention_with_flash3(
                query,
                key,
                value,
                num_heads=attn.heads,
                dropout_p=0.0,
                causal=False,
            )
        else:
            hidden_states = attention_with_flash2(
                query,
                key,
                value,
                num_heads=attn.heads,
                dropout_p=0.0,
                causal=False,
            )

        # 输出投影
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SageAttentionProcessor:
    """Sage Attention Processor"""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Sage Attention 计算
        hidden_states = attention_with_sage(
            query,
            key,
            value,
            num_heads=attn.heads,
            causal=False,
        )

        # 输出投影
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def create_attention_processor(
    processor_type: AttentionProcessorType = AttentionProcessorType.AUTO,
):
    """
    创建 Attention Processor

    Args:
        processor_type: Processor 类型

    Returns:
        Attention Processor 实例
    """
    if processor_type == AttentionProcessorType.AUTO:
        # 自动选择：FA3 > FA2 > Sage > SDPA
        if is_flash_attn3_available():
            return FlashAttentionProcessor(use_fa3=True)
        elif is_flash_attn_available():
            return FlashAttentionProcessor(use_fa3=False)
        elif is_sage_attn_available():
            return SageAttentionProcessor()
        else:
            return None  # 使用默认 SDPA

    elif processor_type == AttentionProcessorType.FLASH_ATTENTION_2:
        if not is_flash_attn_available():
            raise ImportError(
                "Flash Attention 2 未安装。\n"
                "请运行: pip install flash-attn --no-build-isolation"
            )
        return FlashAttentionProcessor(use_fa3=False)

    elif processor_type == AttentionProcessorType.FLASH_ATTENTION_3:
        if not is_flash_attn3_available():
            raise ImportError(
                "Flash Attention 3 未安装或不支持当前显卡。\n"
                "FA3 需要 Hopper 架构显卡（H100 等）。"
            )
        return FlashAttentionProcessor(use_fa3=True)

    elif processor_type == AttentionProcessorType.SAGE_ATTENTION:
        if not is_sage_attn_available():
            raise ImportError(
                "Sage Attention 未安装。\n"
                "请运行: pip install sageattention"
            )
        return SageAttentionProcessor()

    elif processor_type == AttentionProcessorType.PYTORCH_SDPA:
        return None  # 使用默认 SDPA

    else:
        raise ValueError(f"未知的 Processor 类型: {processor_type}")


def replace_attention_processor(
    model: nn.Module,
    processor_type: AttentionProcessorType = AttentionProcessorType.AUTO,
) -> nn.Module:
    """
    替换模型中所有 Attention 层的 Processor

    Args:
        model: DiT 模型
        processor_type: Processor 类型

    Returns:
        修改后的模型
    """
    processor = create_attention_processor(processor_type)

    if processor is None:
        print(f"使用默认 PyTorch SDPA")
        return model

    # 遍历所有模块，替换 Attention 的 processor
    replaced_count = 0
    for name, module in model.named_modules():
        # 检查是否是 Attention 模块（通过属性判断）
        if hasattr(module, "set_processor") and callable(module.set_processor):
            module.set_processor(processor)
            replaced_count += 1
        elif hasattr(module, "processor"):
            module.processor = processor
            replaced_count += 1

    print(f"已替换 {replaced_count} 个 Attention Processor")
    return model


def get_attention_info() -> dict:
    """获取当前环境支持的 Attention 优化信息"""
    info = {
        "flash_attention_2": is_flash_attn_available(),
        "flash_attention_3": is_flash_attn3_available(),
        "sage_attention": is_sage_attn_available(),
        "recommended": None,
    }

    # 推荐顺序：FA3 > FA2 > Sage > SDPA
    if info["flash_attention_3"]:
        info["recommended"] = "flash_attention_3"
    elif info["flash_attention_2"]:
        info["recommended"] = "flash_attention_2"
    elif info["sage_attention"]:
        info["recommended"] = "sage_attention"
    else:
        info["recommended"] = "pytorch_sdpa"

    return info
