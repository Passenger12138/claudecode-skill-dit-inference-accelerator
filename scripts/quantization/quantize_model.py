#!/usr/bin/env python3
"""
DiT 模型量化脚本

将 safetensor 模型量化到 FP8 格式。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
    method: str = "fp8_cast",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    将张量量化到 FP8

    Args:
        tensor: 原始张量
        method: 量化方法 ("fp8_cast" 或 "fp8_scaled_mm")

    Returns:
        (quantized_tensor, scale): 量化后的张量和缩放因子（fp8_scaled_mm 需要）
    """
    if method == "fp8_cast":
        # 直接转换到 FP8
        return tensor.to(dtype=torch.float8_e4m3fn), None

    elif method == "fp8_scaled_mm":
        # 带缩放因子的量化
        tensor_fp32 = tensor.to(torch.float32)
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        max_abs = torch.amax(torch.abs(tensor_fp32))
        scale = fp8_max / max_abs

        quantized = torch.clamp(tensor_fp32 * scale, min=fp8_min, max=fp8_max).to(
            torch.float8_e4m3fn
        )

        return quantized, scale.reciprocal()

    else:
        raise ValueError(f"未知的量化方法: {method}")


def should_quantize_key(key: str, excluded_patterns: list[str]) -> bool:
    """
    判断是否应该量化该 key

    Args:
        key: 状态字典的 key
        excluded_patterns: 排除的模式列表

    Returns:
        是否应该量化
    """
    # 只量化 transformer_blocks 中的权重
    if not key.startswith("transformer_blocks."):
        return False

    # 检查是否在排除列表中
    for pattern in excluded_patterns:
        if pattern in key:
            return False

    # 只量化 weight 和 bias
    if not (key.endswith(".weight") or key.endswith(".bias")):
        return False

    return True


# 默认排除的模式
DEFAULT_EXCLUDED_PATTERNS = [
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "caption_projection",
    "transformer_blocks.0.",  # 第一层
]


def quantize_model(
    input_path: str,
    output_path: str,
    method: str = "fp8_cast",
    excluded_patterns: list[str] | None = None,
) -> dict:
    """
    量化模型

    Args:
        input_path: 输入模型路径（safetensor 文件或目录）
        output_path: 输出路径
        method: 量化方法
        excluded_patterns: 排除的模式列表

    Returns:
        量化统计信息
    """
    if excluded_patterns is None:
        excluded_patterns = DEFAULT_EXCLUDED_PATTERNS

    stats = {
        "total_tensors": 0,
        "quantized_tensors": 0,
        "skipped_tensors": 0,
        "original_size_bytes": 0,
        "quantized_size_bytes": 0,
    }

    # 收集所有文件
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".safetensors")
        ]
    else:
        files = [input_path]

    if not files:
        raise ValueError(f"未找到 safetensor 文件: {input_path}")

    # 确保输出目录存在
    output_dir = Path(output_path)
    if output_dir.suffix == ".safetensors":
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个文件
    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")

        quantized_state_dict = {}
        scales_dict = {}  # fp8_scaled_mm 需要保存缩放因子

        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            for key in tqdm(keys, desc="量化中"):
                tensor = f.get_tensor(key)
                stats["total_tensors"] += 1
                stats["original_size_bytes"] += tensor.nbytes

                if should_quantize_key(key, excluded_patterns):
                    quantized, scale = quantize_tensor_to_fp8(tensor, method)
                    quantized_state_dict[key] = quantized
                    stats["quantized_tensors"] += 1
                    stats["quantized_size_bytes"] += quantized.nbytes

                    if scale is not None:
                        scales_dict[key.replace(".weight", ".weight_scale")] = scale
                else:
                    quantized_state_dict[key] = tensor
                    stats["skipped_tensors"] += 1
                    stats["quantized_size_bytes"] += tensor.nbytes

        # 合并缩放因子
        quantized_state_dict.update(scales_dict)

        # 保存量化后的模型
        if os.path.isdir(input_path):
            output_file = output_dir / os.path.basename(file_path)
        else:
            output_file = (
                Path(output_path)
                if output_path.endswith(".safetensors")
                else output_dir / os.path.basename(file_path)
            )

        print(f"保存到: {output_file}")
        save_file(quantized_state_dict, str(output_file))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="DiT 模型量化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # FP8 Cast 量化（通用）
  python quantize_model.py --model-path model.safetensors --output-path model_fp8.safetensors --method fp8_cast

  # FP8 Scaled MM 量化（Hopper 显卡）
  python quantize_model.py --model-path model.safetensors --output-path model_fp8.safetensors --method fp8_scaled_mm
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="输入模型路径（safetensor 文件或目录）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="输出路径",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fp8_cast",
        choices=["fp8_cast", "fp8_scaled_mm"],
        help="量化方法（默认: fp8_cast）",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        help="额外排除的模式",
    )

    args = parser.parse_args()

    # 合并排除模式
    excluded_patterns = DEFAULT_EXCLUDED_PATTERNS.copy()
    if args.exclude:
        excluded_patterns.extend(args.exclude)

    print("=" * 60)
    print(" DiT 模型量化".center(60))
    print("=" * 60)
    print(f"\n输入: {args.model_path}")
    print(f"输出: {args.output_path}")
    print(f"方法: {args.method}")
    print(f"排除模式: {excluded_patterns}")

    try:
        stats = quantize_model(
            args.model_path,
            args.output_path,
            args.method,
            excluded_patterns,
        )

        print("\n" + "=" * 60)
        print(" 量化完成".center(60))
        print("=" * 60)
        print(f"\n总张量数: {stats['total_tensors']}")
        print(f"已量化: {stats['quantized_tensors']}")
        print(f"已跳过: {stats['skipped_tensors']}")
        print(
            f"原始大小: {stats['original_size_bytes'] / (1024**3):.2f} GB"
        )
        print(
            f"量化后大小: {stats['quantized_size_bytes'] / (1024**3):.2f} GB"
        )
        print(
            f"压缩率: {stats['quantized_size_bytes'] / stats['original_size_bytes'] * 100:.1f}%"
        )

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
