#!/usr/bin/env python3
"""
DiT 模型诊断分析工具

分析 safetensor 模型文件，检测 GPU 显存，并给出加速策略建议。
运行后会询问用户确认是否采纳量化建议。
"""

import argparse
import os
import sys
from pathlib import Path


def analyze_safetensor_model(model_path: str) -> dict:
    """
    分析 safetensor 模型文件，返回模型大小信息

    Args:
        model_path: safetensor 文件路径或包含多个 safetensor 的目录

    Returns:
        dict: 包含总参数量、显存占用估算等信息
    """
    from safetensors import safe_open

    total_params = 0
    total_bytes = 0
    dtype_counts = {}

    # 支持单文件或目录
    if os.path.isdir(model_path):
        files = [
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        ]
        if not files:
            raise ValueError(f"目录 {model_path} 中没有找到 .safetensors 文件")
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"文件不存在: {model_path}")
        files = [model_path]

    print(f"\n正在分析 {len(files)} 个文件...")
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()

                # 计算参数量
                params = 1
                for dim in shape:
                    params *= dim

                # 计算字节数
                dtype_size = {
                    "F64": 8,
                    "F32": 4,
                    "F16": 2,
                    "BF16": 2,
                    "F8_E4M3": 1,
                    "F8_E5M2": 1,
                    "I64": 8,
                    "I32": 4,
                    "I16": 2,
                    "I8": 1,
                    "U8": 1,
                    "BOOL": 1,
                }.get(dtype, 4)

                bytes_size = params * dtype_size

                total_params += params
                total_bytes += bytes_size
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params

    # 计算不同精度下的显存占用
    bf16_memory_gb = (total_params * 2) / (1024**3)  # BF16: 2 bytes per param
    fp8_memory_gb = (total_params * 1) / (1024**3)  # FP8: 1 byte per param

    return {
        "total_params": total_params,
        "total_params_b": total_params / 1e9,  # 单位：B（十亿）
        "current_size_gb": total_bytes / (1024**3),
        "bf16_memory_gb": bf16_memory_gb,
        "fp8_memory_gb": fp8_memory_gb,
        "dtype_distribution": dtype_counts,
    }


def get_gpu_info() -> dict:
    """获取 GPU 信息"""
    import torch

    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}",
        "is_hopper": props.major >= 9,  # H100 等 Hopper 架构
        "is_ada": props.major == 8 and props.minor >= 9,  # RTX 40 系列
        "is_ampere": props.major == 8 and props.minor < 9,  # RTX 30 系列
    }


def recommend_acceleration_strategy(model_info: dict, gpu_info: dict) -> dict:
    """
    根据模型大小和显卡显存推荐加速策略

    Returns:
        dict: 推荐的策略配置
    """
    if not gpu_info["available"]:
        return {
            "error": "未检测到 CUDA GPU",
            "quantization": None,
            "offloading": False,
            "reason": ["无法使用 GPU 加速"],
        }

    model_bf16_gb = model_info["bf16_memory_gb"]
    model_fp8_gb = model_info["fp8_memory_gb"]
    gpu_memory_gb = gpu_info["total_memory_gb"]

    # 预留显存给激活值和其他组件（约 30%）
    available_for_model = gpu_memory_gb * 0.7

    recommendations = {
        "quantization": None,
        "quantization_method": None,
        "offloading": False,
        "reason": [],
    }

    # 判断是否需要量化
    if model_bf16_gb <= available_for_model:
        recommendations["reason"].append(
            f"✓ BF16 模型 ({model_bf16_gb:.1f}GB) 可直接放入显存 ({available_for_model:.1f}GB 可用)"
        )
        recommendations["reason"].append("建议: 不需要量化或流式轮转")
    elif model_fp8_gb <= available_for_model:
        recommendations["quantization"] = "fp8"
        recommendations["reason"].append(
            f"⚠ BF16 ({model_bf16_gb:.1f}GB) > 可用显存 ({available_for_model:.1f}GB)"
        )
        recommendations["reason"].append(
            f"✓ FP8 量化后 ({model_fp8_gb:.1f}GB) 可放入显存"
        )

        # Hopper 显卡使用 fp8_scaled_mm，其他使用 fp8_cast
        if gpu_info.get("is_hopper"):
            recommendations["quantization_method"] = "fp8_scaled_mm"
            recommendations["reason"].append(
                "建议: 使用 fp8_scaled_mm 量化（Hopper 显卡，需要 TensorRT-LLM）"
            )
        else:
            recommendations["quantization_method"] = "fp8_cast"
            recommendations["reason"].append("建议: 使用 fp8_cast 量化（通用方法）")
    else:
        recommendations["quantization"] = "fp8"
        recommendations["offloading"] = True
        recommendations["reason"].append(
            f"✗ BF16 ({model_bf16_gb:.1f}GB) > 可用显存 ({available_for_model:.1f}GB)"
        )
        recommendations["reason"].append(
            f"✗ 即使 FP8 ({model_fp8_gb:.1f}GB) 仍超出可用显存"
        )

        if gpu_info.get("is_hopper"):
            recommendations["quantization_method"] = "fp8_scaled_mm"
        else:
            recommendations["quantization_method"] = "fp8_cast"

        recommendations["reason"].append(
            "建议: FP8 量化 + 流式轮转（Group Offloading）"
        )

    return recommendations


def print_report(model_info: dict, gpu_info: dict, strategy: dict):
    """打印诊断报告"""
    print("\n" + "=" * 70)
    print(" DiT 模型诊断报告".center(70))
    print("=" * 70)

    # 模型信息
    print("\n【模型信息】")
    print(f"  参数量: {model_info['total_params_b']:.2f}B")
    print(f"  当前大小: {model_info['current_size_gb']:.2f} GB")
    print(f"  BF16 显存占用估算: {model_info['bf16_memory_gb']:.2f} GB")
    print(f"  FP8 显存占用估算: {model_info['fp8_memory_gb']:.2f} GB")

    if model_info["dtype_distribution"]:
        print(f"\n  数据类型分布:")
        for dtype, count in model_info["dtype_distribution"].items():
            print(f"    {dtype}: {count / 1e9:.2f}B 参数")

    # GPU 信息
    print("\n【GPU 信息】")
    if gpu_info["available"]:
        print(f"  显卡: {gpu_info['name']}")
        print(f"  显存: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"  计算能力: {gpu_info['compute_capability']}")
        arch = (
            "Hopper"
            if gpu_info["is_hopper"]
            else "Ada Lovelace" if gpu_info["is_ada"] else "Ampere"
        )
        print(f"  架构: {arch}")
    else:
        print("  ✗ 未检测到 CUDA GPU")

    # 推荐策略
    print("\n【加速策略推荐】")
    for reason in strategy["reason"]:
        print(f"  {reason}")

    print("\n【推荐配置】")
    print(f"  量化: {strategy['quantization'] or '不需要'}")
    if strategy.get("quantization_method"):
        print(f"  量化方法: {strategy['quantization_method']}")
    print(f"  流式轮转: {'需要' if strategy['offloading'] else '不需要'}")

    print("\n" + "=" * 70)


def ask_user_confirmation(strategy: dict, gpu_info: dict) -> dict:
    """
    询问用户是否采纳建议

    无论系统建议如何，都会询问用户确认。
    """
    print("\n" + "=" * 70)
    print(" 用户确认".center(70))
    print("=" * 70)

    final_strategy = {
        "quantization": None,
        "quantization_method": None,
        "offloading": False,
    }

    # ========== 询问是否量化 ==========
    if strategy["quantization"]:
        print(f"\n系统建议: 使用 {strategy['quantization_method']} 量化")
        print("  (FP8 量化可减少约 50% 显存占用)")
    else:
        print("\n系统建议: 不需要量化")
        print("  (模型可直接放入显存)")

    print("\n是否进行 FP8 量化?")
    print("  1. 是，使用 fp8_cast（通用方法，推荐）")
    print("  2. 是，使用 fp8_scaled_mm（Hopper 显卡专用）")
    print("  3. 否，不进行量化")

    while True:
        choice = input("\n请选择 [1/2/3]: ").strip()
        if choice == "1":
            final_strategy["quantization"] = "fp8"
            final_strategy["quantization_method"] = "fp8_cast"
            print("✓ 已选择 fp8_cast 量化")
            break
        elif choice == "2":
            if not gpu_info.get("is_hopper"):
                print("⚠ 警告: fp8_scaled_mm 仅支持 Hopper 显卡（H100 等）")
                confirm = input("  确定要继续吗? [y/N]: ").strip().lower()
                if confirm not in ["y", "yes"]:
                    continue
            final_strategy["quantization"] = "fp8"
            final_strategy["quantization_method"] = "fp8_scaled_mm"
            print("✓ 已选择 fp8_scaled_mm 量化")
            break
        elif choice == "3":
            print("✓ 不进行量化")
            break
        else:
            print("无效选择，请输入 1、2 或 3")

    # ========== 询问是否流式轮转 ==========
    if strategy["offloading"]:
        print(f"\n系统建议: 启用流式轮转")
        print("  (即使 FP8 量化后显存仍不足，需要流式轮转)")
    else:
        print("\n系统建议: 不需要流式轮转")

    print("\n是否启用流式轮转 (Group Offloading)?")
    print("  1. 是，使用 leaf_level（最细粒度，显存占用最小）")
    print("  2. 是，使用 block_level（按 block 分组，传输效率更高）")
    print("  3. 否，不启用流式轮转")

    while True:
        choice = input("\n请选择 [1/2/3]: ").strip()
        if choice == "1":
            final_strategy["offloading"] = True
            final_strategy["offload_type"] = "leaf_level"
            print("✓ 已选择 leaf_level 流式轮转")
            break
        elif choice == "2":
            final_strategy["offloading"] = True
            final_strategy["offload_type"] = "block_level"
            print("✓ 已选择 block_level 流式轮转")
            break
        elif choice == "3":
            print("✓ 不启用流式轮转")
            break
        else:
            print("无效选择，请输入 1、2 或 3")

    return final_strategy


def generate_code_snippet(strategy: dict, model_path: str) -> str:
    """生成配置代码片段"""
    code = """
# ========== DiT 模型加速配置 ==========

import torch
"""

    if strategy.get("quantization"):
        code += f"""
# 1. 量化配置
from quantization import QuantizationPolicy

quantization = QuantizationPolicy.{strategy['quantization_method']}()
"""
    else:
        code += """
# 1. 不使用量化
quantization = None
"""

    code += f"""
# 2. 加载模型
pipeline = YourPipeline(
    checkpoint_path="{model_path}",
    quantization=quantization,
)
"""

    if strategy.get("offloading"):
        offload_type = strategy.get("offload_type", "leaf_level")
        code += f"""
# 3. 流式轮转
from diffusers.hooks import apply_group_offloading

pipeline.transformer.to("cpu")
apply_group_offloading(
    pipeline.transformer,
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="{offload_type}",
    use_stream=True,
)
"""

    code += """
# 4. 推理
output = pipeline(prompt, num_inference_steps=8)
"""

    return code


def main():
    parser = argparse.ArgumentParser(
        description="DiT 模型诊断分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="safetensor 模型文件路径或目录",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="非交互模式，直接采纳系统建议",
    )

    args = parser.parse_args()

    try:
        # 分析模型
        print("正在分析模型...")
        model_info = analyze_safetensor_model(args.model_path)

        # 获取 GPU 信息
        print("正在检测 GPU...")
        gpu_info = get_gpu_info()

        # 推荐策略
        strategy = recommend_acceleration_strategy(model_info, gpu_info)

        # 打印报告
        print_report(model_info, gpu_info, strategy)

        # 用户确认
        if args.non_interactive:
            final_strategy = strategy
            print("\n[非交互模式] 采纳系统建议")
        else:
            final_strategy = ask_user_confirmation(strategy, gpu_info)

        # 生成代码片段
        print("\n" + "=" * 70)
        print(" 配置代码".center(70))
        print("=" * 70)
        code = generate_code_snippet(final_strategy, args.model_path)
        print(code)

        # 保存配置
        print("\n是否保存配置到文件? [y/N]: ", end="")
        if not args.non_interactive:
            save = input().strip().lower()
            if save in ["y", "yes"]:
                config_path = Path(args.model_path).parent / "acceleration_config.py"
                with open(config_path, "w") as f:
                    f.write(code)
                print(f"✓ 配置已保存到: {config_path}")

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
