---
name: dit-inference-accelerator
description: DiT 模型推理加速。当用户提到：DiT/Diffusion Transformer、视频生成（LTX-Video/Mochi/CogVideo/HunyuanVideo/Wan）、图像生成（Flux/SD3/SDXL）、显存不足/OOM/VRAM、4090/3090/24GB 部署、FP8 量化、Flash/Sage Attention、model offload、group offloading、蒸馏 LoRA、推理加速时触发。
---

# DiT 模型 4090 显卡部署与加速指南

将几十B参数的 DiT 模型部署到 24GB 消费级显卡（如 RTX 4090）需要解决两个核心问题：

## 第一步：让模型能放进显存

| 方案 | 原理 | 显存占用 | 精度损失 |
|------|------|---------|---------|
| FP8 量化 | 降低权重精度 | ~50% | 极小 |
| 流式轮转 | CPU/GPU 动态传输 | ~10% | 无 |

## 第二步：加速推理

| 技术 | 加速效果 | 精度影响 |
|------|---------|---------|
| 步数蒸馏 LoRA | ~5x | 极小 |
| Flash Attention 2 | ~1.3x | 无损 |
| Sage Attention | ~1.6x | 有损 |
| 计算精度设置 | ~1.1x | 有损 |

---

# 交互式配置流程

**使用此 skill 时，Claude 会一步步询问用户完成配置：**

```
步骤 1：分析模型组件
    ↓
步骤 2：询问是否 FP8 量化
    ↓
步骤 3：询问流式轮转策略
    ↓
步骤 4：询问是否有蒸馏 LoRA
    ↓
步骤 5：询问 Attention 加速方式
    ↓
步骤 6：询问是否启用 Torch 计算加速
    ↓
步骤 7：确认是否有其他需求
    ↓
步骤 8：生成完整配置代码
```

---

## 步骤 1：分析 Pipeline 组件

首先分析用户 pipeline 的**所有组件**。注意不同模型架构差异很大：

| 模型 | 架构特点 |
|------|---------|
| **Wan2.1/2.2** | 多阶段 DiT（多个 transformer） |
| **Flux** | 多个 Text Encoder（CLIP + T5） |
| **SD3** | 多个 Text Encoder（CLIP × 2 + T5） |
| **CogVideoX** | Image Encoder + Text Encoder |
| **HunyuanVideo** | 多个 Text Encoder + 大型 DiT |

**Claude 会执行以下代码分析组件**：

```python
def analyze_pipeline_components(pipe):
    """分析 pipeline 所有组件的参数量和显存占用"""
    components = {
        'transformers': [],      # DiT 模型（可能多个）
        'text_encoders': [],     # Text Encoder（可能多个）
        'other_models': [],      # 其他模型（VAE、Image Encoder 等）
    }

    # 扫描所有 Transformer/DiT
    for attr in ['transformer', 'transformer_1', 'transformer_2',
                 'transformer_3', 'dit', 'dit_1', 'dit_2', 'unet']:
        if hasattr(pipe, attr) and getattr(pipe, attr) is not None:
            model = getattr(pipe, attr)
            params = sum(p.numel() for p in model.parameters())
            components['transformers'].append({
                'name': attr,
                'params_b': params / 1e9,
                'bf16_gb': params * 2 / 1024**3,
            })

    # 扫描所有 Text Encoder
    for attr in ['text_encoder', 'text_encoder_2', 'text_encoder_3']:
        if hasattr(pipe, attr) and getattr(pipe, attr) is not None:
            model = getattr(pipe, attr)
            params = sum(p.numel() for p in model.parameters())
            components['text_encoders'].append({
                'name': attr,
                'params_b': params / 1e9,
                'bf16_gb': params * 2 / 1024**3,
            })

    # 扫描其他模型
    for attr in ['vae', 'image_encoder', 'controlnet', 'adapter']:
        if hasattr(pipe, attr) and getattr(pipe, attr) is not None:
            model = getattr(pipe, attr)
            if hasattr(model, 'parameters'):
                params = sum(p.numel() for p in model.parameters())
                if params > 0:
                    components['other_models'].append({
                        'name': attr,
                        'params_b': params / 1e9,
                        'bf16_gb': params * 2 / 1024**3,
                    })

    return components
```

**输出示例**：
```
Pipeline 组件分析：

【DiT/Transformer】
┌─────────────────┬──────────┬──────────┐
│ 组件            │ 参数量   │ BF16显存 │
├─────────────────┼──────────┼──────────┤
│ transformer     │ 12.0B    │ 22.4GB   │
└─────────────────┴──────────┴──────────┘

【Text Encoder】
┌─────────────────┬──────────┬──────────┐
│ 组件            │ 参数量   │ BF16显存 │
├─────────────────┼──────────┼──────────┤
│ text_encoder    │ 0.12B    │ 0.23GB   │ (CLIP)
│ text_encoder_2  │ 4.7B     │ 8.8GB    │ (T5-XXL)
└─────────────────┴──────────┴──────────┘

【其他模型】
┌─────────────────┬──────────┬──────────┐
│ 组件            │ 参数量   │ BF16显存 │
├─────────────────┼──────────┼──────────┤
│ vae             │ 0.08B    │ 0.15GB   │
└─────────────────┴──────────┴──────────┘

总计：16.9B 参数，31.6GB BF16 显存
GPU 显存：24GB（可用约 16.8GB，预留 30% 给激活值）
```

---

## 步骤 2：询问是否需要 FP8 量化

> **问题：是否需要进行 FP8 量化？**
>
> 当前 BF16 总显存占用：31.6GB
> 可用显存：16.8GB（24GB × 70%）
>
> - **选项 1**：是，量化所有 Transformer/DiT（推荐）
> - **选项 2**：是，量化所有组件
> - **选项 3**：否，不进行量化

**量化建议**：

| 组件 | 是否建议量化 | 原因 |
|------|-------------|------|
| Transformer/DiT | ✅ 推荐 | 参数量最大，DiT 对量化不敏感 |
| Text Encoder | ❌ 不推荐 | 影响文本理解质量 |
| VAE | ❌ 不推荐 | 参数量太小，可能影响图像质量 |

---

## 步骤 3：询问流式轮转策略

> **问题：请选择流式轮转策略**
>
> 检测到以下组件：
> - Transformer: transformer (22.4GB)
> - Text Encoder: text_encoder (0.23GB), text_encoder_2 (8.8GB)
> - 其他: vae (0.15GB)
>
> **必须流式轮转**（参数量大）：
> - ✅ transformer
> - ✅ text_encoder_2 (T5-XXL)
>
> **建议流式轮转**：
> - ✅ text_encoder (CLIP)
>
> **其他模型（VAE 等）处理方式**：
> - **选项 A**：常驻 CUDA（速度快，显存够时推荐）
> - **选项 B**：按需加载卸载（更省显存，显存紧张时选择）

**流式轮转策略说明**：

| 组件类型 | 策略 | 原因 |
|---------|------|------|
| Transformer/DiT | 必须流式轮转 | 参数量最大 |
| Text Encoder | 必须流式轮转 | 参数量较大，尤其是 T5 |
| VAE 等小模型 | 可选 | 较小，显存够可常驻 CUDA |

---

## 步骤 4：询问是否有蒸馏 LoRA

> **问题：是否有蒸馏 LoRA 可用？**
>
> 蒸馏 LoRA 可以大幅减少推理步数，实现约 **5 倍**加速：
> - 原始模型：40-50 步去噪
> - 使用蒸馏 LoRA：4-8 步去噪
>
> - **选项 1**：是，我有蒸馏 LoRA
> - **选项 2**：否，没有蒸馏 LoRA

**如果选择"是"，Claude 会继续询问**：

> **请提供蒸馏 LoRA 相关信息：**
>
> 1. **LoRA 文件路径**：
>    - 本地路径（如 `/path/to/distilled_lora.safetensors`）
>    - 或 HuggingFace 模型 ID（如 `someone/flux-turbo-lora`）
>
> 2. **推荐推理步数**：
>    - 蒸馏 LoRA 通常有推荐的步数（如 4 步、8 步）
>    - 如不确定，默认使用 8 步
>
> 3. **LoRA 强度**（可选）：
>    - 默认 1.0，部分 LoRA 需要调整
>    - 如有特殊要求请说明

**常见蒸馏 LoRA**：

| 模型 | 蒸馏 LoRA | 推荐步数 |
|------|----------|---------|
| Flux | FLUX.1-Turbo LoRA | 4 步 |
| SD3 | SD3-Turbo | 4 步 |
| SDXL | LCM-LoRA / Turbo | 4-8 步 |
| LTX-Video | - | 需要社区版 |

**加载示例**：
```python
# 从 HuggingFace 加载
pipe.load_lora_weights("alimama-creative/FLUX.1-Turbo-Alpha")

# 从本地文件加载
pipe.load_lora_weights("/path/to/distilled_lora.safetensors")

# 设置 LoRA 强度（可选）
pipe.set_adapters(["default"], adapter_weights=[0.8])

# 使用蒸馏后的步数推理
output = pipe(prompt, num_inference_steps=4)
```

---

## 步骤 5：询问 Attention 加速方式

> **问题：请选择 Attention 加速方式**
>
> | 方案 | 加速效果 | 精度影响 | 安装难度 |
> |------|---------|---------|---------|
> | Flash Attention 2 | ~1.3x | **无损** | 需编译安装 |
> | Sage Attention | ~1.6x | **有损** | pip 直接安装 |
> | 不使用优化 | 1x | 无 | 无需安装 |
>
> - **选项 1**：Flash Attention 2（无损加速，推荐追求质量）
> - **选项 2**：Sage Attention（有损但更快，推荐追求速度）
> - **选项 3**：不使用 Attention 优化

**详细对比**：

| 特性 | Flash Attention 2 | Sage Attention |
|------|------------------|----------------|
| **加速效果** | ~1.3x | ~1.6x |
| **精度影响** | ✅ 无损 | ⚠️ 有损（INT8 量化 QK） |
| **安装方式** | `pip install flash-attn --no-build-isolation` | `pip install sageattention` |
| **安装难度** | 需要 CUDA 编译环境 | 简单，pip 直接装 |
| **适用场景** | 追求质量，生产环境 | 追求速度，对精度容忍 |
| **显卡支持** | Ampere/Ada/Hopper | Ampere/Ada |

**选择建议**：
- 追求**质量** → Flash Attention 2（无损）
- 追求**速度** → Sage Attention（更快但有损）
- 安装有困难 → Sage Attention（更容易安装）

---

## 步骤 6：询问是否启用 Torch 计算加速

> **问题：是否启用 PyTorch 低精度计算加速？**
>
> | 选项 | 加速效果 | 精度影响 | 要求 |
> |------|---------|---------|------|
> | 启用 | ~10-15% | ⚠️ 有损 | PyTorch >= 2.7 |
> | 不启用 | 无 | ✅ 无损 | 无 |
>
> - **选项 1**：是，启用低精度计算加速
> - **选项 2**：否，保持默认精度（推荐）

**详细说明**：

| 特性 | 启用 | 不启用 |
|------|------|--------|
| **加速效果** | 约 10-15% | 无 |
| **精度影响** | 有损（FP16/BF16 累加替代 FP32） | 无损 |
| **适用场景** | 追求极致速度，能容忍精度损失 | 追求质量，生产环境 |
| **风险** | 可能导致数值不稳定、画面瑕疵 | 无风险 |
| **要求** | PyTorch >= 2.7 | 无 |

**选择建议**：
- 追求**质量** → 不启用（保守选择，推荐）
- 追求**速度** → 启用（需验证效果后使用）

---

## 步骤 7：确认是否有其他需求

> **问题：是否有其他需求？**
>
> - **选项 1**：添加 ControlNet 支持
> - **选项 2**：添加 IP-Adapter 支持
> - **选项 3**：没有其他需求，生成配置代码

---

## 步骤 8：生成完整配置代码

根据用户所有选择，Claude 生成完整配置代码：

```python
import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading

# ========== 加载模型 ==========
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

# ========== 1. FP8 量化（仅 Transformer） ==========
pipe.transformer = pipe.transformer.to(torch.float8_e4m3fn)

# ========== 2. 流式轮转 ==========
def apply_offload(model):
    model.to("cpu")
    apply_group_offloading(
        model,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
    )

# Transformer（必须）
apply_offload(pipe.transformer)

# Text Encoder（必须）
apply_offload(pipe.text_encoder)
apply_offload(pipe.text_encoder_2)

# VAE（用户选择：常驻 CUDA）
pipe.vae.to("cuda")

# ========== 3. 蒸馏 LoRA（用户提供） ==========
pipe.load_lora_weights("alimama-creative/FLUX.1-Turbo-Alpha")

# ========== 4. Attention 加速（用户选择：Flash Attention 2） ==========
# 需要修改模型的 attention 实现
# 安装：pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func

def optimized_attention(q, k, v, heads):
    b, seq_len, _ = q.shape
    dim_head = q.shape[-1] // heads
    q = q.view(b, seq_len, heads, dim_head)
    k = k.view(b, seq_len, heads, dim_head)
    v = v.view(b, seq_len, heads, dim_head)
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    return out.reshape(b, seq_len, heads * dim_head)

# ========== 5. Torch 计算加速（用户选择：不启用） ==========
# 保持默认精度，不启用低精度累加

# ========== 推理 ==========
output = pipe(
    prompt="A cat walking in the garden",
    num_inference_steps=4,  # 使用蒸馏 LoRA，4 步即可
)
```

---

# 技术详解

---

## FP8 量化

将模型权重从 BF16（16位）量化到 FP8（8位），显存占用减半。

```python
import torch

# 量化所有 Transformer/DiT
pipe.transformer = pipe.transformer.to(torch.float8_e4m3fn)

# 多阶段 DiT（如 Wan2.1/2.2）
# pipe.transformer_1 = pipe.transformer_1.to(torch.float8_e4m3fn)
# pipe.transformer_2 = pipe.transformer_2.to(torch.float8_e4m3fn)
```

---

## 流式轮转

将模型权重存储在 CPU，推理时按需传输到 GPU。

```python
from diffusers.hooks import apply_group_offloading
import torch

def apply_offload(model, offload_type="leaf_level"):
    model.to("cpu")
    apply_group_offloading(
        model,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type=offload_type,
        use_stream=True,
    )

# 必须流式轮转
apply_offload(pipe.transformer)
apply_offload(pipe.text_encoder)

# 可选（显存紧张时）
# apply_offload(pipe.vae)
```

| offload_type | 显存占用 | 传输效率 |
|--------------|---------|---------|
| `leaf_level` | 最小 | 较低（推荐） |
| `block_level` | 较小 | 较高 |

---

## 蒸馏 LoRA

使用蒸馏 LoRA 减少推理步数，实现约 5 倍加速。

```python
# 加载蒸馏 LoRA
pipe.load_lora_weights("path/to/distilled_lora.safetensors")

# 设置 LoRA 强度（可选）
pipe.set_adapters(["default"], adapter_weights=[1.0])

# 使用蒸馏步数推理
output = pipe(prompt, num_inference_steps=4)  # 从 50 步降到 4 步
```

---

## Attention 加速

**Flash Attention 2**（无损）：

```python
# 安装：pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func

def attention(q, k, v, heads):
    b, seq_len, _ = q.shape
    dim_head = q.shape[-1] // heads
    q = q.view(b, seq_len, heads, dim_head)
    k = k.view(b, seq_len, heads, dim_head)
    v = v.view(b, seq_len, heads, dim_head)
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    return out.reshape(b, seq_len, heads * dim_head)
```

**Sage Attention**（有损但更快）：

```python
# 安装：pip install sageattention
from sageattention import sageattn

def attention(q, k, v, heads):
    b, seq_len, _ = q.shape
    dim_head = q.shape[-1] // heads
    q = q.view(b, seq_len, heads, dim_head).transpose(1, 2)
    k = k.view(b, seq_len, heads, dim_head).transpose(1, 2)
    v = v.view(b, seq_len, heads, dim_head).transpose(1, 2)
    out = sageattn(q, k, v, is_causal=False)
    return out.transpose(1, 2).reshape(b, seq_len, heads * dim_head)
```

---

## Torch 计算加速

```python
import torch

# PyTorch >= 2.7，约 10-15% 加速，但有精度损失
if hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

---

# 常见模型配置

| 模型 | 架构特点 | 推荐方案 |
|------|---------|---------|
| LTX-Video | 单 DiT + 单 TE | 流式轮转 |
| Mochi | 单 DiT + 单 TE | 流式轮转 |
| HunyuanVideo | 大 DiT + 多 TE | FP8(DiT) + 流式轮转 |
| CogVideoX | DiT + Image Encoder | 流式轮转 |
| Flux | 大 DiT + CLIP + T5 | FP8(DiT) + 流式轮转 + FA2 |
| SD3 | DiT + CLIP×2 + T5 | 流式轮转 |
| Wan2.1/2.2 | 多阶段 DiT | FP8(所有DiT) + 流式轮转 |

---

# 故障排除

| 问题 | 解决方案 |
|------|---------|
| FP8 后仍 OOM | 同时启用流式轮转，确保所有大模型都 offload |
| 多个 Text Encoder 漏掉 | 检查是否对所有 text_encoder_N 都做了 offload |
| 多阶段 DiT 漏掉 | 检查是否对所有 transformer_N 都做了 offload |
| 流式轮转太慢 | 确保 `use_stream=True`，尝试 `block_level` |
| 蒸馏 LoRA 效果不佳 | 确认步数设置正确，尝试调整 LoRA 强度 |
| 推理质量下降 | 1. 检查是否量化了 Text Encoder<br>2. 检查是否启用了 Torch 计算加速<br>3. 如使用 Sage Attention 考虑换 Flash Attention |
| Flash Attention 安装失败 | 尝试 `pip install flash-attn --no-build-isolation` 或改用 Sage Attention |

---

# 参考代码

```
/home/ecs-user/.claude/skills/dit-inference-accelerator/scripts/
├── analyze_model.py          # 诊断分析工具
├── quantization/             # FP8 量化
├── attention/                # Attention 优化 (Flash/Sage)
└── group_offload/            # 流式轮转
```

---

# 参考资源

- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Sage Attention](https://github.com/thu-ml/SageAttention)
- [diffusers Group Offloading](https://huggingface.co/docs/diffusers/main/en/optimization/memory)
- [torchao FP8](https://github.com/pytorch/ao)
