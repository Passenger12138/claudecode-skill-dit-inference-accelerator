# DiT Inference Accelerator

🚀 **在消费级 24GB 显卡（RTX 4090/3090）上部署和加速超大规模 DiT 模型**

让几十B参数的 Diffusion Transformer 模型在消费级显卡上流畅运行，涵盖视频生成和图像生成模型。

---

## 📋 支持的模型

### 视频生成
- **LTX-Video**
- **Mochi**
- **CogVideoX**
- **HunyuanVideo**
- **Wan 2.1/2.2**

### 图像生成
- **Flux** (Dev/Schnell)
- **Stable Diffusion 3** (SD3)
- **SDXL**

---

## ⚡ 核心优化技术

### 显存优化（让模型能放进 24GB）

| 技术 | 原理 | 显存占用 | 精度损失 |
|------|------|---------|---------|
| **FP8 量化** | 降低权重精度到 8-bit | ~50% | 极小 |
| **Group Offloading** | CPU/GPU 动态传输 | ~10% | 无 |

### 推理加速

| 技术 | 加速效果 | 精度影响 |
|------|---------|---------|
| **步数蒸馏 LoRA** | ~5x | 极小 |
| **Flash Attention 2** | ~1.3x | 无损 |
| **Sage Attention** | ~1.6x | 有损 |
| **计算精度优化** | ~1.1x | 有损 |

---

## 🛠️ 项目结构

```
dit-inference-accelerator/
├── scripts/
│   ├── analyze_model.py          # 模型组件分析工具
│   ├── attention/                # Attention 优化模块
│   │   ├── flash_attention.py    # Flash Attention 2 实现
│   │   ├── sage_attention.py     # Sage Attention 实现
│   │   └── processor.py          # Attention 处理器替换
│   ├── quantization/             # 量化模块
│   │   ├── quantize_model.py     # FP8 量化入口
│   │   ├── fp8_cast.py          # FP8 类型转换
│   │   ├── fp8_scaled_mm.py     # FP8 矩阵乘法
│   │   ├── policy.py            # 量化策略
│   │   └── base.py              # 基础量化类
│   └── group_offload/            # 分组卸载模块
│       ├── group_offloading.py  # 分组卸载实现
│       └── offloading.py        # 基础卸载工具
└── SKILL.md                      # 完整技术文档
```

---

## 🚀 快速开始

### 1. 分析模型组件

```python
from scripts.analyze_model import analyze_pipeline_components

# 分析 pipeline 的所有组件
components = analyze_pipeline_components(pipe)

# 查看各组件显存占用
for name, info in components.items():
    print(f"{name}: {info['params_b']:.2f}B 参数, {info['bf16_gb']:.2f}GB 显存")
```

### 2. FP8 量化（降低显存占用）

```python
from scripts.quantization import quantize_model_fp8

# 量化 Transformer
pipe.transformer = quantize_model_fp8(
    pipe.transformer,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    quantize_mode="scaled_mm"  # 或 "cast"
)
```

### 3. Group Offloading（流式轮转）

```python
from scripts.group_offload import enable_group_offloading

# 启用分组卸载
enable_group_offloading(
    pipe.transformer,
    num_groups=4,  # 将模型分成 4 组
    strategy="balanced"  # 或 "memory_first"
)
```

### 4. Attention 加速

```python
from scripts.attention import replace_attention

# 替换为 Flash Attention 2
replace_attention(pipe.transformer, mode="flash")

# 或使用 Sage Attention
replace_attention(pipe.transformer, mode="sage")
```

---

## 📦 依赖要求

```bash
# 核心依赖
torch >= 2.0.0
diffusers >= 0.28.0
transformers >= 4.40.0

# FP8 量化
torchao >= 0.5.0        # 或 transformer_engine

# Attention 加速
flash-attn >= 2.5.0     # Flash Attention 2
sageattention >= 1.0.0  # Sage Attention（可选）
```

---

## 📖 使用示例

### 完整示例：Flux.1-dev 在 4090 上运行

```python
import torch
from diffusers import FluxPipeline
from scripts.quantization import quantize_model_fp8
from scripts.attention import replace_attention

# 1. 加载模型
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# 2. FP8 量化 Transformer
pipe.transformer = quantize_model_fp8(
    pipe.transformer,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
)

# 3. 启用 Flash Attention 2
replace_attention(pipe.transformer, mode="flash")

# 4. 推理
image = pipe(
    prompt="A cat sitting on a bench",
    num_inference_steps=28,
    guidance_scale=3.5
).images[0]

image.save("output.png")
```

---

## 🔬 技术细节

详细的技术文档和配置流程请参考 [SKILL.md](./SKILL.md)

包括：
- 📊 模型组件分析方法
- 🎯 量化策略选择
- ⚙️ Group Offloading 配置
- 🚀 Attention 加速对比
- 💡 完整的配置流程指南

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 🙏 致谢

本项目基于以下开源项目：
- [diffusers](https://github.com/huggingface/diffusers) - Hugging Face 的扩散模型库
- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Flash Attention 2 实现
- [sageattention](https://github.com/thu-ml/SageAttention) - Sage Attention 实现
- [torchao](https://github.com/pytorch/ao) - PyTorch 量化工具

---

**如有问题或建议，欢迎提交 Issue！** 🎉
