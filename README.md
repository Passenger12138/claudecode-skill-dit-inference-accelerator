# DiT Inference Accelerator

🚀 **在消费级 24GB 显卡（RTX 4090/3090）上部署和加速超大规模 DiT 模型**

让几十B参数的 Diffusion Transformer 模型在消费级显卡上流畅运行，涵盖视频生成和图像生成模型。

---

## 📥 安装指南

### 方式 1：在线安装（推荐）

```bash
claude plugin marketplace add https://github.com/Passenger12138/claudecode-skill-dit-inference-accelerator
```

### 方式 2：离线安装

```bash
git clone https://github.com/Passenger12138/claudecode-skill-dit-inference-accelerator.git
mkdir -p ~/.claude/skills
cp -r claudecode-skill-dit-inference-accelerator ~/.claude/skills/
```

---

## 🎯 使用指南

### Claude Code Skill 自动化配置

本项目可作为 Claude Code 的 Skill 使用，通过交互式问答自动生成优化配置：

```bash
# 在 Claude Code 中调用
/dit-inference-accelerator 帮我优化 Flux 在 4090 上的运行代码
```

### Skills 执行流程

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

### 完整交互示例

<details>
<summary>点击展开查看完整交互过程</summary>

#### 步骤 1：自动分析模型架构

```
基于 FLUX.1-dev 模型的标准架构，组件分析如下：

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

#### 步骤 2：询问量化策略

```
是否对模型进行 FP8 量化？可降低约 50% 显存占用。

❯ 1. 是，使用 FP8 量化（推荐）
     显存占用减半，精度损失极小
  2. 否，保持 BF16
     显存占用高，但精度最佳
```

#### 步骤 3：询问流式轮转策略

```
请选择 Group Offloading 策略：

❯ 1. 大型组件流式轮转 + VAE 常驻 CUDA（推荐）
     transformer, text_encoder, text_encoder_2 全部进行流式轮转，VAE 常驻 CUDA。速度较快，显存占用约 2-3GB。
  2. 所有组件全部流式轮转
     所有组件（包括 VAE）全部进行流式轮转。最省显存（约 1-2GB），但速度稍慢。
  3. Type something.

```

#### 步骤 4：询问蒸馏 LoRA

```
是否已有蒸馏 LoRA？蒸馏 LoRA 可以实现约 5 倍加速（从 50 步降到 4-8 步）。

❯ 1. 是，我有蒸馏 LoRA
     请提供 LoRA 路径或 HuggingFace ID，例如 'alimama-creative/FLUX.1-Turbo-Alpha'
  2. 否，没有蒸馏 LoRA
     将使用原始 50 步推理，速度较慢但质量最佳
```

#### 步骤 5：询问 Attention 加速方式

```
请选择 Attention 加速方式。Flash Attention 2 无损但安装复杂，Sage Attention 有损但更快且易安装。

❯ 1. Flash Attention 2（推荐）
     ~1.3x 加速，无损精度，需编译安装。适合追求质量。
  2. Sage Attention
     ~1.6x 加速，有损精度（INT8 量化 QK），pip 直接安装。适合追求速度。
  3. 不使用 Attention 优化
     保持默认实现，无加速
```

#### 步骤 6：询问 Torch 计算加速

```
是否启用 PyTorch 低精度计算加速？可获得约 10-15% 加速但有精度损失，可能导致数值不稳定。

❯ 1. 不启用（推荐）
     保持默认精度，质量最佳，无风险
  2. 启用
     约 10-15% 加速，有损精度，需 PyTorch >= 2.7
```

#### 步骤 7：生成完整配置代码

Claude 会自动生成优化后的完整代码，可直接运行！

</details>

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

## 🔧 手动使用方式（适合 Python 脚本集成）

如果你想在自己的 Python 代码中手动调用各个模块，可以参考以下示例：

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

## 📖 完整示例

### Flux.1-dev 在 4090 上运行的端到端代码

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

## ⚡ 技术原理

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
