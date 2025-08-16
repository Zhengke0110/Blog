---
title: 部署：AI视频生成模型 CogVideoX-5b 源码部署
date: 2025-08-15
type: notes-llm
---

CogVideoX-5b 是由清华大学和智谱 AI 联合开发的开源多模态视频生成大模型，基于 3D 变分自编码器（3D VAE）和扩散变换器（DiT）架构。该模型支持中英文文本到视频的生成，能够生成时长 6 秒、分辨率 720x480、帧率 8fps 的高质量视频内容。本文档详细介绍如何从源码进行本地化部署和配置优化。

## 系统要求

### 核心技术规格

CogVideoX-5b 是一个拥有 **50 亿参数** 的视频生成模型，采用了以下核心技术：

- **模型架构**: 3D 变分自编码器 (3D VAE) + 扩散变换器 (DiT)
- **输出规格**: 6 秒视频，720×480 分辨率，8fps 帧率，共 48 帧
- **语言支持**: 中文和英文双语文本提示词
- **推理精度**: 支持 FP16 和 BF16 混合精度推理

### 硬件要求详解

#### GPU 显存需求分析

不同使用场景的显存要求：

| 使用模式     | 最低显存 | 推荐显存 | 推荐 GPU 型号        | 生成质量         |
| ------------ | -------- | -------- | -------------------- | ---------------- |
| **快速体验** | 8GB      | 12GB     | RTX 4070, RTX 3080   | 基础质量 (13 帧) |
| **标准使用** | 12GB     | 18GB     | RTX 4080, RTX 4070Ti | 标准质量 (25 帧) |
| **最佳质量** | 18GB     | 24GB+    | RTX 4090, A100, H100 | 高质量 (49 帧)   |

> **注意**: 实际显存占用会根据生成参数动态变化。启用内存优化功能可降低约 20-30% 的显存需求。

#### 系统配置要求

- **CPU**: Intel i5-8400 / AMD Ryzen 5 3600 或更高
- **内存**:
  - 最低: 16GB (仅限快速体验模式)
  - 推荐: 32GB (标准使用)
  - 最佳: 64GB+ (高质量生成 + 多任务处理)
- **存储**:
  - 系统盘: 至少 20GB 可用空间
  - 模型存储: 50GB+ (CogVideoX-5b 模型约 18GB)
  - 输出空间: 建议预留 50GB+ (生成的视频文件)
- **CUDA**: 支持 CUDA 11.8 或 12.1+ (必须与 PyTorch 版本匹配)

#### 网络要求

- **初次部署**: 稳定网络连接，用于下载 18GB 模型文件
- **镜像加速**: 建议配置 HuggingFace 镜像源 (如 hf-mirror.com)
- **带宽建议**: 100Mbps+ (加快模型下载速度)

### 软件环境要求

#### Python 环境

- **Python 版本**: 3.8 - 3.11 (推荐 3.10)
- **包管理器**: pip 或 conda (推荐使用 conda 进行环境隔离)

#### 深度学习框架

- **PyTorch**: 2.0.0+ (必须支持 CUDA)
- **核心依赖**:
  - `diffusers >= 0.30.0` (视频生成管道)
  - `transformers >= 4.44.0` (文本编码器)
  - `accelerate >= 0.20.0` (分布式训练支持)

### 性能预估

#### 生成时间参考 (单个视频)

| GPU 型号 | 13 帧模式 | 25 帧模式   | 49 帧模式   |
| -------- | --------- | ----------- | ----------- |
| RTX 4090 | ~1-2 分钟 | ~2-3 分钟   | ~4-6 分钟   |
| RTX 4080 | ~2-3 分钟 | ~3-5 分钟   | ~6-9 分钟   |
| RTX 4070 | ~3-5 分钟 | ~5-8 分钟   | ~10-15 分钟 |
| A100     | ~1 分钟   | ~1-2 分钟   | ~2-4 分钟   |

> **提示**: 实际生成时间受提示词复杂度、推理步数等参数影响。首次运行需要额外的模型加载时间。

## 部署步骤

### 1. 环境准备

```bash
# 创建conda环境
conda create -n cogvideox python=3.10 -y
conda activate cogvideox

# 安装CUDA相关依赖
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 版本确认

在安装 PyTorch 之前，请确认您的 CUDA 版本：

```bash
# 检查CUDA版本
nvidia-smi
```

### 2. 安装依赖

直接安装所需的 Python 依赖包：

```bash
# 安装核心依赖包
pip install diffusers[torch]>=0.30.0
pip install transformers>=4.44.0
pip install accelerate>=0.20.0
pip install imageio-ffmpeg
pip install safetensors

# 安装ModelScope和HuggingFace工具
pip install modelscope
pip install huggingface_hub

# 安装其他必需依赖
pip install pillow
pip install numpy
pip install opencv-python
```

#### 依赖包说明

| 包名              | 版本要求 | 作用                 |
| ----------------- | -------- | -------------------- |
| `diffusers`       | >=0.30.0 | 视频生成管道核心库   |
| `transformers`    | >=4.44.0 | 文本编码器           |
| `accelerate`      | >=0.20.0 | 模型加速和内存优化   |
| `torch`           | >=2.0.0  | 深度学习框架         |
| `imageio-ffmpeg`  | 最新     | 视频文件处理         |
| `safetensors`     | 最新     | 安全的模型文件格式   |
| `modelscope`      | 最新     | ModelScope 平台工具  |
| `huggingface_hub` | 最新     | HuggingFace 平台工具 |

#### 验证依赖安装

```bash
# 验证关键包是否正确安装
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import accelerate; print('Accelerate:', accelerate.__version__)"
```

### 3. 下载模型

```bash
# 创建模型目录
mkdir -p models

# 进入模型目录
cd models
```

#### 方法一：ModelScope 下载（推荐）

```bash
# 如果还没安装ModelScope
pip install modelscope

# 下载CogVideoX-5b模型
modelscope download --model ZhipuAI/CogVideoX-5b --local_dir ./CogVideoX-5b

# 验证下载完成
ls -la ./CogVideoX-5b/
```

![ModelScope ](/images/notes/llm/Deploy-CogVideoX-5b/download.png)

#### 方法二：HuggingFace 下载

```bash
# 安装huggingface-hub CLI工具
pip install huggingface_hub[cli]

# 下载模型
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./CogVideoX-5b

# 设置镜像加速（可选，国内用户）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./CogVideoX-5b
```

### 4. 创建环境检查脚本

创建 `check_environment.py` 文件，用于检查系统环境和依赖包：

```python
import torch
import os
import sys
import shutil

def check_system_requirements():
    """检查系统要求"""
    print("=== 系统环境检查 ===")

    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    else:
        print("✅ Python版本符合要求")

    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, {gpu_memory:.1f} GB")
            if gpu_memory < 16:
                print(f"❌ GPU {i} 显存可能不足，建议18GB+")
            else:
                print(f"✅ GPU {i} 显存充足")
    else:
        print("❌ 将使用CPU模式（速度较慢）")

    return True

def check_dependencies():
    """检查依赖包"""
    print("\n=== 依赖包检查 ===")

    required_packages = [
        ("torch", "2.0.0"),
        ("diffusers", "0.30.0"),
        ("transformers", "4.44.0"),
        ("accelerate", "0.20.0"),
        ("imageio-ffmpeg", None),
        ("safetensors", None),
        ("modelscope", None),
        ("huggingface_hub", None)
    ]

    all_ok = True
    for package, min_version in required_packages:
        try:
            if package == "imageio-ffmpeg":
                __import__("imageio_ffmpeg")
            elif package == "huggingface_hub":
                __import__("huggingface_hub")
            else:
                __import__(package)

            module = __import__(package.replace("-", "_"))
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 未安装")
            all_ok = False

    return all_ok

def check_disk_space():
    """检查磁盘空间"""
    print("\n=== 磁盘空间检查 ===")

    total, used, free = shutil.disk_usage(".")
    free_gb = free // (1024**3)

    print(f"可用磁盘空间: {free_gb} GB")

    if free_gb < 10:
        print("❌ 磁盘空间不足，建议至少保留10GB用于视频生成")
        return False
    else:
        print("✅ 磁盘空间充足")
        return True

def check_model_directory():
    """简单检查模型目录是否存在"""
    print("\n=== 模型目录检查 ===")

    model_path = "./models/CogVideoX-5b"
    if os.path.exists(model_path):
        # 计算模型总大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
                file_count += 1

        total_size_gb = total_size / (1024**3)
        print(f"✅ 模型目录存在: {model_path}")
        print(f"✅ 文件数量: {file_count}")
        print(f"✅ 总大小: {total_size_gb:.2f} GB")

        if total_size_gb < 10:
            print("❌ 模型大小异常，可能下载不完整")
            return False
        else:
            print("✅ 模型大小正常")
            return True
    else:
        print(f"❌ 模型目录不存在: {model_path}")
        print("\n请先下载模型:")
        print("# ModelScope 下载 (推荐)")
        print("modelscope download --model ZhipuAI/CogVideoX-5b --local_dir ./models/CogVideoX-5b")
        print("\n# HuggingFace 下载")
        print("huggingface-cli download THUDM/CogVideoX-5b --local-dir ./models/CogVideoX-5b")
        return False

def main():
    """主检查函数"""
    print("CogVideoX-5b 部署环境检查")
    print("=" * 40)

    # 执行检查
    system_ok = check_system_requirements()
    deps_ok = check_dependencies()
    disk_ok = check_disk_space()
    model_ok = check_model_directory()

    # 总结结果
    print(f"\n{'='*40}")
    print("环境检查结果:")
    print(f"系统环境: {'✅ 通过' if system_ok else '❌ 失败'}")
    print(f"依赖包: {'✅ 通过' if deps_ok else '❌ 失败'}")
    print(f"磁盘空间: {'✅ 通过' if disk_ok else '❌ 失败'}")
    print(f"模型文件: {'✅ 通过' if model_ok else '❌ 失败'}")

    overall_status = all([system_ok, deps_ok, disk_ok, model_ok])

    if overall_status:
        print(f"\n✅ 环境检查全部通过！")
    else:
        print(f"\n❌ 环境检查未通过，请解决上述问题后重试")

    return overall_status

if __name__ == "__main__":
    main()
```

### 5. 创建视频生成脚本

创建 `generate_video.py` 文件，专注于视频生成功能：

```python
import torch
import os
import argparse
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.utils import export_to_video

def generate_video(prompt, output_path="output.mp4",
                  num_frames=49, num_inference_steps=50,
                  height=480, width=720, guidance_scale=6.0, seed=42):
    """生成视频"""

    model_path = "./models/CogVideoX-5b"

    # 快速检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        print("请先运行: python check_environment.py")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"正在生成视频: {prompt}")
    print(f"参数: {num_frames}帧, {height}x{width}, {num_inference_steps}步")

    try:
        # 加载模型
        print("正在加载模型...")
        
        # 加载调度器
        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_path, 
            subfolder="scheduler"
        )
        
        # 加载pipeline
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # 优化内存使用
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            print("✅ 启用模型CPU卸载")
            
            # 启用VAE切片以节省内存
            if hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
                print("✅ 启用VAE切片")
            
            # 启用注意力切片
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
                print("✅ 启用注意力切片")
        else:
            pipe = pipe.to(device)

        print("✅ 模型加载成功")

        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # CogVideoX-5b标准分辨率配置
        # 根据模型配置，支持的分辨率是基于 60x90 的基础尺寸
        standard_resolutions = {
            "480p": (480, 720),    # 基础分辨率
            "720p": (720, 1280),   # 高清
            "1080p": (1080, 1920), # 全高清  
        }
        
        # 选择最接近的标准分辨率
        target_ratio = width / height
        best_res = min(standard_resolutions.values(), 
                      key=lambda res: abs(res[1]/res[0] - target_ratio))
        
        final_height, final_width = best_res
        if final_height != height or final_width != width:
            print(f"⚠️ 调整视频尺寸: {height}x{width} -> {final_height}x{final_width}")
            print(f"   (使用CogVideoX支持的标准分辨率)")
        
        # 确保帧数符合模型要求
        if num_frames not in [13, 25, 49]:
            adjusted_frames = min([13, 25, 49], key=lambda x: abs(x - num_frames))
            print(f"⚠️ 调整帧数: {num_frames} -> {adjusted_frames} (模型支持: 13, 25, 49)")
            num_frames = adjusted_frames
        
        # 生成视频
        print("正在生成视频...")
        try:
            video_frames = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=device).manual_seed(seed),
                height=final_height,
                width=final_width,
            ).frames[0]
        except Exception as e:
            print(f"❌ 生成过程出错: {e}")
            # 如果出错，尝试使用最小的配置
            print("🔄 尝试使用最小配置重新生成...")
            video_frames = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=20,  # 减少步数
                num_frames=13,           # 最少帧数
                guidance_scale=6.0,
                generator=torch.Generator(device=device).manual_seed(seed),
                height=480,              # 最小分辨率
                width=720,
            ).frames[0]

        # 保存视频
        print("正在保存视频...")
        export_to_video(video_frames, output_path, fps=8)

        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ 视频生成完成: {output_path} ({file_size:.1f} MB)")
            return True
        else:
            print(f"❌ 视频文件生成失败")
            return False

    except torch.cuda.OutOfMemoryError:
        print("❌ GPU显存不足！")
        return False

    except Exception as e:
        print(f"❌ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="CogVideoX-5b 视频生成")
    parser.add_argument("--prompt", type=str, required=True, help="视频描述")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出文件路径")
    parser.add_argument("--frames", type=int, default=13, help="视频帧数 (13, 25, 49)")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=720, help="视频宽度")
    parser.add_argument("--guidance", type=float, default=6.0, help="引导尺度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    success = generate_video(
        prompt=args.prompt,
        output_path=args.output,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        seed=args.seed
    )

    if not success:
        print("视频生成失败，请检查错误信息。")
if __name__ == "__main__":
    main()
```

### 6. 使用方法

#### 简化的部署流程

```bash
# 第一步：检查环境（包含模型检查）
python check_environment.py

# 第二步：正常使用
python generate_video.py --prompt "一只大熊猫在竹林中吃竹子" --output "panda.mp4"
```

![部署](/images/notes/llm/Deploy-CogVideoX-5b/generate.png)

> **重要提示**：`generate_video.py` 脚本必须提供 `--prompt` 参数，这是生成视频的文本描述。如果直接运行 `python generate_video.py` 会显示参数错误。

#### 参数说明

| 参数         | 是否必需 | 默认值     | 说明                       | 内存影响 |
| ------------ | -------- | ---------- | -------------------------- | -------- |
| `--prompt`   | **必需** | 无         | 视频描述文本（中英文均可） | 无       |
| `--output`   | 可选     | output.mp4 | 输出视频文件路径           | 无       |
| `--frames`   | 可选     | **13**     | **视频帧数 (13, 25, 49)**  | **高**   |
| `--steps`    | 可选     | **50**     | 推理步数                   | 中       |
| `--height`   | 可选     | **480**    | 视频高度                   | 中       |
| `--width`    | 可选     | **720**    | 视频宽度                   | 中       |
| `--guidance` | 可选     | 6.0        | 引导尺度                   | 低       |
| `--seed`     | 可选     | 42         | 随机种子                   | 无       |

> **内存优化提示**：默认参数已调整为低内存模式（13 帧，480x720 分辨率）。如需高质量，可手动设置：`--frames 49 --height 480 --width 720`。
