---
title: XTuner 微调实操：使用 InternLM2-Chat-1.8B 进行自定义数据微调
date: 2025-08-17
type: llm
---

XTuner 是一个高效、灵活、功能齐全的大语言模型微调工具包，由上海人工智能实验室开发。它提供了从预训练到部署的完整工具链，支持多种模型架构和训练策略。

## 核心特性

### 高效性 (Efficient)

- **低内存需求**：支持在单张 8GB GPU 上微调 7B 参数的大语言模型
- **多节点训练**：可扩展至 70B+参数模型的多节点微调
- **高性能算子**：自动调度 FlashAttention 和 Triton kernels 提升训练吞吐量
- **DeepSpeed 集成**：兼容多种 ZeRO 优化技术

### 灵活性 (Flexible)

- **多模型支持**：InternLM、Llama、ChatGLM、Qwen、Baichuan 等主流 LLM
- **多模态能力**：支持 LLaVA 架构的视觉-语言模型微调
- **数据管道**：支持任意格式数据集，包括开源和自定义格式
- **训练算法**：QLoRA、LoRA、全参数微调等多种选择

### 功能完整 (Full-featured)

- **训练类型**：持续预训练、指令微调、智能体微调
- **对话功能**：内置对话模板，支持直接与模型交互
- **部署集成**：无缝对接 LMDeploy 部署工具和 OpenCompass 评估框架

## 快速开始

XTuner 采用配置文件驱动的设计模式：

```bash
# 查看所有可用配置
xtuner list-cfg
```

## 实战案例：使用 InternLM2-Chat-1.8B 进行自定义数据微调

基于实际项目经验，我们以 InternLM2-Chat-1.8B 模型进行自定义数据集微调为例，展示完整的微调流程和常见问题解决方案。

### 系统要求

#### 硬件配置

- **GPU**: 8GB+ 显存（推荐 RTX 3090/4090 或 A10/A100）
- **内存**: 16GB 起步，推荐 32GB+
- **存储**: 50GB 可用空间

#### 软件环境

- **Python**: 3.10+
- **CUDA**: 11.8+ 或 12.1+
- **核心库**: PyTorch 2.0+, XTuner, mmengine, bitsandbytes

#### 性能参考

| GPU 配置 | 训练速度         | 显存占用 |
| -------- | ---------------- | -------- |
| RTX 3090 | 500-800 步/小时  | 8-12GB   |
| RTX 4090 | 600-1000 步/小时 | 8-12GB   |
| A10      | 400-600 步/小时  | 8-12GB   |
| A100     | 800-1200 步/小时 | 8-12GB   |

### 部署步骤

#### 1. 环境准备

```bash
# 创建Python环境
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env

# 方案 1: 通过 pip 直接安装（推荐）
pip install -U 'xtuner[deepspeed]'

# 方案 2: 从源码安装
# git clone https://github.com/InternLM/xtuner.git
# cd xtuner
# pip install -e '.[deepspeed]'

# 注意：XTuner 会自动安装所需依赖，包括：
# mmengine, transformers, datasets, peft, bitsandbytes, accelerate 等
```

#### 验证安装

```bash
# 检查XTuner安装
xtuner list-cfg
```

#### 2. 数据集准备

创建自定义数据集，支持 JSON 格式：

```json
[
  {
    "instruction": "只剩一个心脏了还能活吗？",
    "input": "",
    "output": "从医学角度来说，人类只有一颗心脏，如果心脏停止跳动，生命就会终止。不过如果您指的是心脏疾病，现代医学有很多治疗方案，包括心脏移植、人工心脏等技术可以帮助患者。"
  },
  {
    "instruction": "爸爸再婚，我是不是就有了个新娘？",
    "input": "",
    "output": "不是的，如果爸爸再婚，您应该称呼爸爸的新妻子为继母或阿姨。新娘是指新结婚的女性，通常用来称呼自己的配偶。"
  },
  {
    "instruction": "樟脑丸是我吃过最难吃的硬糖有奇怪的味道怎么还有人买",
    "input": "",
    "output": "樟脑丸不是糖果，而是一种防虫剂，含有化学成分，绝对不能食用！如果误食请立即就医。樟脑丸是用来防止衣物被虫蛀的，放在衣柜里使用，不是食品。"
  }
]
```

将数据保存为 `./data/target_data.json`

#### 3. 配置文件准备

XTuner 提供了多个开箱即用的配置文件，可以通过以下方式获取：

```bash
# 查看所有可用配置文件
xtuner list-cfg

# 查找 InternLM2-Chat-1.8B 相关配置
xtuner list-cfg -p internlm2_1_8b

# 复制官方配置文件到当前目录
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 ./
```

复制后会得到 `internlm2_1_8b_qlora_alpaca_e3_copy.py` 配置文件。

基于官方默认配置文件，我们需要修改以下关键部分来适配自定义数据集：

##### 修改 1: 模型路径（使用本地路径）

```python
# 官方默认
pretrained_model_name_or_path = "internlm/internlm2-chat-1_8b"

# 修改为本地路径
pretrained_model_name_or_path = "./llm/internlm2-1.8b-chat/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
```

##### 修改 2: 数据源配置（切换为自定义数据）

```python
# 官方默认
alpaca_en_path = "tatsu-lab/alpaca"

# 添加自定义数据文件路径
#微调数据存放的位置
data_files = './data/target_data.json'
```

##### 修改 3: 训练参数优化（节省显存和提高效率）

```python
# 官方默认
max_length = 2048
batch_size = 1

# 修改为
max_length = 512  # 调整为512以节省内存
batch_size = 2    # 增加批次大小以提高效率
```

##### 修改 4: 数据加载方式（支持 JSON 格式）

```python
# 官方默认
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=alpaca_en_path),
    dataset_map_fn=alpaca_map_fn,
    # ...
)

# 修改为
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_files)),
    dataset_map_fn=None,  # 移除alpaca_map_fn，直接使用JSON格式
    # ...
)
```

##### 修改 5: 评估输入（适配中文数据集）

```python
# 官方默认
evaluation_inputs = ["请给我介绍五个上海的景点", "Please tell me five scenic spots in Shanghai"]

# 修改为与训练数据匹配的中文问题
evaluation_inputs = [
    "只剩一个心脏了还能活吗？",
    "爸爸再婚，我是不是就有了个新娘？",
    "樟脑丸是我吃过最难吃的硬糖有奇怪的味道怎么还有人买",
    "马上要上游泳课了，昨天洗的泳裤还没干，怎么办",
    "我只出生了一次，为什么每年都要庆生",
]
```

创建完整的配置文件 `internlm2_chat_1_8b_qlora_alpaca_e3.py`，将以上修改应用到官方模板中。

#### 4. 开始训练

```bash
# 设置环境变量（可选，针对某些CUDA兼容性问题）
export CUDA_VISIBLE_DEVICES=0

# 开始训练
xtuner train internlm2_chat_1_8b_qlora_alpaca_e3.py
```

#### 常见问题解决

##### 问题 1: ModuleNotFoundError: No module named 'mmengine'

```bash
# 解决方案
pip install mmengine
```

##### 问题 2: bitsandbytes CUDA 支持问题

```bash
# 重新安装支持CUDA的bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes
```

##### 问题 3: ModuleNotFoundError: No module named 'triton.ops'

```bash
# 安装triton
pip install triton
```

##### 问题 4: ModuleNotFoundError: No module named 'protobuf'

```bash
# 安装protobuf
pip install protobuf
```

#### 5. 模型转换

```bash
# 将训练好的LoRA权重转换为HuggingFace格式
xtuner convert pth_to_hf \
    ./internlm2_chat_1_8b_qlora_alpaca_e3.py \
    ./work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3/epoch_3.pth \
    ./hf_adapter

# 合并LoRA权重到基础模型
xtuner convert merge \
    ./llm/internlm2-1.8b-chat/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
    ./hf_adapter \
    ./merged_model \
    --max-shard-size 2GB
```

## 扩展阅读

### 官方资源

- [XTuner 官方文档](https://xtuner.readthedocs.io/zh-cn/latest/)
- [GitHub 仓库](https://github.com/InternLM/xtuner)
- [模型集合](https://huggingface.co/xtuner)

### 论文参考

- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- QLoRA: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## 总结

通过本文的实操指南，您可以使用 XTuner 快速完成 InternLM2-Chat-1.8B 模型的自定义数据微调。整个流程包括环境准备、数据集制作、配置文件修改、训练执行、模型转换和效果测试等关键步骤。

XTuner 作为一个功能完整的 LLM 微调工具包，通过 QLoRA 技术实现了在有限硬件资源下的高效微调，为开发者提供了便捷的大语言模型定制化解决方案。
