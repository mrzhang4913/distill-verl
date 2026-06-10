# 使用指南

本文档提供详细的使用说明和最佳实践。

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/distill-verl.git
cd distill-verl

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import verl; print('VERL installed')"
2. 准备模型
确保模型已下载到本地缓存：

# 设置 HuggingFace 镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（如果未缓存）
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')
"
3. 单个实验训练
# Forward KL
python train_distill.py --config configs/forward_kl.yaml --no-verl

# Reverse KL
python train_distill.py --config configs/reverse_kl.yaml --no-verl

# Entropy-weighted JS
python train_distill.py --config configs/entropy_js.yaml --no-verl
4. 评估训练好的模型
# 评估 Forward KL 模型
python eval_math500.py \
    --model-path ./outputs/forward_kl/checkpoints/final \
    --base-model Qwen/Qwen3-1.7B \
    --output-dir ./results/forward_kl \
    --num-responses 10

# 评估 Reverse KL 模型
python eval_math500.py \
    --model-path ./outputs/reverse_kl/checkpoints/final \
    --base-model Qwen/Qwen3-1.7B \
    --output-dir ./results/reverse_kl \
    --num-responses 10

# 评估 Entropy-weighted JS 模型
python eval_math500.py \
    --model-path ./outputs/entropy_js/checkpoints/final \
    --base-model Qwen/Qwen3-1.7B \
    --output-dir ./results/entropy_js \
    --num-responses 10
5. 运行完整流程
# 一键运行所有实验（训练 + 评估）
bash run_all_experiments.sh
6. 分析结果
# 比较三种方法
python analyze_results.py --results-dir ./results
配置详解
训练配置
在 YAML 配置文件中可以调整以下参数：

模型配置
teacher_model: "Qwen/Qwen3-8B"       # 教师模型
student_model: "Qwen/Qwen3-1.7B"    # 学生模型

# LoRA 配置
lora_config:
  r: 64                              # LoRA 秩（越大表达能力越强，显存占用越多）
  lora_alpha: 128                    # LoRA 缩放因子
  lora_dropout: 0.05                 # Dropout 率
  target_modules:                    # 应用 LoRA 的模块
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
调优建议：

r 值常用范围：8-128，默认 64 是较好的平衡点
lora_alpha 通常设为 2 * r
如果显存不足，可以减少 target_modules
训练超参数
training:
  num_epochs: 3                      # 训练轮数
  batch_size_per_device: 2           # 每张卡的批次大小
  gradient_accumulation_steps: 8     # 梯度累积步数
  learning_rate: 2.0e-4              # 学习率
  weight_decay: 0.01                 # 权重衰减
  warmup_ratio: 0.1                  # 预热比例
  lr_scheduler_type: "cosine"        # 学习率调度器
调优建议：

有效批次大小 = batch_size_per_device * gradient_accumulation_steps * num_gpus
建议有效批次大小在 16-32 之间
学习率可以根据批次大小调整：lr = base_lr * sqrt(batch_size / base_batch_size)
蒸馏配置
distillation:
  mode: "forward_kl"                 # forward_kl | reverse_kl | entropy_js
  temperature: 2.0                   # KL 温度（越高分布越软）
  alpha: 0.5                         # CE loss 权重
  beta: 0.5                          # KL/JS loss 权重
  entropy_temp: 1.0                  # (仅 entropy_js) 熵聚合温度
调优建议：

temperature 常用范围：1.0-4.0，越高蒸馏效果越明显但可能过度软化
alpha + beta = 1.0 是常见设置，但不强制
entropy_temp 越小，低熵模型的权重越高（默认 1.0）
分布式训练
多卡训练（DDP）
# 使用 torchrun（推荐）
torchrun --nproc_per_node=4 train_distill.py \
    --config configs/forward_kl.yaml

# 或使用 accelerate
accelerate launch --multi_gpu --num_processes=4 train_distill.py \
    --config configs/forward_kl.yaml
使用 VERL 多卡
VERL 会自动处理分布式，只需在配置中设置：

verl:
  enable: true
  actor_num_gpus: 4      # 学生模型使用的 GPU 数
  critic_num_gpus: 0     # 不需要 critic
使用 DeepSpeed
在配置中添加：

training:
  deepspeed: "configs/ds_config.json"
创建 configs/ds_config.json：

