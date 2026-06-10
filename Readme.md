1. README.md
# Knowledge Distillation with VERL for Math Problem Solving

基于 VERL 框架的知识蒸馏训练，使用 Qwen3-8B 作为教师模型，Qwen3-1.7B 作为学生模型，在 MATH-7500 数据集上进行训练。

## 特性

- **三种蒸馏策略**：
  - Forward KL: `KL(P_teacher || P_student)`
  - Reverse KL: `KL(P_student || P_teacher)` 
  - Entropy-weighted JS: 基于熵的动态加权 Jensen-Shannon 散度

- **Entropy-weighted JS 创新点**：
  组合分布 M 使用 exponential aggregation，熵越低的模型权重越高：
weight_i = exp(-H_i / temp) / Σ exp(-H_j / temp) M = weight_teacher * P_teacher + weight_student * P_student


- **自动化评估**：训练后自动在 MATH-500 上评估，每题生成 10 次，输出详细统计

## 环境要求

```bash
# Python 3.9+
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
peft>=0.7.0
verl>=0.1.0
pyyaml
numpy
pandas
安装
git clone https://github.com/yourusername/distill-verl.git
cd distill-verl
pip install -r requirements.txt

# 设置镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
快速开始
单个实验
# Forward KL
python train_distill.py --config configs/forward_kl.yaml --no-verl

# Reverse KL
python train_distill.py --config configs/reverse_kl.yaml --no-verl

# Entropy-weighted JS
python train_distill.py --config configs/entropy_js.yaml --no-verl
运行所有实验并评估
bash run_all_experiments.sh
这会依次训练三个模型，然后在 MATH-500 上评估，输出到 results/ 目录。

输出文件
每个实验会生成：

results/{method}/
├── checkpoint/              # 训练好的模型
├── responses.json           # 每次推理的完整记录（NDJSON 格式）
├── tokens.csv              # Token级别的熵和权重
└── summary.csv             # 每次推理的汇总统计
配置说明
关键配置参数（在 YAML 文件中）：

# 模型
teacher_model: "Qwen/Qwen3-8B"
student_model: "Qwen/Qwen3-1.7B"

# 训练
num_epochs: 3
batch_size_per_device: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4

# 蒸馏
distill_mode: "forward_kl"  # forward_kl | reverse_kl | entropy_js
temperature: 2.0
alpha: 0.5  # CE loss 权重
beta: 0.5   # KL/JS loss 权重

# Entropy-weighted JS 特有
entropy_temp: 1.0  # exponential aggregation 温度
实现细节
蒸馏损失
Forward KL (知识从教师流向学生):

loss = KL(P_teacher || P_student) = Σ P_teacher * log(P_teacher / P_student)
Reverse KL (学生拟合教师的高概率区域):

loss = KL(P_student || P_teacher) = Σ P_student * log(P_student / P_teacher)
Entropy-weighted JS:

# 动态计算权重
H_teacher = -Σ P_teacher * log(P_teacher)
H_student = -Σ P_student * log(P_student)
w_teacher = exp(-H_teacher / temp) / Z
w_student = exp(-H_student / temp) / Z

# 组合分布
M = w_teacher * P_teacher + w_student * P_student

# JS 散度
loss = 0.5 * KL(P_teacher || M) + 0.5 * KL(P_student || M)
VERL 集成
使用 VERL 的 Actor-Critic 架构，但将 Critic 替换为教师模型的 logits 作为监督信号。

评估指标
Accuracy: 最终答案是否正确
Mean Entropy: 平均聚合熵
Perplexity (PPL): 基于交叉熵的困惑度
Token Count: 生成的 token 数量
引用
如果使用本代码，请引用：

@software{distill_verl_2026,
  title = {Knowledge Distillation with VERL for Math Problem Solving},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/distill-verl}
}
License
MIT License

