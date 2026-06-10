#!/bin/bash
# run_all_experiments.sh
# 运行所有三种蒸馏实验并评估

set -e  # 遇到错误立即退出

echo "========================================"
echo "Knowledge Distillation Experiments"
echo "========================================"
echo ""

# 配置
CONFIGS=("configs/forward_kl.yaml" "configs/reverse_kl.yaml" "configs/entropy_js.yaml")
METHODS=("forward_kl" "reverse_kl" "entropy_js")
BASE_MODEL="Qwen/Qwen3-1.7B"
NUM_RESPONSES=10

# 创建结果目录
mkdir -p results

# 训练和评估每个方法
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    METHOD="${METHODS[$i]}"
    
    echo ""
    echo "========================================"
    echo "Experiment: ${METHOD}"
    echo "========================================"
    echo ""
    
    # 训练
    echo "Step 1/2: Training with ${CONFIG}..."
    python train_distill.py --config "${CONFIG}" --no-verl
    
    # 检查训练是否成功
    CHECKPOINT_DIR="./outputs/${METHOD}/checkpoints/final"
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Error: Training failed, checkpoint not found: ${CHECKPOINT_DIR}"
        exit 1
    fi
    
    echo "✓ Training completed"
    echo ""
    
    # 评估
    echo "Step 2/2: Evaluating on MATH-500..."
    OUTPUT_DIR="./results/${METHOD}"
    
    python eval_math500.py \
        --model-path "${CHECKPOINT_DIR}" \
        --base-model "${BASE_MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-responses ${NUM_RESPONSES} \
        --seed 42
    
    echo "✓ Evaluation completed"
    echo "  Results: ${OUTPUT_DIR}"
    echo ""
    
    # 清理显存
    sleep 5
done

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo ""
echo "Results summary:"
for METHOD in "${METHODS[@]}"; do
    STATS_FILE="./results/${METHOD}/statistics.txt"
    if [ -f "${STATS_FILE}" ]; then
        echo ""
        echo "--- ${METHOD} ---"
        cat "${STATS_FILE}"
    fi
done

echo ""
echo "Detailed results in:"
echo "  ./results/forward_kl/"
echo "  ./results/reverse_kl/"
echo "  ./results/entropy_js/"
echo ""
