import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# eval_math500.py
"""
在 MATH-500 数据集上评估训练好的模型

修复了原始代码的 bugs：
1. 正确的 NDJSON 格式
2. 断点续跑不重复写入
3. 设备分配清晰
4. KV Cache 正确释放
5. 词表大小检查
"""

import os
os.environ["TRANSFORMERS_OFFLINE"]    = "1"
os.environ["HF_DATASETS_OFFLINE"]     = "1"
os.environ["HF_ENDPOINT"]             = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import csv
import json
import time
import random
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

from src.models import load_student_for_inference
from src.data import normalize_answer, is_correct, extract_final_answer
from src.utils import set_seed, get_logger


# ============================================================
# 配置
# ============================================================

class EvalConfig:
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen3-1.7B",
        output_dir: str = "./eval_results",
        dataset_name: str = "HuggingFaceH4/MATH-500",
        num_responses: int = 10,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 42,
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.num_responses = num_responses
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.device = device
        
        # 系统提示词（与训练时一致）
        self.system_prompt = """You are an expert math problem solver. Your task is to solve the given problem step by step and output the final answer in a strictly defined format.

### 1. RESPONSE STRUCTURE
You MUST format your response in two parts:
1. **Step-by-step reasoning**: Explain your solution clearly.
2. **Final Answer Block**: You MUST conclude your response with the exact block below. If you omit this, your response is invalid.

<output>
[Insert ONLY the final bare answer here, with NO reasoning, NO text, and NO units]
</output>

### 2. RULES FOR THE FINAL ANSWER (Inside `<output>`)
You must strictly follow these formatting constraints for the final answer:

**✅ STRICTLY ALLOWED (DO):**
- Use PLAIN TEXT or MINIMAL LaTeX.
- Fractions: Use `\\frac{a}{b}` (Always prefer fractions over decimals unless asked otherwise).
- Math Constants/Operators: Use LaTeX only (e.g., `\\pi`, `\\sqrt{}`, `\\infty`).
- Coordinates/Tuples: Use `(a, b)` (Ensure a comma and a space).
- Sets: Use `\\{a, b\\}`.
- Intervals: Use `[a, b]`, `(a, b)`, `[a, b)`, etc.
- Degrees: Append `^\\circ` (e.g., `45^\\circ`).
- Integers: Write exactly as integers (e.g., `14`).

**❌ STRICTLY PROHIBITED (DO NOT):**
- DO NOT use auto-scaling or styling brackets: `\left`, `\right`, `\Big`, `\bigg`.
- DO NOT use text formatting: `\displaystyle`, `\boldsymbol`, `\text{}`.
- DO NOT use Unicode math symbols (e.g., NO π, √, ∞).
- DO NOT include units or trailing text (e.g., NO "radians", "cm", "x = ").
- DO NOT use trailing zeros or decimals for exact integers (e.g., NO `14.0`).
"""


# ============================================================
# 持久化 CSV 写入器
# ============================================================

class PersistentCSVWriter:
    def __init__(self, path: str, mode: str = "a"):
        self._f = open(path, mode, encoding="utf-8", newline="", buffering=1)
        self._writer = csv.writer(self._f)
    
    def writerow(self, row):
        self._writer.writerow(row)
    
    def flush(self):
        self._f.flush()
    
    def close(self):
        self._f.flush()
        self._f.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()


# ============================================================
# 文件初始化 & 断点续跑（修复 Bug 3）
# ============================================================

def detect_completed_responses(summary_csv: Path) -> Tuple[int, Dict[int, set]]:
    """
    检测已完成的响应，支持精确断点续跑
    
    Returns:
        start_from: 从哪个 q_idx 开始
        completed: {q_idx: {resp_idx, ...}}
    """
    if not summary_csv.exists():
        return 0, {}
    
    completed = {}
    with open(summary_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                q = int(row["q_idx"])
                r = int(row["response_index"])
                completed.setdefault(q, set()).add(r)
            except (ValueError, KeyError):
                continue
    
    if not completed:
        return 0, {}
    
    max_q = max(completed.keys())
    
    # 如果最大 q_idx 已完成所有响应，从下一个开始
    # 否则从 max_q 开始，但跳过已完成的 resp_idx
    return max_q, completed


def init_output_files(output_dir: Path, start_from: int):
    """
    初始化输出文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    responses_json = output_dir / "responses.json"
    tokens_csv = output_dir / "tokens.csv"
    summary_csv = output_dir / "summary.csv"
    
    if start_from == 0:
        # 清空或创建新文件
        responses_json.write_text("", encoding="utf-8")
        
        with open(tokens_csv, "w", encoding="utf-8", newline="") as f:
            header = [
                "q_idx", "response_index", "token_pos", "token_str",
                "entropy", "cross_entropy",
            ]
            csv.writer(f).writerow(header)
        
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                "q_idx", "response_index", "seed",
                "mean_entropy", "ppl",
                "final_answer", "ground_truth", "is_correct",
                "num_tokens", "time_seconds",
            ])
        
        print(f"✓ Initialized output files in {output_dir}\n")
    else:
        print(f"✓ Resuming from q_idx = {start_from}\n")


# ============================================================
# 生成函数
# ============================================================

def generate_response(
    model,
    tokenizer,
    problem: str,
    system_prompt: str,
    config: EvalConfig,
    seed: int,
) -> Tuple[str, List[Dict], float, float]:
    """
    生成单个响应
    
    Returns:
        generated_text: 生成的文本
        token_records: token 级别统计
        mean_entropy: 平均熵
        ppl: 困惑度
    """
    # 设置种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 构造输入
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(config.device)
    
    # 生成
    generated_ids = []
    token_records = []
    entropies = []
    cross_entropies = []
    
    past_key_values = None
    
    with torch.no_grad():
        for step in range(config.max_new_tokens):
            # 前向传播
            if step == 0:
                outputs = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1, :].float()
            else:
                next_token_tensor = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long
                ).to(config.device)
                
                outputs = model(
                    input_ids=next_token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1, :].float()
            
            # 温度缩放
            logits = logits / config.temperature
            
            # 计算概率
            logits = torch.nan_to_num(
                logits - logits.max(), nan=0.0, posinf=1e4, neginf=-1e4
            )
            probs = F.softmax(logits, dim=-1)
            
            # 熵
            entropy = float(torch.clamp(
                -torch.sum(probs * torch.log(probs + 1e-10)), min=0.0
            ).item())
            
            # 采样
            next_token_id = sample_next_token(probs, config.top_k, config.top_p)
            next_token_str = tokenizer.decode([next_token_id])
            
            # 交叉熵
            token_prob = float(probs[next_token_id].item())
            ce = float(-np.log(token_prob + 1e-10))
            
            # 记录
            token_records.append({
                "token_pos": step,
                "token_str": next_token_str,
                "entropy": entropy,
                "cross_entropy": ce,
            })
            
            entropies.append(entropy)
            cross_entropies.append(ce)
            generated_ids.append(next_token_id)
            
            # 停止条件
            if next_token_id == tokenizer.eos_token_id:
                break
            
            generated_text_so_far = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            if "</output>" in generated_text_so_far:
                break
    
    # 清理
    del past_key_values, outputs, logits, probs
    torch.cuda.empty_cache()
    
    # 解码
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 修复不完整的 <output> 标签
    if "<output>" in generated_text and "</output>" not in generated_text:
        partial = re.search(r"<output>(.*?)$", generated_text, re.DOTALL)
        if partial and partial.group(1).strip():
            generated_text += "\n</output>"
    
    # 统计
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    ppl = safe_ppl(cross_entropies)
    
    return generated_text, token_records, mean_entropy, ppl


def sample_next_token(
    probs: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """
    采样下一个 token（Top-K + Top-P）
    """
    p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    p = torch.clamp(p, min=0.0)
    s = p.sum()
    p = p / s if s > 0 else torch.ones_like(p) / p.shape[0]
    
    # Top-K
    if top_k > 0:
        k = min(top_k, p.shape[-1])
        threshold = torch.topk(p, k)[0][..., -1, None]
        p[p < threshold] = 0.0
    
    # Top-P
    if top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(p, descending=True)
        cum_p = torch.cumsum(sorted_p, dim=-1)
        remove = cum_p - sorted_p > top_p
        p[remove.scatter(-1, sorted_idx, remove)] = 0.0
    
    s = p.sum()
    p = p / s if s > 0 else torch.ones_like(p) / p.shape[0]
    
    result = torch.multinomial(p.float(), num_samples=1).item()
    return result


def safe_ppl(cross_entropies: List[float]) -> float:
    """
    安全计算困惑度
    """
    if not cross_entropies:
        return float("inf")
    arr = np.array(cross_entropies, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("inf")
    valid = np.clip(valid, 0.0, 20.0)
    return float(np.exp(np.mean(valid)))


# ============================================================
# 文件写入（修复 Bug 2：NDJSON 格式）
# ============================================================

def append_response_json(output_dir: Path, record: Dict):
    """
    追加响应记录到 NDJSON 文件
    
    修复：去掉 indent，变成真正的 NDJSON
    """
    responses_json = output_dir / "responses.json"
    with open(responses_json, "a", encoding="utf-8") as f:
        # ✅ 修复：不使用 indent，每条记录一行
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_tokens_csv(
    writer: PersistentCSVWriter,
    q_idx: int,
    resp_idx: int,
    token_records: List[Dict],
):
    """
    追加 token 级别记录
    """
    for rec in token_records:
        row = [
            q_idx,
            resp_idx,
            rec["token_pos"],
            rec["token_str"],
            f"{rec['entropy']:.6f}",
            f"{rec['cross_entropy']:.6f}",
        ]
        writer.writerow(row)


def append_summary_csv(
    writer: PersistentCSVWriter,
    q_idx: int,
    resp_idx: int,
    seed: int,
    mean_entropy: float,
    ppl: float,
    final_answer: str,
    ground_truth: str,
    num_tokens: int,
    time_seconds: float,
):
    """
    追加汇总记录
    """
    correct = is_correct(final_answer, ground_truth)
    
    # 确保反斜杠不被解释
    safe_answer = final_answer.replace('\f', '\\f').replace('\n', ' ').replace('\r', '')
    safe_truth = ground_truth.replace('\f', '\\f').replace('\n', ' ').replace('\r', '')
    
    writer.writerow([
        q_idx,
        resp_idx,
        seed,
        f"{mean_entropy:.6f}",
        f"{ppl:.4f}",
        safe_answer,
        safe_truth,
        correct,
        num_tokens,
        f"{time_seconds:.2f}",
    ])


# ============================================================
# 主评估函数
# ============================================================

def evaluate_on_math500(config: EvalConfig):
    """
    在 MATH-500 上评估模型
    """
    logger = get_logger("eval_math500")
    
    logger.info("=" * 60)
    logger.info("Evaluating on MATH-500")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Num responses per problem: {config.num_responses}")
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 加载模型
    logger.info("\n" + "=" * 60)
    logger.info("Loading Model")
    logger.info("=" * 60)
    
    tokenizer, model = load_student_for_inference(
        model_name=config.base_model,
        checkpoint_path=config.model_path,
        device_map=config.device,
        torch_dtype=torch.bfloat16,
    )
    
    model.eval()
    logger.info(f"✓ Model loaded on {config.device}")
    
    # 加载数据集
    logger.info("\n" + "=" * 60)
    logger.info("Loading MATH-500")
    logger.info("=" * 60)
    
    try:
        dataset = load_dataset(config.dataset_name, split="test")
        problems = list(dataset)
        logger.info(f"✓ Loaded {len(problems)} problems")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # 断点续跑检测
    summary_csv = config.output_dir / "summary.csv"
    start_from, completed = detect_completed_responses(summary_csv)
    
    # 初始化输出文件
    init_output_files(config.output_dir, start_from)
    
    # 打开 CSV 写入器
    tokens_csv = config.output_dir / "tokens.csv"
    summary_csv = config.output_dir / "summary.csv"
    
    with PersistentCSVWriter(tokens_csv, mode="a") as tokens_writer, \
         PersistentCSVWriter(summary_csv, mode="a") as summary_writer:
        
        # 主循环
        for q_idx in range(start_from, len(problems)):
            problem_data = problems[q_idx]
            problem = problem_data["problem"]
            ground_truth = problem_data.get("answer", "")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[Q {q_idx+1:3d}/{len(problems)}] {problem[:80]}...")
            logger.info(f"  Ground truth: {ground_truth}")
            
            # 获取已完成的 resp_idx
            done_resps = completed.get(q_idx, set())
            
            for resp_idx in range(1, config.num_responses + 1):
                # ✅ 修复 Bug 3：跳过已完成的响应
                if resp_idx in done_resps:
                    logger.info(f"  [{resp_idx:2d}/{config.num_responses}] Already completed, skipping")
                    continue
                
                seed = random.randint(0, 99999)
                t0 = time.time()
                
                try:
                    generated_text, token_records, mean_entropy, ppl = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        problem=problem,
                        system_prompt=config.system_prompt,
                        config=config,
                        seed=seed,
                    )
                except RuntimeError as e:
                    logger.error(f"  [{resp_idx:2d}/{config.num_responses}] RuntimeError: {str(e)[:120]}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    continue
                except Exception as e:
                    logger.error(f"  [{resp_idx:2d}/{config.num_responses}] Error: {type(e).__name__}: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    continue
                
                # 提取答案
                final_answer = extract_final_answer(generated_text)
                correct = is_correct(final_answer, ground_truth)
                
                elapsed = time.time() - t0
                num_tokens = len(token_records)
                
                # 写入文件
                append_response_json(config.output_dir, {
                    "q_idx": q_idx,
                    "response_index": resp_idx,
                    "seed": seed,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "response_text": generated_text,
                    "final_answer": final_answer,
                    "mean_entropy": round(mean_entropy, 6),
                    "ppl": round(ppl, 4),
                    "is_correct": correct,
                    "num_tokens": num_tokens,
                    "time_seconds": round(elapsed, 2),
                })
                
                append_tokens_csv(tokens_writer, q_idx, resp_idx, token_records)
                
                append_summary_csv(
                    summary_writer, q_idx, resp_idx, seed,
                    mean_entropy, ppl, final_answer, ground_truth,
                    num_tokens, elapsed,
                )
                
                logger.info(
                    f"  [{resp_idx:2d}/{config.num_responses}] "
                    f"seed={seed:5d} | H={mean_entropy:.4f} | PPL={ppl:6.2f} | "
                    f"tokens={num_tokens:4d} | ans={str(final_answer)[:20]!r:22s} | "
                    f"{'✓' if correct else '✗'} | {elapsed:.1f}s"
                )
                
                # 清理
                del token_records, generated_text, final_answer
                
            # 每题完成后刷新
            gc.collect()
            torch.cuda.empty_cache()
            tokens_writer.flush()
            summary_writer.flush()
            
            # 每 20 题打印显存状态
            if (q_idx + 1) % 20 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(config.device) / 1e9
                    reserved = torch.cuda.memory_reserved(config.device) / 1e9
                    logger.info(
                        f"  📊 GPU: alloc={allocated:.2f}GB, reserved={reserved:.2f}GB"
                    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Completed!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"  - responses.json (NDJSON format)")
    logger.info(f"  - tokens.csv")
    logger.info(f"  - summary.csv")
    
    # 计算统计
    compute_statistics(config.output_dir)


# ============================================================
# 统计分析
# ============================================================

def compute_statistics(output_dir: Path):
    """
    计算并打印统计信息
    """
    logger = get_logger("eval_math500")
    summary_csv = output_dir / "summary.csv"
    
    if not summary_csv.exists():
        return
    
    import pandas as pd
    
    df = pd.read_csv(summary_csv)
    
    logger.info("\n" + "=" * 60)
    logger.info("Statistics")
    logger.info("=" * 60)
    
    # 准确率
    accuracy = df["is_correct"].mean() * 100
    logger.info(f"Accuracy: {accuracy:.2f}% ({df['is_correct'].sum()}/{len(df)})")
    
    # 按问题聚合
    problem_accuracy = df.groupby("q_idx")["is_correct"].mean().mean() * 100
    logger.info(f"Problem-level Accuracy: {problem_accuracy:.2f}%")
    
    # 平均熵
    mean_entropy = df["mean_entropy"].mean()
    logger.info(f"Mean Entropy: {mean_entropy:.4f}")
    
    # 平均 PPL
    mean_ppl = df["ppl"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    logger.info(f"Mean PPL: {mean_ppl:.2f}")
    
    # 平均 token 数
    mean_tokens = df["num_tokens"].mean()
    logger.info(f"Mean Tokens: {mean_tokens:.1f}")
    
    # 平均时间
    mean_time = df["time_seconds"].mean()
    logger.info(f"Mean Time: {mean_time:.2f}s")
    
    logger.info("=" * 60)
    
    # 保存统计到文件
    stats_file = output_dir / "statistics.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("MATH-500 Evaluation Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.2f}% ({df['is_correct'].sum()}/{len(df)})\n")
        f.write(f"Problem-level Accuracy: {problem_accuracy:.2f}%\n")
        f.write(f"Mean Entropy: {mean_entropy:.4f}\n")
        f.write(f"Mean PPL: {mean_ppl:.2f}\n")
        f.write(f"Mean Tokens: {mean_tokens:.1f}\n")
        f.write(f"Mean Time: {mean_time:.2f}s\n")
    
    logger.info(f"Statistics saved to: {stats_file}")
# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on MATH-500"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Dataset name",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=10,
        help="Number of responses per problem",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P (nucleus) sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvalConfig(
        model_path=args.model_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_responses=args.num_responses,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )
    
    # 执行评估
    try:
        evaluate_on_math500(config)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()