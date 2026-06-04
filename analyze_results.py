import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# analyze_results.py
"""
分析和比较三种蒸馏方法的结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: Path) -> pd.DataFrame:
    """
    加载单个实验的结果
    """
    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"Warning: {summary_csv} not found")
        return None
    
    df = pd.read_csv(summary_csv)
    return df


def analyze_method(method_name: str, results_dir: Path) -> dict:
    """
    分析单个方法的结果
    """
    df = load_results(results_dir)
    if df is None:
        return None
    
    stats = {
        "method": method_name,
        "accuracy": df["is_correct"].mean() * 100,
        "problem_accuracy": df.groupby("q_idx")["is_correct"].mean().mean() * 100,
        "mean_entropy": df["mean_entropy"].mean(),
        "std_entropy": df["mean_entropy"].std(),
        "mean_ppl": df["ppl"].replace([np.inf, -np.inf], np.nan).dropna().mean(),
        "median_ppl": df["ppl"].replace([np.inf, -np.inf], np.nan).dropna().median(),
        "mean_tokens": df["num_tokens"].mean(),
        "std_tokens": df["num_tokens"].std(),
        "mean_time": df["time_seconds"].mean(),
        "total_samples": len(df),
    }
    
    return stats


def compare_all_methods(base_dir: Path = Path("./results")):
    """
    比较所有三种方法
    """
    methods = ["forward_kl", "reverse_kl", "entropy_js"]
    all_stats = []
    
    print("=" * 80)
    print("Comparing Knowledge Distillation Methods")
    print("=" * 80)
    print()
    
    for method in methods:
        method_dir = base_dir / method
        stats = analyze_method(method, method_dir)
        
        if stats is not None:
            all_stats.append(stats)
            
            print(f"--- {method.upper().replace('_', ' ')} ---")
            print(f"  Accuracy:               {stats['accuracy']:.2f}%")
            print(f"  Problem-level Accuracy: {stats['problem_accuracy']:.2f}%")
            print(f"  Mean Entropy:           {stats['mean_entropy']:.4f} ± {stats['std_entropy']:.4f}")
            print(f"  Mean PPL:               {stats['mean_ppl']:.2f} (median: {stats['median_ppl']:.2f})")
            print(f"  Mean Tokens:            {stats['mean_tokens']:.1f} ± {stats['std_tokens']:.1f}")
            print(f"  Mean Time:              {stats['mean_time']:.2f}s")
            print(f"  Total Samples:          {stats['total_samples']}")
            print()
    
    if not all_stats:
        print("No results found!")
        return
    
    # 创建对比表
    df_stats = pd.DataFrame(all_stats)
    
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print()
    print(df_stats.to_string(index=False))
    print()
    
    # 保存到 CSV
    output_file = base_dir / "comparison.csv"
    df_stats.to_csv(output_file, index=False)
    print(f"Comparison saved to: {output_file}")
    
    # 可视化（如果有 matplotlib）
    try:
        plot_comparison(all_stats, base_dir)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")


def plot_comparison(all_stats: list, base_dir: Path):
    """
    绘制对比图
    """
    df = pd.DataFrame(all_stats)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Knowledge Distillation Methods Comparison", fontsize=16)
    
    # Accuracy
    axes[0, 0].bar(df["method"], df["accuracy"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Response Accuracy")
    axes[0, 0].set_ylim(0, 100)
    
    # Problem-level Accuracy
    axes[0, 1].bar(df["method"], df["problem_accuracy"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Problem-level Accuracy")
    axes[0, 1].set_ylim(0, 100)
    
    # Mean Entropy
    axes[0, 2].bar(df["method"], df["mean_entropy"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 2].errorbar(df["method"], df["mean_entropy"], yerr=df["std_entropy"], 
                        fmt='none', color='black', capsize=5)
    axes[0, 2].set_ylabel("Entropy")
    axes[0, 2].set_title("Mean Entropy")
    
    # Mean PPL
    axes[1, 0].bar(df["method"], df["mean_ppl"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_ylabel("PPL")
    axes[1, 0].set_title("Mean Perplexity")
    
    # Mean Tokens
    axes[1, 1].bar(df["method"], df["mean_tokens"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].errorbar(df["method"], df["mean_tokens"], yerr=df["std_tokens"],
                        fmt='none', color='black', capsize=5)
    axes[1, 1].set_ylabel("Tokens")
    axes[1, 1].set_title("Mean Generated Tokens")
    
    # Mean Time
    axes[1, 2].bar(df["method"], df["mean_time"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 2].set_ylabel("Time (s)")
    axes[1, 2].set_title("Mean Generation Time")
    
    plt.tight_layout()
    
    output_file = base_dir / "comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze distillation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Base directory containing results",
    )
    
    args = parser.parse_args()
    
    compare_all_methods(Path(args.results_dir))
