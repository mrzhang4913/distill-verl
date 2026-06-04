# src/data/math_dataset.py
"""
MATH 数据集加载和预处理
支持 off-policy 和 on-policy 蒸馏
"""

import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class MATHDataset(Dataset):
    """MATH 数据集封装"""

    def __init__(
        self,
        dataset_name: str = "nlile/hendrycks-MATH-benchmark",
        split: str = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 2048,
        max_prompt_length: int = 512,   # prompt 最大长度
        system_prompt: str = "",
        add_solution: bool = True,
        on_policy: bool = False,        # on-policy 模式只返回 prompt
    ):
        self.dataset_name     = dataset_name
        self.split            = split
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.max_prompt_length = max_prompt_length
        self.system_prompt    = system_prompt
        self.add_solution     = add_solution
        self.on_policy        = on_policy

        print(f"Loading {dataset_name} ({split})...")
        try:
            self.dataset = load_dataset(dataset_name, split=split)
            print(f"  ✓ Loaded {len(self.dataset)} problems")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item     = self.dataset[idx]
        problem  = item["problem"]
        solution = item.get("solution", "")

        if self.tokenizer is None:
            return {"problem": problem, "solution": solution}

        if self.on_policy:
            # ── On-policy：只返回 prompt ──────────────────────
            return self._make_prompt_only(problem)
        else:
            # ── Off-policy：返回完整序列 ──────────────────────
            return self._make_full_sequence(problem, solution)

    def _make_prompt_only(self, problem: str) -> Dict[str, Any]:
        """
        只返回 prompt 部分（system + user），用于 on-policy 生成
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": problem},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 加上 <|im_start|>assistant\n
            enable_thinking=False,
        )

        encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "prompt_length":  len(encoding["input_ids"]),
        }

    def _make_full_sequence(self, problem: str, solution: str) -> Dict[str, Any]:
        """
        返回完整序列（system + user + assistant），用于 off-policy 蒸馏
        """
        messages = [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": problem},
        ]
        if self.add_solution and solution:
            messages.append({"role": "assistant", "content": solution})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        result = {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        if self.add_solution and solution:
            result["labels"] = encoding["input_ids"].copy()

        return result


@dataclass
class MATHDataCollator:
    """数据批处理（同时支持 on-policy 和 off-policy）"""

    tokenizer:          PreTrainedTokenizer
    padding:            bool          = True
    max_length:         Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # ── Input IDs ──────────────────────────────────────────
        input_ids = [f["input_ids"] for f in features if "input_ids" in f]
        if input_ids:
            batch["input_ids"] = self._pad_sequence(
                input_ids, self.tokenizer.pad_token_id
            )

        # ── Attention Mask ─────────────────────────────────────
        attn = [f["attention_mask"] for f in features if "attention_mask" in f]
        if attn:
            batch["attention_mask"] = self._pad_sequence(attn, 0)

        # ── Labels（off-policy 用）─────────────────────────────
        labels = [f["labels"] for f in features if "labels" in f]
        if labels:
            batch["labels"] = self._pad_sequence(labels, -100)

        # ── Prompt lengths（on-policy 用）──────────────────────
        prompt_lengths = [f["prompt_length"] for f in features
                          if "prompt_length" in f]
        if prompt_lengths:
            batch["prompt_lengths"] = torch.tensor(
                prompt_lengths, dtype=torch.long
            )

        return batch

    def _pad_sequence(self, sequences, pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of * self.pad_to_multiple_of
            )
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)


# ── 答案处理工具 ────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = str(s).strip()
    s = s.replace('\f', '\\f').replace('\n', ' ').replace('\r', '')
    s = s.replace("$", "")
    s = s.replace("π", "\\pi")
    s = s.replace("√", "\\sqrt")
    s = re.sub(r"\\(left|right|[Bb]ig{1,2})\s*", "", s)
    s = re.sub(r"\\(displaystyle|textstyle|scriptstyle|boldsymbol|mathbf|mathrm)\s*", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\b(\d+)\.0\b", r"\1", s)
    s = s.lower()
    return s


def is_correct(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    return re.sub(r"\s", "", p) == re.sub(r"\s", "", g)


def extract_final_answer(text: str) -> str:
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    partial = re.search(r"<output>(.*?)$", text, re.DOTALL)
    if partial and partial.group(1).strip():
        return partial.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        if not any(kw in last.lower() for kw in
                   ["therefore", "thus", "so the", "we get",
                    "answer is", "solution", "step", "let ", "we have"]):
            return last
    return ""
