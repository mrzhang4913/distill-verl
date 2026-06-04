# src/models/distill_models.py
"""
模型加载工具
"""

import os
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def find_local_model_path(model_name: str) -> str:
    """
    查找本地缓存的模型路径
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen3-8B"
    
    Returns:
        本地路径
    """
    if os.path.exists(model_name):
        return model_name
    
    hf_cache = os.path.expanduser(
        os.environ.get("HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    )
    cache_name = "models--" + model_name.replace("/", "--")
    snap_dir = os.path.join(hf_cache, cache_name, "snapshots")
    
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f"No local cache found: {snap_dir}")
    
    # 使用最新的 snapshot
    latest = sorted(os.listdir(snap_dir))[-1]
    return os.path.join(snap_dir, latest)


def load_teacher_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedTokenizer, AutoModelForCausalLM]:
    """
    加载教师模型（冻结参数）
    
    Args:
        model_name: 模型名称
        device_map: 设备映射策略
        torch_dtype: 数据类型
    
    Returns:
        (tokenizer, model)
    """
    path = find_local_model_path(model_name)
    print(f"Loading Teacher: {model_name}")
    print(f"  Path: {path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # 冻结所有参数
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  ✓ Teacher loaded and frozen")
    return tokenizer, model


def load_student_model_with_lora(
    model_name: str,
    lora_config: dict,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[str] = None,
) -> Tuple[PreTrainedTokenizer, PeftModel]:
    """
    加载学生模型并添加 LoRA
    
    Args:
        model_name: 模型名称
        lora_config: LoRA 配置字典
        device_map: 设备映射策略
        torch_dtype: 数据类型
        checkpoint_path: 如果提供，加载已训练的 LoRA checkpoint
    
    Returns:
        (tokenizer, peft_model)
    """
    path = find_local_model_path(model_name)
    print(f"Loading Student: {model_name}")
    print(f"  Path: {path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # 设置 pad_token（如果没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # 如果提供了 checkpoint，加载已训练的 LoRA
    if checkpoint_path is not None:
        print(f"  Loading LoRA checkpoint from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print(f"  ✓ Student loaded with trained LoRA")
        return tokenizer, model
    
    # 否则，添加新的 LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
    )
    
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    
    print(f"  ✓ Student loaded with LoRA")
    return tokenizer, model


def load_student_for_inference(
    model_name: str,
    checkpoint_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedTokenizer, PeftModel]:
    """
    加载训练好的学生模型用于推理
    
    Args:
        model_name: 基座模型名称
        checkpoint_path: LoRA checkpoint 路径
        device_map: 设备映射
        torch_dtype: 数据类型
    
    Returns:
        (tokenizer, model)
    """
    return load_student_model_with_lora(
        model_name=model_name,
        lora_config={},  # 从 checkpoint 加载时不需要
        device_map=device_map,
        torch_dtype=torch_dtype,
        checkpoint_path=checkpoint_path,
    )
