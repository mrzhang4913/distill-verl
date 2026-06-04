# src/utils/helpers.py
"""
辅助工具函数
"""

import os
import random
import logging
from typing import Any, Dict

import yaml
import numpy as np
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """
    设置随机种子以保证可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 性能会稍微下降，但保证可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取 logger
    
    Args:
        name: logger 名称
        level: 日志级别
    
    Returns:
        logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def print_gpu_memory():
    """
    打印所有 GPU 的显存使用情况
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"GPU Memory Status ({num_gpus} GPUs)")
    print(f"{'='*60}")
    
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Usage:     {allocated/total*100:.1f}%")
    
    print(f"{'='*60}\n")


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
