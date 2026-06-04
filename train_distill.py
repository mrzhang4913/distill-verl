import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# train_distill.py
"""
基于 VERL 框架的知识蒸馏训练脚本

支持三种蒸馏模式：
1. Forward KL: KL(P_teacher || P_student)
2. Reverse KL: KL(P_student || P_teacher)
3. Entropy-weighted JS: 基于熵的动态加权 JS 散度
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback

# VERL imports
try:
    import verl
    from verl.trainer import RLTrainer
    from verl.utils.distributed import get_rank, get_world_size, init_distributed
except ImportError:
    verl = None
    print("Warning: VERL not installed. Using HuggingFace Trainer instead.")

    # Fallback 函数
    def get_rank():
        return 0

    def get_world_size():
        return 1

    def init_distributed():
        pass

# 本地模块
from src.data import MATHDataset, MATHDataCollator
from src.models import load_teacher_model, load_student_model_with_lora
from src.losses import get_distillation_loss
from src.utils import load_config, set_seed, get_logger, print_gpu_memory


# ============================================================
# VERL Distillation Trainer
# ============================================================

class VERLDistillationTrainer:
    """
    基于 VERL 的蒸馏训练器
    
    VERL 原本用于 RLHF，这里改造为知识蒸馏：
    - Actor: 学生模型
    - Critic: 不需要（教师 logits 作为监督信号）
    - Reward: 基于蒸馏损失
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        teacher_model,
        student_model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
    ):
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 训练配置
        self.training_config = config["training"]
        self.distill_config = config["distillation"]
        
        # 损失函数
        self.loss_fn = get_distillation_loss(
            mode=self.distill_config["mode"],
            temperature=self.distill_config["temperature"],
            alpha=self.distill_config["alpha"],
            beta=self.distill_config["beta"],
            ignore_index=self.distill_config["ignore_index"],
            entropy_temp=self.distill_config.get("entropy_temp", 1.0),
        )
        
        # Logger
        self.logger = get_logger("VERLDistillTrainer")
        
        # 输出目录
        self.output_dir = Path(self.training_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        
        # 分布式训练
        self.rank = get_rank() if verl else 0
        self.world_size = get_world_size() if verl else 1
        self.is_main_process = self.rank == 0
        
        self.logger.info(f"Initialized VERLDistillationTrainer")
        self.logger.info(f"  Distillation mode: {self.distill_config['mode']}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Rank: {self.rank}/{self.world_size}")
    
    def train(self):
        """
        执行训练
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        
        # 数据加载器
        data_collator = MATHDataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config["dataset"]["max_length"],
        )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size_per_device"],
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.config["dataset"]["num_workers"],
            pin_memory=True,
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )
        
        # 学习率调度器
        num_training_steps = (
            len(train_dataloader) 
            * self.training_config["num_epochs"]
            // self.training_config["gradient_accumulation_steps"]
        )
        num_warmup_steps = int(
            num_training_steps * self.training_config["warmup_ratio"]
        )
        
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=self.training_config["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # 混合精度
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.training_config.get("fp16", False)
        )
        
        # 训练循环
        global_step = 0
        best_eval_loss = float("inf")
        
        for epoch in range(self.training_config["num_epochs"]):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.training_config['num_epochs']}")
            self.logger.info(f"{'='*60}")
            
            self.student_model.train()
            epoch_stats = self._train_epoch(
                train_dataloader,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                global_step,
            )
            
            global_step = epoch_stats["global_step"]
            
            # 评估
            if self.eval_dataset is not None and (epoch + 1) % 1 == 0:
                eval_loss = self._evaluate(epoch)
                
                # 保存最佳模型
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._save_checkpoint(epoch, "best")
                    self.logger.info(f"  New best eval loss: {eval_loss:.4f}")
            
            # 定期保存
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch+1}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training Completed!")
        self.logger.info("=" * 60)
        
        # 保存最终模型
        self._save_checkpoint(
            self.training_config["num_epochs"] - 1,
            "final"
        )
        
        self.writer.close()
    
    def _train_epoch(
        self,
        dataloader,
        optimizer,
        lr_scheduler,
        scaler,
        epoch,
        global_step,
    ) -> Dict[str, Any]:
        """
        训练一个 epoch
        """
        from tqdm import tqdm
        from src.utils.helpers import AverageMeter
        
        loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        
        self.student_model.train()
        optimizer.zero_grad()
        
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}",
            disable=not self.is_main_process,
        )
        
        for step, batch in enumerate(pbar):
            # 移动到设备
            batch = {k: v.to(self.student_model.device) 
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 混合精度训练
            with torch.cuda.amp.autocast(
                enabled=self.training_config.get("bf16", False),
                dtype=torch.bfloat16,
            ):
                # 学生前向传播
                student_outputs = self.student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                student_logits = student_outputs.logits
                
                # 教师前向传播
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True,
                    )
                    teacher_logits = teacher_outputs.logits
                
                # 计算损失
                labels = batch.get("labels", batch["input_ids"])
                loss, stats = self.loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                )
                
                # 梯度累积
                loss = loss / self.training_config["gradient_accumulation_steps"]
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积步骤
            if (step + 1) % self.training_config["gradient_accumulation_steps"] == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.training_config["max_grad_norm"],
                )
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 记录
                loss_meter.update(stats["loss/total"])
                ce_loss_meter.update(stats.get("loss/ce", 0.0))
                kl_js_loss = stats.get("loss/kl", stats.get("loss/js", 0.0))
                kl_loss_meter.update(kl_js_loss)
                
                # TensorBoard
                if global_step % self.training_config["logging_steps"] == 0:
                    self.writer.add_scalar("train/loss", loss_meter.avg, global_step)
                    self.writer.add_scalar("train/ce_loss", ce_loss_meter.avg, global_step)
                    self.writer.add_scalar("train/kl_js_loss", kl_loss_meter.avg, global_step)
                    self.writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                    
                    for key, value in stats.items():
                        if key.startswith("entropy/") or key.startswith("weight/"):
                            self.writer.add_scalar(f"train/{key}", value, global_step)
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "ce": f"{ce_loss_meter.avg:.4f}",
                    "kl_js": f"{kl_loss_meter.avg:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # 定期保存
                if (
                    self.training_config.get("save_steps", 0) > 0
                    and global_step % self.training_config["save_steps"] == 0
                ):
                    self._save_checkpoint(epoch, f"step_{global_step}")
        
        return {
            "global_step": global_step,
            "loss": loss_meter.avg,
            "ce_loss": ce_loss_meter.avg,
            "kl_js_loss": kl_loss_meter.avg,
        }

    def _evaluate(self, epoch: int) -> float:
        """
        在验证集上评估
        
        Returns:
            平均损失
        """
        if self.eval_dataset is None:
            return float("inf")
        
        self.logger.info(f"\nEvaluating at epoch {epoch + 1}...")
        
        from src.utils.helpers import AverageMeter
        
        data_collator = MATHDataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config["dataset"]["max_length"],
        )
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.training_config["eval_batch_size"],
            shuffle=False,
            collate_fn=data_collator,
            num_workers=self.config["dataset"]["num_workers"],
        )
        
        self.student_model.eval()
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # 移动到设备
                batch = {k: v.to(self.student_model.device) 
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # 学生前向传播
                student_outputs = self.student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                student_logits = student_outputs.logits
                
                # 教师前向传播
                teacher_outputs = self.teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                teacher_logits = teacher_outputs.logits
                
                # 计算损失
                labels = batch.get("labels", batch["input_ids"])
                loss, stats = self.loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                )
                
                loss_meter.update(loss.item(), batch["input_ids"].size(0))
        
        avg_loss = loss_meter.avg
        self.logger.info(f"  Eval loss: {avg_loss:.4f}")
        
        # TensorBoard
        self.writer.add_scalar("eval/loss", avg_loss, epoch)
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, tag: str):
        """
        保存模型检查点
        
        Args:
            epoch: 当前 epoch
            tag: 检查点标签（如 "best", "epoch_3", "final"）
        """
        if not self.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints" / tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 LoRA 适配器
        self.student_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        self.logger.info(f"  Saved checkpoint: {checkpoint_dir}")
        
        # 清理旧的检查点（保留最近 N 个）
        save_total_limit = self.training_config.get("save_total_limit", 3)
        if save_total_limit > 0:
            self._cleanup_checkpoints(save_total_limit)
    
    def _cleanup_checkpoints(self, keep_last_n: int):
        """
        清理旧的检查点，保留最近的 N 个
        
        Args:
            keep_last_n: 保留的检查点数量
        """
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return
        
        # 获取所有检查点目录（排除 "best" 和 "final"）
        checkpoints = [
            d for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name not in ["best", "final"]
        ]
        
        # 按修改时间排序
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # 删除旧的检查点
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                import shutil
                shutil.rmtree(checkpoint)
                self.logger.info(f"  Removed old checkpoint: {checkpoint.name}")


# ============================================================
# Fallback: HuggingFace Trainer (如果 VERL 不可用)
# ============================================================

class HFDistillationTrainer(Trainer):
    """
    基于 HuggingFace Trainer 的蒸馏训练器
    支持 on-policy 和 off-policy 两种模式
    """

    def __init__(self, teacher_model, loss_fn, on_policy=False,
                 max_new_tokens=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model  = teacher_model
        self.loss_fn        = loss_fn
        self.on_policy      = on_policy
        self.max_new_tokens = max_new_tokens

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        """
        计算蒸馏损失
        on_policy=True:  学生先生成文本，再对齐教师分布
        on_policy=False: 直接在数据集文本上对齐教师分布
        """
        student_device = next(model.parameters()).device
        teacher_device = next(self.teacher_model.parameters()).device

        if self.on_policy:
            return self._on_policy_loss(
                model, inputs, student_device, teacher_device, return_outputs
            )
        else:
            return self._off_policy_loss(
                model, inputs, student_device, teacher_device, return_outputs
            )

    def _off_policy_loss(self, model, inputs, student_device,
                         teacher_device, return_outputs):
        """Off-policy：在固定文本上对齐教师分布"""

        student_inputs = {
            "input_ids":      inputs["input_ids"].to(student_device),
            "attention_mask": inputs["attention_mask"].to(student_device),
        }
        teacher_inputs = {
            "input_ids":      inputs["input_ids"].to(teacher_device),
            "attention_mask": inputs["attention_mask"].to(teacher_device),
        }
        labels = inputs.get("labels", inputs["input_ids"].clone())
        labels = labels.to(student_device)

        # 学生前向传播
        student_outputs = model(**student_inputs, return_dict=True)
        student_logits  = student_outputs.logits

        # 教师前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs, return_dict=True)
            teacher_logits  = teacher_outputs.logits.to(student_device)

        loss, stats = self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        self._log_stats(stats)
        return (loss, student_outputs) if return_outputs else loss

    def _on_policy_loss(self, model, inputs, student_device,
                        teacher_device, return_outputs):
        """
        On-policy：学生先生成，再对齐教师分布

        流程：
        1. 用 prompt 让学生生成 response
        2. 拼接 [prompt + generated_response]
        3. 教师和学生都在完整序列上前向传播
        4. 只对 generated 部分计算 KL 损失
        """
        input_ids      = inputs["input_ids"].to(student_device)
        attention_mask = inputs["attention_mask"].to(student_device)
        prompt_lengths = inputs.get("prompt_lengths", None)

        # ── Step 1：学生生成 response ─────────────────────────
        with torch.no_grad():
            # 禁止生成 thinking 标签
            think_ids = [
                self.tokenizer.encode('<think>', add_special_tokens=False),
                self.tokenizer.encode('</think>', add_special_tokens=False),
            ]
            bad_words_ids = [ids for ids in think_ids if ids]
            
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids if bad_words_ids else None,
            )

        # generated shape: [batch, prompt_len + gen_len]
        # attention mask for full sequence
        full_mask = (generated != self.tokenizer.pad_token_id).long()

        # ── Step 2：学生在完整序列上前向传播 ──────────────────
        student_outputs = model(
            input_ids=generated.to(student_device),
            attention_mask=full_mask.to(student_device),
            return_dict=True,
        )
        student_logits = student_outputs.logits   # [B, T, V]

        # ── Step 3：教师在完整序列上前向传播 ──────────────────
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=generated.to(teacher_device),
                attention_mask=full_mask.to(teacher_device),
                return_dict=True,
            )
            teacher_logits = teacher_outputs.logits.to(student_device)  # [B, T, V]

        # ── Step 4：只取生成部分计算损失 ──────────────────────
        # prompt_lengths: 每个样本的 prompt 长度
        if prompt_lengths is not None:
            # 取所有样本中最短的 prompt 长度，保守截取
            min_prompt_len = int(prompt_lengths.min().item())
        else:
            # fallback：用 input_ids 的长度作为 prompt 长度
            min_prompt_len = input_ids.shape[1]

        # 只保留生成部分的 logits（去掉 prompt 对应的位置）
        # logits[i] 预测的是 token[i+1]，所以 prompt 的最后一个位置
        # 预测的是生成部分第一个 token
        gen_student_logits = student_logits[:, min_prompt_len - 1: -1, :]
        gen_teacher_logits = teacher_logits[:, min_prompt_len - 1: -1, :]
        gen_labels         = generated[:, min_prompt_len:].to(student_device)

        # 对齐长度（防止边界问题）
        min_len = min(
            gen_student_logits.shape[1],
            gen_teacher_logits.shape[1],
            gen_labels.shape[1],
        )
        gen_student_logits = gen_student_logits[:, :min_len, :]
        gen_teacher_logits = gen_teacher_logits[:, :min_len, :]
        gen_labels         = gen_labels[:, :min_len]

        # ── Step 5：计算损失 ───────────────────────────────────
        loss, stats = self.loss_fn(
            student_logits=gen_student_logits,
            teacher_logits=gen_teacher_logits,
            labels=gen_labels,
        )

        self._log_stats(stats)
        return (loss, student_outputs) if return_outputs else loss

    def _log_stats(self, stats: dict):
        """记录训练统计信息"""
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in stats.items():
                self.log({f"train/{key}": value})


# ============================================================
# 主训练函数
# ============================================================

def train_distillation(config_path: str, use_verl: bool = False):
    """
    执行蒸馏训练
    """
    # 加载配置
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config.get("seed", 42))

    # Logger
    logger = get_logger("train_distillation")
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Training")
    logger.info("=" * 60)
    logger.info(f"Config:     {config_path}")
    logger.info(f"Experiment: {config['experiment_name']}")
    logger.info(f"Mode:       {config['distillation']['mode']}")

    # 加载教师模型
    logger.info("\nLoading Teacher Model...")
    teacher_tokenizer, teacher_model = load_teacher_model(
        model_name=config["teacher_model"],
        device_map="auto",
        torch_dtype=torch.bfloat16
            if config["training"].get("bf16", False) else torch.float32,
    )

    # 加载学生模型
    logger.info("\nLoading Student Model...")
    student_tokenizer, student_model = load_student_model_with_lora(
        model_name=config["student_model"],
        lora_config=config["lora_config"],
        device_map="auto",
        torch_dtype=torch.bfloat16
            if config["training"].get("bf16", False) else torch.float32,
    )

    # 设置 padding_side（left-padding 对生成更安全）
    student_tokenizer.padding_side = "left"
    teacher_tokenizer.padding_side = "left"
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token    = student_tokenizer.eos_token
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    # gradient_checkpointing 和 use_cache 不兼容，禁用 use_cache
    if config["training"].get("gradient_checkpointing", False):
        student_model.config.use_cache = False
        teacher_model.config.use_cache = False
        logger.info("  Disabled use_cache (incompatible with gradient_checkpointing)")

    # 验证词表一致性
    assert teacher_tokenizer.vocab_size == student_tokenizer.vocab_size, (
        f"Vocab size mismatch: "
        f"Teacher={teacher_tokenizer.vocab_size}, "
        f"Student={student_tokenizer.vocab_size}"
    )
    logger.info(f"✓ Vocab sizes match: {teacher_tokenizer.vocab_size}")

    # 打印显存状态
    if torch.cuda.is_available():
        print_gpu_memory()

    # 加载训练集
    on_policy = config["distillation"].get("on_policy", False)
    logger.info(f"\nLoading Train Dataset (on_policy={on_policy})...")
    train_dataset = MATHDataset(
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["train_split"],
        tokenizer=student_tokenizer,
        max_length=config["dataset"]["max_length"],
        max_prompt_length=config["dataset"].get("max_prompt_length", 512),
        system_prompt=config["system_prompt"],
        add_solution=not on_policy,  # on-policy 不需要 solution
        on_policy=on_policy,
    )

    # 加载验证集（用训练集前 100 条）
    eval_dataset = None
    if config["training"].get("eval_steps", 0) > 0:
        logger.info("\nLoading Eval Dataset (first 100 samples)...")
        eval_dataset = MATHDataset(
            dataset_name=config["dataset"]["name"],
            split=config["dataset"]["train_split"],
            tokenizer=student_tokenizer,
            max_length=config["dataset"]["max_length"],
            system_prompt=config["system_prompt"],
            add_solution=True,
        )
        eval_dataset.dataset = eval_dataset.dataset.select(
            range(min(100, len(eval_dataset)))
        )
        logger.info(f"  Eval dataset: {len(eval_dataset)} samples")

    # 损失函数
    on_policy = config["distillation"].get("on_policy", False)
    loss_fn = get_distillation_loss(
        mode=config["distillation"]["mode"],
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        beta=config["distillation"]["beta"],
        ignore_index=config["distillation"]["ignore_index"],
        entropy_temp=config["distillation"].get("entropy_temp", 1.0),
        on_policy=on_policy,
    )
    if on_policy:
        logger.info("  On-policy mode: using KL loss only (CE disabled)")
    else:
        logger.info(f"  Off-policy mode: alpha={config['distillation']['alpha']} * CE + beta={config['distillation']['beta']} * KL")

    # Data Collator
    data_collator = MATHDataCollator(
        tokenizer=student_tokenizer,
        padding=True,
        max_length=config["dataset"]["max_length"],
    )

    # Training Arguments
    logger.info("\nUsing HuggingFace Trainer")
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size_per_device"],
        per_device_eval_batch_size=config["training"].get("eval_batch_size", 4),
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        bf16=config["training"].get("bf16", False),
        fp16=config["training"].get("fp16", False),
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_checkpointing=config["training"].get(
            "gradient_checkpointing", False
        ),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config["training"].get("eval_steps", 500),
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=config["training"].get(
            "ddp_find_unused_parameters", False
        ),
        remove_unused_columns=False,   # ✅ 关键：不自动删除列
    )

    # 创建 Trainer
    trainer = HFDistillationTrainer(
        teacher_model=teacher_model,
        loss_fn=loss_fn,
        on_policy=on_policy,
        max_new_tokens=config["distillation"].get("max_new_tokens", 512),
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.tokenizer = student_tokenizer

    # 训练
    logger.info("\nStarting training...")
    trainer.train()

    # 保存最终模型
    final_dir = os.path.join(config["training"]["output_dir"], "final")
    trainer.save_model(final_dir)
    student_tokenizer.save_pretrained(final_dir)
    logger.info(f"\n✓ Model saved to: {final_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Completed Successfully!")
    logger.info("=" * 60)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Training with VERL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--no-verl",
        action="store_true",
        help="Disable VERL and use HuggingFace Trainer",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        train_distillation(
            config_path=args.config,
            use_verl=not args.no_verl,
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
