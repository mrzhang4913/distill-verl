# src/losses/distill_losses.py
"""
知识蒸馏损失函数
- Forward KL
- Reverse KL  
- Entropy-weighted JS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DistillationLoss(nn.Module):
    """蒸馏损失基类"""

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        on_policy: bool = False,
    ):
        super().__init__()
        self.temperature  = temperature
        self.ignore_index = ignore_index
        self.on_policy    = on_policy
        
        # on-policy 时强制只用 KL/JS
        if on_policy:
            self.alpha = 0.0
            self.beta  = 1.0
        else:
            self.alpha = alpha
            self.beta  = beta

    def _get_soft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """温度缩放后的概率（只用于 KL 计算）"""
        return F.softmax(logits / self.temperature, dim=-1)

    def _get_original_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """原始概率（用于熵计算）"""
        return F.softmax(logits, dim=-1)

    def _get_log_soft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """温度缩放后的 log 概率"""
        return F.log_softmax(logits / self.temperature, dim=-1)

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """标准 CE loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )

    def _compute_entropy(
        self,
        probs: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """香农熵（在原始概率上计算）"""
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    def _valid_token_mask(
        self,
        labels: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """生成有效 token 的 mask"""
        if labels is None:
            return torch.ones(1, seq_len, dtype=torch.bool, device=device)

        shift_labels = labels[:, 1:]
        valid = (shift_labels != self.ignore_index)

        if valid.shape[1] < seq_len:
            pad = torch.zeros(
                valid.shape[0], seq_len - valid.shape[1],
                dtype=torch.bool, device=device
            )
            valid = torch.cat([valid, pad], dim=1)
        else:
            valid = valid[:, :seq_len]

        return valid


class ForwardKLLoss(DistillationLoss):
    """Forward KL: KL(P_teacher || P_student)"""

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        # ── KL 损失 ──────────────────────────────────────────
        log_p_student_soft = self._get_log_soft_probs(student_logits)
        p_teacher_soft     = self._get_soft_probs(teacher_logits)

        kl_per_token = F.kl_div(
            log_p_student_soft,
            p_teacher_soft,
            reduction="none",
        ).sum(dim=-1)

        mask    = self._valid_token_mask(labels, kl_per_token.shape[1], kl_per_token.device)
        n_valid = mask.float().sum().clamp(min=1.0)
        kl_loss = (kl_per_token * mask.float()).sum() / n_valid
        kl_loss = kl_loss * (self.temperature ** 2)

        # ── CE 损失（仅在 off-policy 时计算）──────────────────
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if not self.on_policy and labels is not None:
            ce_loss = self._compute_ce_loss(student_logits, labels)

        # ── 总损失 ───────────────────────────────────────────
        total_loss = self.alpha * ce_loss + self.beta * kl_loss

        # ── 统计信息 ─────────────────────────────────────────
        with torch.no_grad():
            p_student_original = self._get_original_probs(student_logits)
            p_teacher_original = self._get_original_probs(teacher_logits)
            student_entropy    = self._compute_entropy(p_student_original).mean()
            teacher_entropy    = self._compute_entropy(p_teacher_original).mean()

        stats = {
            "loss/total":       total_loss.item(),
            "loss/ce":          ce_loss.item(),
            "loss/kl":          kl_loss.item(),
            "entropy/student":  student_entropy.item(),
            "entropy/teacher":  teacher_entropy.item(),
        }

        return total_loss, stats


class ReverseKLLoss(DistillationLoss):
    """Reverse KL: KL(P_student || P_teacher)"""

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        # ── KL 损失 ──────────────────────────────────────────
        p_student_soft     = self._get_soft_probs(student_logits)
        log_p_student_soft = self._get_log_soft_probs(student_logits)
        log_p_teacher_soft = self._get_log_soft_probs(teacher_logits)

        kl_per_token = (
            p_student_soft * (log_p_student_soft - log_p_teacher_soft)
        ).sum(dim=-1)

        mask    = self._valid_token_mask(labels, kl_per_token.shape[1], kl_per_token.device)
        n_valid = mask.float().sum().clamp(min=1.0)
        kl_loss = (kl_per_token * mask.float()).sum() / n_valid
        kl_loss = kl_loss * (self.temperature ** 2)

        # ── CE 损失（仅在 off-policy 时计算）──────────────────
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if not self.on_policy and labels is not None:
            ce_loss = self._compute_ce_loss(student_logits, labels)

        # ── 总损失 ───────────────────────────────────────────
        total_loss = self.alpha * ce_loss + self.beta * kl_loss

        # ── 统计信息 ─────────────────────────────────────────
        with torch.no_grad():
            p_student_original = self._get_original_probs(student_logits)
            p_teacher_original = self._get_original_probs(teacher_logits)
            student_entropy    = self._compute_entropy(p_student_original).mean()
            teacher_entropy    = self._compute_entropy(p_teacher_original).mean()

        stats = {
            "loss/total":       total_loss.item(),
            "loss/ce":          ce_loss.item(),
            "loss/kl":          kl_loss.item(),
            "entropy/student":  student_entropy.item(),
            "entropy/teacher":  teacher_entropy.item(),
        }

        return total_loss, stats


class EntropyWeightedJSLoss(DistillationLoss):
    """Entropy-weighted JS 散度"""

    def __init__(
        self,
        temperature:  float = 2.0,
        alpha:        float = 0.5,
        beta:         float = 0.5,
        ignore_index: int   = -100,
        entropy_temp: float = 1.0,
        on_policy:    bool  = False,
    ):
        super().__init__(temperature, alpha, beta, ignore_index, on_policy)
        self.entropy_temp = entropy_temp

    def _exponential_aggregation(
        self,
        p_teacher: torch.Tensor,
        p_student: torch.Tensor,
        eps: float = 1e-10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于原始熵的指数加权聚合"""
        # 注意：这里用温度缩放后的概率计算熵（用于权重计算）
        # 因为我们想根据蒸馏空间的熵来决定权重
        H_teacher = self._compute_entropy(p_teacher, eps)
        H_student = self._compute_entropy(p_student, eps)

        exp_t = torch.exp(-H_teacher / self.entropy_temp)
        exp_s = torch.exp(-H_student / self.entropy_temp)
        Z     = exp_t + exp_s

        w_teacher = (exp_t / Z).unsqueeze(-1)
        w_student = (exp_s / Z).unsqueeze(-1)

        M = w_teacher * p_teacher + w_student * p_student
        M = torch.clamp(M, min=eps)
        M = M / M.sum(dim=-1, keepdim=True)

        return M, w_teacher, w_student

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        # ── JS 损失 ──────────────────────────────────────────
        p_teacher_soft = self._get_soft_probs(teacher_logits)
        p_student_soft = self._get_soft_probs(student_logits)

        M, w_teacher, w_student = self._exponential_aggregation(
            p_teacher_soft, p_student_soft
        )
        log_M = torch.log(M + 1e-10)

        kl_t_M = (p_teacher_soft * (torch.log(p_teacher_soft + 1e-10) - log_M)).sum(dim=-1)
        kl_s_M = (p_student_soft * (torch.log(p_student_soft + 1e-10) - log_M)).sum(dim=-1)

        js_per_token = 0.5 * (kl_t_M + kl_s_M)

        mask    = self._valid_token_mask(labels, js_per_token.shape[1], js_per_token.device)
        n_valid = mask.float().sum().clamp(min=1.0)
        js_loss = (js_per_token * mask.float()).sum() / n_valid
        js_loss = js_loss * (self.temperature ** 2)

        # ── CE 损失（仅在 off-policy 时计算）──────────────────
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if not self.on_policy and labels is not None:
            ce_loss = self._compute_ce_loss(student_logits, labels)

        # ── 总损失 ───────────────────────────────────────────
        total_loss = self.alpha * ce_loss + self.beta * js_loss

        # ── 统计信息 ─────────────────────────────────────────
        with torch.no_grad():
            p_student_original = self._get_original_probs(student_logits)
            p_teacher_original = self._get_original_probs(teacher_logits)
            student_entropy    = self._compute_entropy(p_student_original).mean()
            teacher_entropy    = self._compute_entropy(p_teacher_original).mean()
            M_entropy          = self._compute_entropy(M).mean()
            avg_weight_teacher = w_teacher.mean()

        stats = {
            "loss/total":       total_loss.item(),
            "loss/ce":          ce_loss.item(),
            "loss/js":          js_loss.item(),
            "entropy/student":  student_entropy.item(),
            "entropy/teacher":  teacher_entropy.item(),
            "entropy/mixture":  M_entropy.item(),
            "weight/teacher":   avg_weight_teacher.item(),
            "weight/student":   (1.0 - avg_weight_teacher).item(),
        }

        return total_loss, stats


def get_distillation_loss(
    mode:         str   = "forward_kl",
    temperature:  float = 2.0,
    alpha:        float = 0.5,
    beta:         float = 0.5,
    ignore_index: int   = -100,
    entropy_temp: float = 1.0,
    on_policy:    bool  = False,
) -> DistillationLoss:
    """
    工厂函数
    
    on_policy=True:  只用 KL/JS loss（CE 不计算）
    on_policy=False: alpha * CE + beta * KL/JS
    """
    if mode == "forward_kl":
        return ForwardKLLoss(temperature, alpha, beta, ignore_index, on_policy)
    elif mode == "reverse_kl":
        return ReverseKLLoss(temperature, alpha, beta, ignore_index, on_policy)
    elif mode == "entropy_js":
        return EntropyWeightedJSLoss(
            temperature, alpha, beta, ignore_index, entropy_temp, on_policy
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
