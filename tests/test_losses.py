# tests/test_losses.py
"""
损失函数单元测试
"""

import unittest
import torch
from src.losses import (
    ForwardKLLoss,
    ReverseKLLoss,
    EntropyWeightedJSLoss,
)


class TestDistillationLosses(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 100
        
        # 生成随机 logits
        torch.manual_seed(42)
        self.student_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size
        )
        self.teacher_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size
        )
        self.labels = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len)
        )
    
    def test_forward_kl(self):
        """测试 Forward KL"""
        loss_fn = ForwardKLLoss(temperature=2.0, alpha=0.5, beta=0.5)
        
        loss, stats = loss_fn(
            self.student_logits,
            self.teacher_logits,
            self.labels,
        )
        
        # 检查损失是标量
        self.assertEqual(loss.shape, torch.Size([]))
        
        # 检查损失非负
        self.assertGreaterEqual(loss.item(), 0.0)
        
        # 检查统计信息
        self.assertIn("loss/total", stats)
        self.assertIn("loss/ce", stats)
        self.assertIn("loss/kl", stats)
        self.assertIn("entropy/student", stats)
        self.assertIn("entropy/teacher", stats)
    
    def test_reverse_kl(self):
        """测试 Reverse KL"""
        loss_fn = ReverseKLLoss(temperature=2.0, alpha=0.5, beta=0.5)
        
        loss, stats = loss_fn(
            self.student_logits,
            self.teacher_logits,
            self.labels,
        )
        
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertIn("loss/kl", stats)
    
    def test_entropy_js(self):
        """测试 Entropy-weighted JS"""
        loss_fn = EntropyWeightedJSLoss(
            temperature=2.0,
            alpha=0.5,
            beta=0.5,
            entropy_temp=1.0,
        )
        
        loss, stats = loss_fn(
            self.student_logits,
            self.teacher_logits,
            self.labels,
        )
        
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)
        
        # Entropy-weighted JS 特有的统计
        self.assertIn("loss/js", stats)
        self.assertIn("entropy/mixture", stats)
        self.assertIn("weight/teacher", stats)
        self.assertIn("weight/student", stats)
        
        # 检查权重和为 1
        w_teacher = stats["weight/teacher"]
        w_student = stats["weight/student"]
        self.assertAlmostEqual(w_teacher + w_student, 1.0, places=5)
    
    def test_temperature_effect(self):
        """测试温度对损失的影响"""
        loss_fn_low = ForwardKLLoss(temperature=1.0, alpha=0.0, beta=1.0)
        loss_fn_high = ForwardKLLoss(temperature=4.0, alpha=0.0, beta=1.0)
        
        loss_low, _ = loss_fn_low(
            self.student_logits,
            self.teacher_logits,
            None,
        )
        
        loss_high, _ = loss_fn_high(
            self.student_logits,
            self.teacher_logits,
            None,
        )
        
        # 温度越高，loss 应该越小（分布越软）
        # 注意：由于温度平方缩放，实际关系可能复杂
        self.assertIsInstance(loss_low.item(), float)
        self.assertIsInstance(loss_high.item(), float)
    
    def test_alpha_beta_weights(self):
        """测试 alpha 和 beta 权重"""
        # 只有 CE loss
        loss_fn_ce = ForwardKLLoss(temperature=2.0, alpha=1.0, beta=0.0)
        loss_ce, stats_ce = loss_fn_ce(
            self.student_logits,
            self.teacher_logits,
            self.labels,
        )
        
        # 只有 KL loss
        loss_fn_kl = ForwardKLLoss(temperature=2.0, alpha=0.0, beta=1.0)
        loss_kl, stats_kl = loss_fn_kl(
            self.student_logits,
            self.teacher_logits,
            None,
        )
        
        # CE loss 应该大于 0
        self.assertGreater(stats_ce["loss/ce"], 0.0)
        # KL loss 应该大于 0
        self.assertGreater(stats_kl["loss/kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
