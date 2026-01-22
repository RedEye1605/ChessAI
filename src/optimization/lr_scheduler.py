"""
Learning Rate Schedulers
========================
Implementasi berbagai learning rate schedulers untuk training yang stabil.
Termasuk warmup, cosine annealing, dan adaptive scheduling.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List, Dict, Any
import math
import numpy as np


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning Rate Scheduler dengan Linear Warmup + Cosine Annealing.
    
    LR Schedule:
    1. Warmup Phase: LR naik linear dari 0 ke base_lr
    2. Cosine Phase: LR turun dengan cosine curve ke min_lr
    
    Formula:
        warmup: lr = base_lr * (step / warmup_steps)
        cosine: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Inisialisasi scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Jumlah steps untuk warmup
            total_steps: Total training steps
            min_lr: Minimum learning rate setelah decay
            last_epoch: Step terakhir (untuk resume training)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates untuk current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear Warmup + Linear Decay Scheduler.
    
    Lebih simple dari cosine, tetapi efektif untuk banyak kasus.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """Inisialisasi linear scheduler."""
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Warmup
            factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            factor = 1.0 - progress
            
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


class CyclicLRWithWarmup(_LRScheduler):
    """
    Cyclic Learning Rate dengan Warmup.
    
    LR oscillates dalam range [min_lr, max_lr] setelah warmup.
    Berguna untuk escaping local minima dan exploring loss landscape.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        cycle_length: int,
        min_lr: float = 1e-5,
        max_lr: Optional[float] = None,
        gamma: float = 0.99,  # Decay factor per cycle
        last_epoch: int = -1
    ):
        """
        Inisialisasi cyclic scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Warmup steps
            cycle_length: Panjang satu cycle
            min_lr: Minimum LR
            max_lr: Maximum LR (default: base_lr)
            gamma: Decay factor per cycle
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.min_lr = min_lr
        self.max_lr_factor = max_lr  # Will be set in super().__init__
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
        # Set max_lr after base_lrs is available
        if max_lr is None:
            self.max_lrs = self.base_lrs
        else:
            self.max_lrs = [max_lr] * len(self.base_lrs)
    
    def get_lr(self) -> List[float]:
        """Compute cyclic learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Warmup
            factor = self.last_epoch / max(1, self.warmup_steps)
            return [lr * factor for lr in self.max_lrs]
        
        # Cyclic phase
        adjusted_step = self.last_epoch - self.warmup_steps
        cycle_num = adjusted_step // self.cycle_length
        cycle_pos = adjusted_step % self.cycle_length
        
        # Decay max_lr per cycle
        decay = self.gamma ** cycle_num
        
        # Cosine within cycle
        progress = cycle_pos / self.cycle_length
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (max_lr * decay - self.min_lr) * cosine_factor
            for max_lr in self.max_lrs
        ]


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler.
    
    Menyesuaikan LR secara dinamis berdasarkan training metrics:
    - Decrease LR jika loss tidak improve
    - Increase LR jika gradients terlalu kecil
    - Warmup restart jika training stagnant
    
    Note: Ini bukan subclass dari _LRScheduler karena need more control.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        patience: int = 10,
        factor_down: float = 0.5,
        factor_up: float = 1.2,
        threshold: float = 1e-4,
        cooldown: int = 5
    ):
        """
        Inisialisasi adaptive scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
            min_lr: Minimum LR
            max_lr: Maximum LR
            patience: Steps tanpa improvement sebelum reduce
            factor_down: Factor untuk reduce LR
            factor_up: Factor untuk increase LR
            threshold: Minimum improvement untuk dianggap progress
            cooldown: Steps cooldown setelah LR change
        """
        self.optimizer = optimizer
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor_down = factor_down
        self.factor_up = factor_up
        self.threshold = threshold
        self.cooldown = cooldown
        
        # State
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.cooldown_counter = 0
        self.lr_history: List[float] = [initial_lr]
        self.loss_history: List[float] = []
        
        # Set initial LR
        self._set_lr(initial_lr)
    
    def _set_lr(self, lr: float):
        """Set learning rate di optimizer."""
        lr = max(self.min_lr, min(self.max_lr, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr
    
    def step(self, loss: float) -> Dict[str, any]:
        """
        Update scheduler dengan current loss.
        
        Args:
            loss: Current training loss
            
        Returns:
            Dict dengan info tentang LR adjustment
        """
        self.loss_history.append(loss)
        
        info = {
            'lr': self.current_lr,
            'adjusted': False,
            'reason': None
        }
        
        # Check cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return info
        
        # Check improvement
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Reduce LR jika no improvement
        if self.steps_without_improvement >= self.patience:
            new_lr = self.current_lr * self.factor_down
            if new_lr >= self.min_lr:
                self._set_lr(new_lr)
                self.cooldown_counter = self.cooldown
                self.steps_without_improvement = 0
                info['adjusted'] = True
                info['reason'] = 'no_improvement'
        
        self.lr_history.append(self.current_lr)
        info['lr'] = self.current_lr
        
        return info
    
    def step_on_gradient(self, grad_norm: float, low_threshold: float = 1e-6) -> Dict[str, any]:
        """
        Adjust LR berdasarkan gradient norm.
        
        Args:
            grad_norm: Current gradient norm
            low_threshold: Threshold untuk "too small" gradients
            
        Returns:
            Adjustment info
        """
        info = {
            'lr': self.current_lr,
            'adjusted': False,
            'reason': None
        }
        
        if self.cooldown_counter > 0:
            return info
        
        # Increase LR jika gradients terlalu kecil
        if grad_norm < low_threshold:
            new_lr = min(self.current_lr * self.factor_up, self.max_lr)
            if new_lr > self.current_lr:
                self._set_lr(new_lr)
                self.cooldown_counter = self.cooldown
                info['adjusted'] = True
                info['reason'] = 'small_gradients'
        
        info['lr'] = self.current_lr
        return info
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def get_statistics(self) -> Dict[str, any]:
        """Get scheduler statistics."""
        return {
            'current_lr': self.current_lr,
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'lr_changes': len([i for i in range(1, len(self.lr_history)) 
                             if self.lr_history[i] != self.lr_history[i-1]])
        }


class OneCycleLR(_LRScheduler):
    """
    One Cycle Learning Rate Policy.
    
    LR schedule:
    1. Warmup: Increase dari min_lr ke max_lr
    2. Annealing: Decrease dari max_lr ke min_lr (bahkan lebih rendah)
    
    Reference: Smith & Topin, "Super-Convergence", 2018
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        Inisialisasi One Cycle scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            max_lr: Maximum learning rate
            total_steps: Total training steps
            pct_start: Percentage of steps untuk warmup
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = initial_lr / final_div_factor
            last_epoch: Last epoch
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.warmup_steps = int(total_steps * pct_start)
        self.annealing_steps = total_steps - self.warmup_steps
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            progress = self.last_epoch / max(1, self.warmup_steps)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.annealing_steps)
            progress = min(progress, 1.0)
            
            # Cosine annealing ke final_lr
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.final_lr + (self.max_lr - self.final_lr) * cosine_factor
        
        return [lr] * len(self.base_lrs)


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    **kwargs
) -> _LRScheduler:
    """
    Factory function untuk membuat scheduler.
    
    Args:
        scheduler_type: Tipe scheduler
        optimizer: PyTorch optimizer
        **kwargs: Additional arguments
        
    Returns:
        LR Scheduler
    """
    schedulers = {
        'cosine': WarmupCosineScheduler,
        'linear': WarmupLinearScheduler,
        'cyclic': CyclicLRWithWarmup,
        'onecycle': OneCycleLR
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. "
                        f"Available: {list(schedulers.keys())}")
    
    return schedulers[scheduler_type](optimizer, **kwargs)
