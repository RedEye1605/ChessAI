"""
Adaptive Optimizer
==================
Implementasi wrapper optimizer dengan fitur adaptif untuk stabilitas training.
Mengkombinasikan berbagai teknik seperti:
- Gradient clipping
- Learning rate warmup
- Gradient smoothing
- Automatic mixed precision support
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD
from typing import Optional, Dict, Any, Callable, List, Iterator
import math

from .gradient_utils import compute_gradient_norm, clip_gradients, AdaptiveGradientClipper
from .lr_scheduler import WarmupCosineScheduler, AdaptiveLRScheduler


class AdaptiveOptimizer:
    """
    Wrapper optimizer dengan fitur adaptif untuk training yang stabil.
    
    Fitur:
    - Gradient clipping (fixed atau adaptive)
    - Learning rate scheduling dengan warmup
    - Gradient accumulation
    - Gradient smoothing (EMA)
    - Automatic mixed precision (AMP) support
    - Training statistics tracking
    
    Usage:
        optimizer = AdaptiveOptimizer(model, config)
        
        for batch in dataloader:
            loss = compute_loss(batch)
            stats = optimizer.step(loss)
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        optimizer_type: str = 'adamw',
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        # Gradient clipping
        max_grad_norm: float = 1.0,
        gradient_clipping: str = 'global_norm',  # global_norm, per_param, adaptive, none
        # Learning rate scheduling
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        lr_scheduler: str = 'cosine',  # cosine, linear, adaptive, none
        # Gradient accumulation
        accumulation_steps: int = 1,
        # Gradient smoothing
        use_gradient_smoothing: bool = False,
        smoothing_beta: float = 0.9,
        # AMP
        use_amp: bool = False,
        # Additional options
        log_frequency: int = 100
    ):
        """
        Inisialisasi Adaptive Optimizer.
        
        Args:
            model: PyTorch model untuk optimize
            learning_rate: Base learning rate
            optimizer_type: Tipe optimizer ('adam', 'adamw', 'sgd')
            weight_decay: Weight decay coefficient
            betas: Adam betas
            eps: Adam epsilon
            max_grad_norm: Maximum gradient norm untuk clipping
            gradient_clipping: Tipe gradient clipping
            warmup_steps: Steps untuk LR warmup
            total_steps: Total training steps
            min_lr: Minimum learning rate
            lr_scheduler: Tipe LR scheduler
            accumulation_steps: Gradient accumulation steps
            use_gradient_smoothing: Enable gradient EMA
            smoothing_beta: Beta untuk gradient smoothing
            use_amp: Enable automatic mixed precision
            log_frequency: Frequency untuk logging statistics
        """
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps
        self.use_gradient_smoothing = use_gradient_smoothing
        self.smoothing_beta = smoothing_beta
        self.use_amp = use_amp
        self.log_frequency = log_frequency
        
        # Create base optimizer
        self.optimizer = self._create_optimizer(
            optimizer_type, learning_rate, weight_decay, betas, eps
        )
        
        # Create LR scheduler
        self.scheduler = self._create_scheduler(
            lr_scheduler, warmup_steps, total_steps, min_lr
        )
        
        # Adaptive gradient clipper
        if gradient_clipping == 'adaptive':
            self.adaptive_clipper = AdaptiveGradientClipper(
                initial_clip_norm=max_grad_norm
            )
        else:
            self.adaptive_clipper = None
        
        # AMP scaler
        if use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
        
        # Gradient smoothing (EMA of gradients)
        if use_gradient_smoothing:
            self.gradient_ema: Dict[str, torch.Tensor] = {}
        
        # Statistics tracking
        self.step_count = 0
        self.accumulation_count = 0
        self.statistics = {
            'grad_norm_history': [],
            'lr_history': [],
            'loss_history': [],
            'clipped_steps': 0
        }
    
    def _create_optimizer(
        self,
        optimizer_type: str,
        lr: float,
        weight_decay: float,
        betas: tuple,
        eps: float
    ) -> Optimizer:
        """Create base optimizer."""
        params = self.model.parameters()
        
        if optimizer_type == 'adam':
            return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return SGD(params, lr=lr, momentum=betas[0], weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(
        self,
        scheduler_type: str,
        warmup_steps: int,
        total_steps: int,
        min_lr: float
    ):
        """Create LR scheduler."""
        if scheduler_type == 'none':
            return None
        elif scheduler_type == 'cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr
            )
        elif scheduler_type == 'adaptive':
            return AdaptiveLRScheduler(
                self.optimizer,
                initial_lr=self.learning_rate,
                min_lr=min_lr
            )
        else:
            # Default to cosine
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr
            )
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass dengan optional AMP scaling.
        
        Args:
            loss: Loss tensor
        """
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, loss: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform optimizer step dengan semua enhancements.
        
        Args:
            loss: Optional loss untuk backward (jika belum dilakukan)
            
        Returns:
            Dictionary dengan training statistics
        """
        stats = {
            'step': self.step_count,
            'lr': self._get_current_lr(),
            'grad_norm': 0.0,
            'clipped': False
        }
        
        # Backward jika loss provided
        if loss is not None:
            self.backward(loss)
            stats['loss'] = loss.item()
            self.statistics['loss_history'].append(loss.item())
        
        # Gradient accumulation
        self.accumulation_count += 1
        if self.accumulation_count < self.accumulation_steps:
            return stats
        
        self.accumulation_count = 0
        
        # Unscale gradients untuk AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient smoothing
        if self.use_gradient_smoothing:
            self._apply_gradient_smoothing()
        
        # Compute gradient norm
        grad_norm = compute_gradient_norm(self.model)
        stats['grad_norm'] = grad_norm
        self.statistics['grad_norm_history'].append(grad_norm)
        
        # Gradient clipping
        if self.gradient_clipping != 'none':
            clipped = self._clip_gradients()
            stats['clipped'] = clipped
            if clipped:
                self.statistics['clipped_steps'] += 1
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.zero_grad()
        
        # LR scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, AdaptiveLRScheduler):
                if loss is not None:
                    self.scheduler.step(loss.item())
            else:
                self.scheduler.step()
        
        self.step_count += 1
        stats['lr'] = self._get_current_lr()
        self.statistics['lr_history'].append(stats['lr'])
        
        return stats
    
    def _clip_gradients(self) -> bool:
        """Clip gradients dan return apakah clipping terjadi."""
        if self.gradient_clipping == 'adaptive':
            _, clipped = self.adaptive_clipper.clip(self.model)
            return clipped
        elif self.gradient_clipping in ['global_norm', 'per_param', 'value']:
            original_norm = compute_gradient_norm(self.model)
            clip_gradients(self.model, self.max_grad_norm, self.gradient_clipping)
            return original_norm > self.max_grad_norm
        return False
    
    def _apply_gradient_smoothing(self):
        """Apply EMA smoothing ke gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            if name not in self.gradient_ema:
                self.gradient_ema[name] = param.grad.clone()
            else:
                self.gradient_ema[name].mul_(self.smoothing_beta).add_(
                    param.grad, alpha=1 - self.smoothing_beta
                )
                param.grad.copy_(self.gradient_ema[name])
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'total_steps': self.step_count,
            'current_lr': self._get_current_lr(),
            'clipped_ratio': self.statistics['clipped_steps'] / max(1, self.step_count)
        }
        
        if self.statistics['grad_norm_history']:
            recent_norms = self.statistics['grad_norm_history'][-100:]
            stats['mean_grad_norm'] = sum(recent_norms) / len(recent_norms)
            stats['max_grad_norm'] = max(recent_norms)
        
        if self.statistics['loss_history']:
            recent_losses = self.statistics['loss_history'][-100:]
            stats['mean_loss'] = sum(recent_losses) / len(recent_losses)
        
        if self.adaptive_clipper:
            stats['adaptive_clip_threshold'] = self.adaptive_clipper.clip_norm
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict untuk checkpointing."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'statistics': self.statistics
        }
        
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            state['scheduler'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict untuk resume training."""
        self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state.get('step_count', 0)
        self.statistics = state.get('statistics', self.statistics)
        
        if 'scheduler' in state and self.scheduler is not None:
            if hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(state['scheduler'])
        
        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> AdaptiveOptimizer:
    """
    Factory function untuk membuat AdaptiveOptimizer dari config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        AdaptiveOptimizer instance
    """
    return AdaptiveOptimizer(
        model=model,
        learning_rate=config.get('learning_rate', 3e-4),
        optimizer_type=config.get('optimizer_type', 'adamw'),
        weight_decay=config.get('weight_decay', 0.01),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        gradient_clipping=config.get('gradient_clipping', 'global_norm'),
        warmup_steps=config.get('warmup_steps', 1000),
        total_steps=config.get('total_steps', 100000),
        min_lr=config.get('min_lr', 1e-6),
        lr_scheduler=config.get('lr_scheduler', 'cosine'),
        accumulation_steps=config.get('accumulation_steps', 1),
        use_gradient_smoothing=config.get('use_gradient_smoothing', False),
        use_amp=config.get('use_amp', False)
    )
