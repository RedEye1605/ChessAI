"""
Gradient Utilities
==================
Utilitas untuk monitoring dan manipulasi gradients.
Penting untuk stabilitas training dalam reinforcement learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from collections import deque


def compute_gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0
) -> float:
    """
    Hitung total gradient norm dari semua parameters model.
    
    Args:
        model: PyTorch model
        norm_type: Tipe norm (default L2)
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def compute_gradient_norms_per_layer(
    model: nn.Module,
    norm_type: float = 2.0
) -> Dict[str, float]:
    """
    Hitung gradient norm per layer.
    
    Args:
        model: PyTorch model
        norm_type: Tipe norm
        
    Returns:
        Dictionary dengan nama layer dan gradient norm
    """
    norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(norm_type).item()
    
    return norms


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    clip_type: str = 'global_norm',
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients dengan berbagai strategi.
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm untuk clipping
        clip_type: Tipe clipping ('global_norm', 'per_param', 'value')
        norm_type: Tipe norm untuk computation
        
    Returns:
        Gradient norm sebelum clipping
    """
    if clip_type == 'global_norm':
        # Standard gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm, 
            norm_type=norm_type
        )
        return total_norm.item() if torch.is_tensor(total_norm) else total_norm
    
    elif clip_type == 'per_param':
        # Clip setiap parameter secara terpisah
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                
                if param_norm > max_norm:
                    param.grad.data.mul_(max_norm / param_norm)
        
        return total_norm ** (1. / norm_type)
    
    elif clip_type == 'value':
        # Clip berdasarkan nilai absolut
        total_norm = compute_gradient_norm(model, norm_type)
        torch.nn.utils.clip_grad_value_(model.parameters(), max_norm)
        return total_norm
    
    else:
        raise ValueError(f"Unknown clip_type: {clip_type}")


def check_gradients_health(
    model: nn.Module,
    warn_threshold: float = 100.0,
    nan_check: bool = True
) -> Dict[str, any]:
    """
    Check kesehatan gradients.
    
    Args:
        model: PyTorch model
        warn_threshold: Threshold untuk warning
        nan_check: Apakah check untuk NaN/Inf
        
    Returns:
        Dictionary dengan status kesehatan
    """
    status = {
        'healthy': True,
        'has_nan': False,
        'has_inf': False,
        'max_norm': 0.0,
        'mean_norm': 0.0,
        'problematic_layers': []
    }
    
    norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            
            # Check NaN
            if nan_check and torch.isnan(grad).any():
                status['has_nan'] = True
                status['healthy'] = False
                status['problematic_layers'].append((name, 'nan'))
            
            # Check Inf
            if nan_check and torch.isinf(grad).any():
                status['has_inf'] = True
                status['healthy'] = False
                status['problematic_layers'].append((name, 'inf'))
            
            # Compute norm
            norm = grad.norm().item()
            norms.append(norm)
            
            if norm > warn_threshold:
                status['problematic_layers'].append((name, f'high_norm:{norm:.2f}'))
    
    if norms:
        status['max_norm'] = max(norms)
        status['mean_norm'] = sum(norms) / len(norms)
        
        if status['max_norm'] > warn_threshold:
            status['healthy'] = False
    
    return status


class GradientMonitor:
    """
    Monitor gradient statistics selama training.
    
    Melacak:
    - Gradient norms over time
    - Per-layer statistics
    - Anomaly detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
        warn_threshold: float = 100.0,
        alert_on_anomaly: bool = True
    ):
        """
        Inisialisasi gradient monitor.
        
        Args:
            model: PyTorch model untuk monitoring
            window_size: Ukuran window untuk moving average
            warn_threshold: Threshold untuk warning
            alert_on_anomaly: Print alert jika ada anomaly
        """
        self.model = model
        self.window_size = window_size
        self.warn_threshold = warn_threshold
        self.alert_on_anomaly = alert_on_anomaly
        
        # History storage
        self.norm_history: deque = deque(maxlen=window_size)
        self.layer_norms_history: Dict[str, deque] = {}
        
        # Statistics
        self.total_steps = 0
        self.anomaly_count = 0
        self.max_observed_norm = 0.0
        
        # Initialize layer tracking
        for name, _ in model.named_parameters():
            self.layer_norms_history[name] = deque(maxlen=window_size)
    
    def update(self) -> Dict[str, any]:
        """
        Update monitor dengan current gradients.
        
        Returns:
            Current gradient statistics
        """
        self.total_steps += 1
        
        # Compute norms
        total_norm = compute_gradient_norm(self.model)
        layer_norms = compute_gradient_norms_per_layer(self.model)
        
        # Update history
        self.norm_history.append(total_norm)
        for name, norm in layer_norms.items():
            if name in self.layer_norms_history:
                self.layer_norms_history[name].append(norm)
        
        # Update max
        self.max_observed_norm = max(self.max_observed_norm, total_norm)
        
        # Check for anomalies
        anomaly = self._check_anomaly(total_norm)
        if anomaly and self.alert_on_anomaly:
            self.anomaly_count += 1
            print(f"⚠️ Gradient anomaly detected at step {self.total_steps}: norm={total_norm:.4f}")
        
        return {
            'norm': total_norm,
            'is_anomaly': anomaly,
            'mean_norm': np.mean(list(self.norm_history)) if self.norm_history else 0,
            'std_norm': np.std(list(self.norm_history)) if len(self.norm_history) > 1 else 0
        }
    
    def _check_anomaly(self, current_norm: float) -> bool:
        """Check apakah current norm adalah anomaly."""
        if current_norm > self.warn_threshold:
            return True
        
        if len(self.norm_history) < 10:
            return False
        
        mean = np.mean(list(self.norm_history))
        std = np.std(list(self.norm_history))
        
        # Z-score based anomaly detection
        if std > 0:
            z_score = abs(current_norm - mean) / std
            if z_score > 3:  # 3 sigma rule
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive gradient statistics."""
        if not self.norm_history:
            return {'total_steps': 0}
        
        norms = list(self.norm_history)
        
        return {
            'total_steps': self.total_steps,
            'current_norm': norms[-1] if norms else 0,
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'min_norm': np.min(norms),
            'max_norm': np.max(norms),
            'max_observed': self.max_observed_norm,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(1, self.total_steps)
        }
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get per-layer gradient statistics."""
        stats = {}
        
        for name, history in self.layer_norms_history.items():
            if history:
                norms = list(history)
                stats[name] = {
                    'mean': np.mean(norms),
                    'std': np.std(norms),
                    'max': np.max(norms),
                    'current': norms[-1]
                }
        
        return stats
    
    def reset(self):
        """Reset monitor statistics."""
        self.norm_history.clear()
        for history in self.layer_norms_history.values():
            history.clear()
        self.total_steps = 0
        self.anomaly_count = 0
        self.max_observed_norm = 0.0


class AdaptiveGradientClipper:
    """
    Adaptive gradient clipping berdasarkan gradient history.
    
    Menyesuaikan clip threshold secara dinamis berdasarkan
    observed gradient statistics.
    """
    
    def __init__(
        self,
        initial_clip_norm: float = 1.0,
        clip_factor: float = 1.5,
        history_size: int = 100,
        min_clip_norm: float = 0.1,
        max_clip_norm: float = 10.0
    ):
        """
        Inisialisasi adaptive clipper.
        
        Args:
            initial_clip_norm: Initial clip threshold
            clip_factor: Multiplier untuk adaptive threshold
            history_size: Ukuran history untuk statistics
            min_clip_norm: Minimum clip threshold
            max_clip_norm: Maximum clip threshold
        """
        self.clip_norm = initial_clip_norm
        self.clip_factor = clip_factor
        self.min_clip_norm = min_clip_norm
        self.max_clip_norm = max_clip_norm
        
        self.norm_history: deque = deque(maxlen=history_size)
        self.clipped_count = 0
        self.total_count = 0
    
    def clip(self, model: nn.Module) -> Tuple[float, bool]:
        """
        Clip gradients dengan adaptive threshold.
        
        Args:
            model: PyTorch model
            
        Returns:
            (original_norm, was_clipped)
        """
        self.total_count += 1
        
        # Compute current norm
        current_norm = compute_gradient_norm(model)
        self.norm_history.append(current_norm)
        
        # Check if clipping needed
        was_clipped = current_norm > self.clip_norm
        
        if was_clipped:
            self.clipped_count += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        
        # Update adaptive threshold
        self._update_threshold()
        
        return current_norm, was_clipped
    
    def _update_threshold(self):
        """Update clip threshold berdasarkan history."""
        if len(self.norm_history) < 10:
            return
        
        norms = list(self.norm_history)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Set threshold ke mean + factor * std
        new_threshold = mean_norm + self.clip_factor * std_norm
        
        # Clamp ke range yang valid
        self.clip_norm = np.clip(
            new_threshold,
            self.min_clip_norm,
            self.max_clip_norm
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get clipping statistics."""
        return {
            'current_threshold': self.clip_norm,
            'clip_rate': self.clipped_count / max(1, self.total_count),
            'total_clips': self.clipped_count,
            'mean_norm': np.mean(list(self.norm_history)) if self.norm_history else 0
        }
