"""
Stability Metrics
=================
Metrik untuk mengukur stabilitas training.
"""

import numpy as np
from typing import Dict, List, Any
from collections import deque


class StabilityMetrics:
    """
    Track dan compute metrik stabilitas training.
    
    Metrics:
    - Gradient stability (variance of gradient norms)
    - Loss stability (variance of losses)
    - Policy stability (KL divergence tracking)
    - Entropy stability
    """
    
    def __init__(self, window_size: int = 100):
        """
        Inisialisasi stability metrics.
        
        Args:
            window_size: Window size untuk moving statistics
        """
        self.window_size = window_size
        
        # History storage
        self.gradient_norms = deque(maxlen=window_size)
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.entropies = deque(maxlen=window_size)
        self.kl_divergences = deque(maxlen=window_size)
        
        # Threshold
        self.grad_threshold = 100.0
        self.loss_variance_threshold = 1.0
        self.kl_threshold = 0.1
    
    def update(
        self,
        grad_norm: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl_div: float
    ):
        """Update metrics dengan data baru."""
        self.gradient_norms.append(grad_norm)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.kl_divergences.append(kl_div)
    
    def compute_stability_score(self) -> float:
        """
        Compute overall stability score (0-1).
        Higher is more stable.
        """
        if len(self.gradient_norms) < 10:
            return 1.0  # Not enough data
        
        scores = []
        
        # Gradient stability
        grad_var = np.var(list(self.gradient_norms))
        grad_score = 1.0 / (1.0 + grad_var)
        scores.append(grad_score)
        
        # Loss stability
        loss_var = np.var(list(self.policy_losses))
        loss_score = 1.0 / (1.0 + loss_var)
        scores.append(loss_score)
        
        # KL stability
        kl_mean = np.mean(list(self.kl_divergences))
        kl_score = 1.0 if kl_mean < self.kl_threshold else self.kl_threshold / kl_mean
        scores.append(kl_score)
        
        # Entropy stability (should not collapse)
        entropy_mean = np.mean(list(self.entropies))
        entropy_score = min(1.0, entropy_mean / 0.1) if entropy_mean > 0 else 0
        scores.append(entropy_score)
        
        return np.mean(scores)
    
    def check_instability(self) -> Dict[str, Any]:
        """
        Check untuk signs of instability.
        
        Returns:
            Dict dengan warning flags
        """
        warnings = {
            'is_stable': True,
            'warnings': []
        }
        
        if len(self.gradient_norms) < 5:
            return warnings
        
        # Check gradient explosion
        recent_grads = list(self.gradient_norms)[-5:]
        if max(recent_grads) > self.grad_threshold:
            warnings['is_stable'] = False
            warnings['warnings'].append('gradient_explosion')
        
        # Check gradient vanishing
        if max(recent_grads) < 1e-7:
            warnings['is_stable'] = False
            warnings['warnings'].append('gradient_vanishing')
        
        # Check loss spike
        if len(self.policy_losses) > 10:
            recent_losses = list(self.policy_losses)[-5:]
            mean_loss = np.mean(list(self.policy_losses)[:-5])
            if max(recent_losses) > mean_loss * 3:
                warnings['is_stable'] = False
                warnings['warnings'].append('loss_spike')
        
        # Check entropy collapse
        if len(self.entropies) > 10:
            recent_entropy = np.mean(list(self.entropies)[-5:])
            if recent_entropy < 0.01:
                warnings['is_stable'] = False
                warnings['warnings'].append('entropy_collapse')
        
        # Check KL explosion
        if len(self.kl_divergences) > 5:
            recent_kl = np.mean(list(self.kl_divergences)[-5:])
            if recent_kl > self.kl_threshold * 2:
                warnings['is_stable'] = False
                warnings['warnings'].append('kl_explosion')
        
        return warnings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'stability_score': self.compute_stability_score()
        }
        
        if self.gradient_norms:
            stats['gradient_mean'] = np.mean(list(self.gradient_norms))
            stats['gradient_std'] = np.std(list(self.gradient_norms))
            stats['gradient_max'] = np.max(list(self.gradient_norms))
        
        if self.policy_losses:
            stats['policy_loss_mean'] = np.mean(list(self.policy_losses))
            stats['policy_loss_std'] = np.std(list(self.policy_losses))
        
        if self.value_losses:
            stats['value_loss_mean'] = np.mean(list(self.value_losses))
        
        if self.entropies:
            stats['entropy_mean'] = np.mean(list(self.entropies))
            stats['entropy_current'] = list(self.entropies)[-1]
        
        if self.kl_divergences:
            stats['kl_mean'] = np.mean(list(self.kl_divergences))
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.gradient_norms.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropies.clear()
        self.kl_divergences.clear()
