"""
Training Callbacks
==================
Callbacks untuk monitoring dan adjustment selama training.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np


class GradientMonitorCallback:
    """
    Callback untuk monitoring gradient health.
    """
    
    def __init__(
        self,
        warn_threshold: float = 100.0,
        alert_callback: Optional[Callable] = None
    ):
        """
        Inisialisasi callback.
        
        Args:
            warn_threshold: Threshold untuk warning
            alert_callback: Optional callback untuk alerts
        """
        self.warn_threshold = warn_threshold
        self.alert_callback = alert_callback
        
        self.anomaly_count = 0
        self.total_steps = 0
    
    def __call__(self, step: int, grad_norm: float, **kwargs) -> Dict[str, Any]:
        """
        Called setiap training step.
        
        Args:
            step: Current step
            grad_norm: Gradient norm
            
        Returns:
            Info dict
        """
        self.total_steps += 1
        
        is_anomaly = grad_norm > self.warn_threshold or np.isnan(grad_norm)
        
        if is_anomaly:
            self.anomaly_count += 1
            
            if self.alert_callback:
                self.alert_callback({
                    'type': 'gradient_anomaly',
                    'step': step,
                    'grad_norm': grad_norm
                })
            
            print(f"⚠️ Gradient anomaly at step {step}: norm={grad_norm:.2f}")
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_rate': self.anomaly_count / max(1, self.total_steps)
        }


class StabilityAlertCallback:
    """
    Callback untuk stability alerts.
    """
    
    def __init__(
        self,
        loss_spike_threshold: float = 3.0,
        entropy_min_threshold: float = 0.01,
        kl_max_threshold: float = 0.1
    ):
        """
        Inisialisasi callback.
        
        Args:
            loss_spike_threshold: Multiplier untuk detect loss spikes
            entropy_min_threshold: Minimum entropy sebelum alert
            kl_max_threshold: Maximum KL sebelum alert
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.entropy_min_threshold = entropy_min_threshold
        self.kl_max_threshold = kl_max_threshold
        
        self.loss_history = []
        self.alerts = []
    
    def __call__(
        self,
        step: int,
        policy_loss: float,
        entropy: float,
        kl_div: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check untuk stability issues.
        
        Args:
            step: Current step
            policy_loss: Policy loss
            entropy: Policy entropy
            kl_div: KL divergence
            
        Returns:
            Alert info
        """
        self.loss_history.append(policy_loss)
        
        current_alerts = []
        
        # Check loss spike
        if len(self.loss_history) > 10:
            mean_loss = np.mean(self.loss_history[-20:-5])
            if policy_loss > mean_loss * self.loss_spike_threshold:
                alert = {'type': 'loss_spike', 'step': step, 'value': policy_loss}
                current_alerts.append(alert)
                self.alerts.append(alert)
        
        # Check entropy collapse
        if entropy < self.entropy_min_threshold:
            alert = {'type': 'entropy_collapse', 'step': step, 'value': entropy}
            current_alerts.append(alert)
            self.alerts.append(alert)
        
        # Check KL explosion
        if kl_div > self.kl_max_threshold:
            alert = {'type': 'kl_explosion', 'step': step, 'value': kl_div}
            current_alerts.append(alert)
            self.alerts.append(alert)
        
        # Print alerts
        for alert in current_alerts:
            print(f"🚨 {alert['type'].upper()} at step {step}: {alert['value']:.4f}")
        
        return {
            'has_alerts': len(current_alerts) > 0,
            'alerts': current_alerts,
            'total_alerts': len(self.alerts)
        }


class AdaptiveAdjustmentCallback:
    """
    Callback yang auto-adjust hyperparameters.
    """
    
    def __init__(
        self,
        optimizer,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        adjustment_factor: float = 0.5
    ):
        """
        Inisialisasi callback.
        
        Args:
            optimizer: Optimizer untuk adjust
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            adjustment_factor: Factor untuk adjustment
        """
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adjustment_factor = adjustment_factor
        
        self.adjustments = []
    
    def on_instability(self, alert_type: str, step: int, value: float):
        """
        Called when instability detected.
        
        Args:
            alert_type: Type of instability
            step: Current step
            value: Metric value
        """
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if alert_type in ['gradient_explosion', 'loss_spike', 'kl_explosion']:
            # Reduce learning rate
            new_lr = max(self.min_lr, current_lr * self.adjustment_factor)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            adjustment = {
                'step': step,
                'type': alert_type,
                'old_lr': current_lr,
                'new_lr': new_lr
            }
            self.adjustments.append(adjustment)
            
            print(f"📉 LR reduced: {current_lr:.2e} -> {new_lr:.2e} (due to {alert_type})")
    
    def get_adjustment_history(self):
        """Get history of adjustments."""
        return self.adjustments


class EarlyStoppingCallback:
    """
    Callback untuk early stopping.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_improvement: float = 0.01,
        metric: str = 'win_rate'
    ):
        """
        Inisialisasi callback.
        
        Args:
            patience: Steps tanpa improvement sebelum stop
            min_improvement: Minimum improvement
            metric: Metric untuk track
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric
        
        self.best_value = float('-inf')
        self.steps_without_improvement = 0
        self.should_stop = False
    
    def __call__(self, step: int, metrics: Dict[str, float]) -> bool:
        """
        Check apakah harus stop.
        
        Args:
            step: Current step
            metrics: Current metrics
            
        Returns:
            True jika harus stop
        """
        if self.metric not in metrics:
            return False
        
        value = metrics[self.metric]
        
        if value > self.best_value + self.min_improvement:
            self.best_value = value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        if self.steps_without_improvement >= self.patience:
            self.should_stop = True
            print(f"\n⚠️ Early stopping triggered at step {step}")
            print(f"   No improvement in {self.metric} for {self.patience} evaluations")
            print(f"   Best value: {self.best_value:.4f}")
        
        return self.should_stop
