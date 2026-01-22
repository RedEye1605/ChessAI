"""
Proximal Policy Optimization (PPO)
==================================
Implementasi PPO untuk training agen catur dengan berbagai
stability enhancements.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from .replay_buffer import RolloutBuffer
from ..optimization import AdaptiveOptimizer


@dataclass
class PPOConfig:
    """Konfigurasi untuk PPO algorithm."""
    # Core PPO parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip range
    
    # Coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training parameters
    n_epochs: int = 4
    batch_size: int = 256
    mini_batch_size: int = 64
    
    # Stability parameters
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.015  # Early stopping KL threshold
    normalize_advantage: bool = True
    
    # Adaptive features
    adaptive_clip_range: bool = True
    entropy_scheduling: bool = True
    final_entropy_coef: float = 0.001
    
    # Logging
    log_frequency: int = 10


class PPO:
    """
    Proximal Policy Optimization dengan stability enhancements.
    
    Fitur:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy regularization dengan scheduling
    - KL divergence early stopping
    - Adaptive clip range
    - Advantage normalization
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        config: PPOConfig,
        device: torch.device = torch.device('cpu')
    ):
        """
        Inisialisasi PPO.
        
        Args:
            policy_network: Policy-Value network
            config: PPO configuration
            device: Torch device
        """
        self.policy = policy_network.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdaptiveOptimizer(
            model=self.policy,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            gradient_clipping='global_norm',
            lr_scheduler='none'  # Handle LR scheduling internally
        )
        
        # Rollout buffer
        self.buffer = None  # Will be initialized per rollout
        
        # Adaptive parameters
        self.current_clip_range = config.clip_range
        self.current_entropy_coef = config.entropy_coef
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
        self.total_updates = 0
    
    def select_action(
        self,
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action menggunakan current policy.
        
        Args:
            state: Current state
            legal_mask: Legal action mask
            deterministic: Use greedy action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        self.policy.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if legal_mask is not None:
                mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)
            else:
                mask_tensor = None
            
            log_probs, value = self.policy(state_tensor, mask_tensor)
            probs = torch.exp(log_probs)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
            
            log_prob = log_probs[0, action].item()
            value = value.item()
        
        self.policy.train()
        
        return action, log_prob, value
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss untuk satu batch.
        
        Args:
            batch: Dictionary dengan batch data
            
        Returns:
            loss: Total loss
            info: Dictionary dengan loss components
        """
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch['old_values']
        legal_masks = batch.get('legal_masks', None)
        
        # Forward pass
        log_probs, values = self.policy(states, legal_masks)
        values = values.squeeze(-1)
        
        # Get action log probs
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute entropy
        probs = torch.exp(log_probs)
        # Handle -inf in log_probs dengan masking
        valid_mask = ~torch.isinf(log_probs)
        entropy = -(probs * log_probs.masked_fill(~valid_mask, 0)).sum(dim=-1).mean()
        
        # Advantage normalization
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (clipped surrogate objective)
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.current_clip_range, 
            1.0 + self.current_clip_range
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (optionally clipped)
        if self.config.clip_range_vf is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf
            )
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = (
            policy_loss + 
            self.config.value_coef * value_loss - 
            self.current_entropy_coef * entropy
        )
        
        # Compute additional metrics
        with torch.no_grad():
            # KL divergence approximation
            log_ratio = action_log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            
            # Clip fraction
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.current_clip_range).float().mean().item()
            )
            
            # Explained variance
            y_true = returns.cpu().numpy()
            y_pred = values.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0
        
        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': approx_kl,
            'clip_fraction': clip_fraction,
            'explained_variance': explained_var
        }
        
        return loss, info
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update dari rollout buffer.
        
        Args:
            buffer: RolloutBuffer dengan collected experiences
            
        Returns:
            Dictionary dengan training statistics
        """
        # Aggregate statistics
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_kls = []
        all_clip_fractions = []
        
        # Training epochs
        for epoch in range(self.config.n_epochs):
            # Early stopping berdasarkan KL
            if self.config.target_kl is not None and all_kls:
                if np.mean(all_kls[-len(buffer) // self.config.mini_batch_size:]) > self.config.target_kl:
                    break
            
            # Iterate over mini-batches
            for batch in buffer.get_batches(self.config.mini_batch_size, shuffle=True):
                # Compute loss
                loss, info = self.compute_loss(batch)
                
                # Optimizer step
                self.optimizer.zero_grad()
                self.optimizer.step(loss)
                
                # Record statistics
                all_policy_losses.append(info['policy_loss'])
                all_value_losses.append(info['value_loss'])
                all_entropies.append(info['entropy'])
                all_kls.append(info['kl_divergence'])
                all_clip_fractions.append(info['clip_fraction'])
        
        # Update adaptive parameters
        self._update_adaptive_params(np.mean(all_kls) if all_kls else 0)
        
        self.total_updates += 1
        
        # Compile statistics
        stats = {
            'policy_loss': np.mean(all_policy_losses) if all_policy_losses else 0,
            'value_loss': np.mean(all_value_losses) if all_value_losses else 0,
            'entropy': np.mean(all_entropies) if all_entropies else 0,
            'kl_divergence': np.mean(all_kls) if all_kls else 0,
            'clip_fraction': np.mean(all_clip_fractions) if all_clip_fractions else 0,
            'n_updates': len(all_policy_losses),
            'clip_range': self.current_clip_range,
            'entropy_coef': self.current_entropy_coef
        }
        
        # Update history
        for key in ['policy_loss', 'value_loss', 'entropy', 'kl_divergence', 'clip_fraction']:
            self.training_stats[key].append(stats[key])
        
        return stats
    
    def _update_adaptive_params(self, mean_kl: float):
        """Update adaptive parameters berdasarkan training dynamics."""
        # Adaptive clip range
        if self.config.adaptive_clip_range and self.config.target_kl is not None:
            if mean_kl > self.config.target_kl * 1.5:
                self.current_clip_range = max(0.05, self.current_clip_range * 0.9)
            elif mean_kl < self.config.target_kl / 1.5:
                self.current_clip_range = min(0.3, self.current_clip_range * 1.1)
        
        # Entropy scheduling
        if self.config.entropy_scheduling:
            # Linear decay ke final value
            decay_rate = 0.9999
            self.current_entropy_coef = max(
                self.config.final_entropy_coef,
                self.current_entropy_coef * decay_rate
            )
    
    def collect_rollout(
        self,
        env,
        n_steps: int,
        state_shape: Tuple[int, ...]
    ) -> Tuple[RolloutBuffer, Dict[str, Any]]:
        """
        Collect rollout dari environment.
        
        Args:
            env: Chess environment
            n_steps: Number of steps to collect
            state_shape: Shape of state
            
        Returns:
            buffer: Filled rollout buffer
            info: Rollout statistics
        """
        buffer = RolloutBuffer(
            buffer_size=n_steps,
            state_shape=state_shape,
            action_size=self.policy.action_size if hasattr(self.policy, 'action_size') else 4672,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Reset environment jika perlu
        state, info = env.reset()
        
        total_reward = 0
        episode_rewards = []
        current_episode_reward = 0
        
        for step in range(n_steps):
            # Get legal mask
            legal_mask = env.get_legal_action_mask()
            
            # Select action
            action, log_prob, value = self.select_action(state, legal_mask)
            
            # Step environment
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Add to buffer
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
                legal_mask=legal_mask
            )
            
            current_episode_reward += reward
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, _ = env.reset()
            else:
                state = next_state
        
        # Compute last value untuk GAE
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, last_value = self.policy(state_tensor)
            last_value = last_value.item()
        
        # Compute returns dan advantages
        buffer.compute_returns_and_advantages(last_value, done)
        
        rollout_info = {
            'total_steps': n_steps,
            'episodes_completed': len(episode_rewards),
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_episode_reward': np.std(episode_rewards) if episode_rewards else 0
        }
        
        return buffer, rollout_info
    
    def save(self, path: str):
        """Save model dan optimizer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'total_updates': self.total_updates,
            'current_clip_range': self.current_clip_range,
            'current_entropy_coef': self.current_entropy_coef
        }, path)
    
    def load(self, path: str):
        """Load model dan optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.total_updates = checkpoint['total_updates']
        self.current_clip_range = checkpoint['current_clip_range']
        self.current_entropy_coef = checkpoint['current_entropy_coef']
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics history."""
        return self.training_stats


def create_ppo_agent(
    network: nn.Module,
    config_dict: Dict[str, Any],
    device: torch.device
) -> PPO:
    """
    Factory function untuk membuat PPO agent.
    
    Args:
        network: Policy-value network
        config_dict: Configuration dictionary
        device: Torch device
        
    Returns:
        PPO agent
    """
    config = PPOConfig(
        learning_rate=config_dict.get('learning_rate', 3e-4),
        gamma=config_dict.get('gamma', 0.99),
        gae_lambda=config_dict.get('gae_lambda', 0.95),
        clip_range=config_dict.get('clip_range', 0.2),
        value_coef=config_dict.get('value_coef', 0.5),
        entropy_coef=config_dict.get('entropy_coef', 0.01),
        n_epochs=config_dict.get('n_epochs', 4),
        batch_size=config_dict.get('batch_size', 256),
        mini_batch_size=config_dict.get('mini_batch_size', 64),
        max_grad_norm=config_dict.get('max_grad_norm', 0.5),
        target_kl=config_dict.get('target_kl', 0.015),
        normalize_advantage=config_dict.get('normalize_advantage', True),
        adaptive_clip_range=config_dict.get('adaptive_clip_range', True),
        entropy_scheduling=config_dict.get('entropy_scheduling', True)
    )
    
    return PPO(network, config, device)
