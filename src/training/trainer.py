"""
Trainer untuk Chess RL
======================
Main training loop dengan semua fitur:
- Tensorboard logging
- Checkpointing
- Early stopping
- Stability monitoring
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from ..environment import ChessEnv
from ..models import ChessNetwork, create_network
from ..algorithms import PPO, PPOConfig
from ..optimization import AdaptiveOptimizer, GradientMonitor
from .self_play import SelfPlayManager


class Trainer:
    """
    Main Trainer untuk Chess RL.
    
    Mengelola:
    - Training loop
    - Logging ke Tensorboard
    - Model checkpointing
    - Early stopping
    - Evaluation
    - Stability monitoring
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Inisialisasi Trainer.
        
        Args:
            config: Training configuration dictionary
            device: Torch device (auto-detect jika None)
        """
        self.config = config
        
        # Setup device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = device
        
        print(f"🔧 Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Create components
        self.env = self.create_environment()
        self.network = self.create_network()
        self.ppo = self.create_ppo_agent()
        self.self_play = self.create_self_play_manager()
        
        # Gradient monitor untuk stability
        self.gradient_monitor = GradientMonitor(
            self.network,
            warn_threshold=config.get('stability', {}).get('gradient_norm_threshold', 100.0)
        )
        
        # Tensorboard
        if HAS_TENSORBOARD and config.get('logging', {}).get('tensorboard', True):
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # Training state
        self.global_step = 0
        self.best_win_rate = 0.0
        self.steps_without_improvement = 0
        
        # Save config
        self.save_config()
    
    def setup_directories(self):
        """Setup direktori untuk logs dan checkpoints."""
        general = self.config.get('general', {})
        
        experiment_name = general.get('experiment_name', 'chess_rl')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        base_log_dir = general.get('log_dir', 'logs')
        base_checkpoint_dir = general.get('checkpoint_dir', 'checkpoints')
        
        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_dir = Path(base_checkpoint_dir) / f"{experiment_name}_{timestamp}"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Log directory: {self.log_dir}")
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
    
    def save_config(self):
        """Save configuration ke file."""
        config_path = self.log_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def create_environment(self) -> ChessEnv:
        """Create chess environment."""
        env_config = self.config.get('environment', {})
        
        return ChessEnv(
            max_moves=env_config.get('max_moves', 200),
            reward_config={
                'win': env_config.get('checkmate_reward', 1.0),
                'lose': -env_config.get('checkmate_reward', 1.0),
                'draw': env_config.get('draw_reward', 0.0),
                'capture_scale': env_config.get('capture_reward_scale', 0.02)
            }
        )
    
    def create_network(self) -> nn.Module:
        """Create neural network."""
        network_config = self.config.get('network', {})
        
        network = create_network({
            'input_channels': network_config.get('input_channels', 14),
            'num_filters': network_config.get('num_filters', 256),
            'num_residual_blocks': network_config.get('num_residual_blocks', 10),
            'action_size': network_config.get('action_size', 4672),
            'normalization': network_config.get('normalization', 'layer'),
            'activation': network_config.get('activation', 'relu'),
            'dropout': network_config.get('dropout', 0.1)
        })
        
        network = network.to(self.device)
        
        # Print model info
        num_params = sum(p.numel() for p in network.parameters())
        print(f"🧠 Network parameters: {num_params:,}")
        
        return network
    
    def create_ppo_agent(self) -> PPO:
        """Create PPO agent."""
        ppo_config = self.config.get('ppo', {})
        adaptive_config = self.config.get('adaptive_optimization', {})
        
        config = PPOConfig(
            learning_rate=ppo_config.get('learning_rate', 3e-4),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            value_coef=ppo_config.get('value_coef', 0.5),
            entropy_coef=ppo_config.get('entropy_coef', 0.01),
            n_epochs=ppo_config.get('n_epochs', 4),
            batch_size=ppo_config.get('batch_size', 256),
            mini_batch_size=ppo_config.get('mini_batch_size', 64),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            target_kl=ppo_config.get('target_kl', 0.01),
            adaptive_clip_range=adaptive_config.get('adaptive_clip_range', True),
            entropy_scheduling=adaptive_config.get('entropy_scheduling', True),
            final_entropy_coef=adaptive_config.get('final_entropy_coef', 0.001)
        )
        
        return PPO(self.network, config, self.device)
    
    def create_self_play_manager(self) -> SelfPlayManager:
        """Create self-play manager."""
        sp_config = self.config.get('self_play', {})
        
        return SelfPlayManager(
            policy=self.network,
            env=self.env,
            device=self.device,
            opponent_pool_size=sp_config.get('opponent_pool_size', 10),
            self_play_prob=sp_config.get('self_play_prob', 0.8)
        )
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Callable] = None
    ):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total training timesteps (override config)
            callback: Optional callback function called after each update
        """
        training_config = self.config.get('training', {})
        
        if total_timesteps is None:
            total_timesteps = training_config.get('total_timesteps', 1_000_000)
        
        n_steps = training_config.get('n_steps', 128)
        eval_frequency = training_config.get('eval_frequency', 50)
        checkpoint_frequency = training_config.get('checkpoint_frequency', 100)
        early_stopping_patience = training_config.get('early_stopping_patience', 10)
        
        n_updates = total_timesteps // n_steps
        
        print(f"\n🚀 Starting training for {total_timesteps:,} timesteps ({n_updates} updates)")
        print(f"   Steps per update: {n_steps}")
        print(f"   Evaluation every {eval_frequency} updates")
        print(f"   Checkpoint every {checkpoint_frequency} updates\n")
        
        # Training progress bar
        pbar = tqdm(range(n_updates), desc="Training")
        
        for update in pbar:
            # Collect rollout
            buffer, rollout_info = self.ppo.collect_rollout(
                self.env,
                n_steps,
                state_shape=(14, 8, 8)
            )
            
            # PPO update
            update_stats = self.ppo.update(buffer)
            
            # Gradient monitoring
            grad_stats = self.gradient_monitor.update()
            
            self.global_step += n_steps
            
            # Logging
            if self.writer is not None:
                self.log_training_stats(update_stats, rollout_info, grad_stats)
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{rollout_info['mean_episode_reward']:.2f}",
                'policy_loss': f"{update_stats['policy_loss']:.4f}",
                'entropy': f"{update_stats['entropy']:.4f}"
            })
            
            # Evaluation
            if (update + 1) % eval_frequency == 0:
                eval_stats = self.evaluate()
                self.log_evaluation_stats(eval_stats)
                
                # Check for improvement
                if eval_stats['win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_stats['win_rate']
                    self.steps_without_improvement = 0
                    self.save_checkpoint('best')
                    print(f"\n✨ New best win rate: {self.best_win_rate:.2%}")
                else:
                    self.steps_without_improvement += 1
                
                # Early stopping
                if self.steps_without_improvement >= early_stopping_patience:
                    print(f"\n⚠️ Early stopping: no improvement for {early_stopping_patience} evaluations")
                    break
            
            # Checkpointing
            if (update + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(f'step_{self.global_step}')
                
                # Update opponent pool
                self.self_play.update_opponent_pool()
            
            # Callback
            if callback is not None:
                callback(update, update_stats, rollout_info)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint('final')
        
        print(f"\n✅ Training complete!")
        print(f"   Total steps: {self.global_step:,}")
        print(f"   Best win rate: {self.best_win_rate:.2%}")
    
    def evaluate(self, num_games: int = None) -> Dict[str, Any]:
        """
        Evaluate current policy.
        
        Args:
            num_games: Number of games for evaluation
            
        Returns:
            Evaluation statistics
        """
        if num_games is None:
            num_games = self.config.get('training', {}).get('eval_games', 20)
        
        self.network.eval()
        
        # Play games
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        for _ in range(num_games):
            result = self.self_play.play_game(
                self.network, self.network, deterministic=True
            )
            
            if result.winner == 'white':
                wins += 1
            elif result.winner == 'black':
                losses += 1
            else:
                draws += 1
            
            total_moves += result.num_moves
        
        self.network.train()
        
        return {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games,
            'avg_game_length': total_moves / num_games
        }
    
    def log_training_stats(
        self,
        update_stats: Dict[str, Any],
        rollout_info: Dict[str, Any],
        grad_stats: Dict[str, Any]
    ):
        """Log training statistics ke Tensorboard."""
        step = self.global_step
        
        # Rollout stats
        self.writer.add_scalar('rollout/mean_reward', rollout_info['mean_episode_reward'], step)
        self.writer.add_scalar('rollout/episodes', rollout_info['episodes_completed'], step)
        
        # PPO stats
        self.writer.add_scalar('train/policy_loss', update_stats['policy_loss'], step)
        self.writer.add_scalar('train/value_loss', update_stats['value_loss'], step)
        self.writer.add_scalar('train/entropy', update_stats['entropy'], step)
        self.writer.add_scalar('train/kl_divergence', update_stats['kl_divergence'], step)
        self.writer.add_scalar('train/clip_fraction', update_stats['clip_fraction'], step)
        
        # Adaptive params
        self.writer.add_scalar('adaptive/clip_range', update_stats['clip_range'], step)
        self.writer.add_scalar('adaptive/entropy_coef', update_stats['entropy_coef'], step)
        
        # Gradient stats
        self.writer.add_scalar('stability/gradient_norm', grad_stats['norm'], step)
        self.writer.add_scalar('stability/gradient_mean', grad_stats['mean_norm'], step)
    
    def log_evaluation_stats(self, eval_stats: Dict[str, Any]):
        """Log evaluation statistics."""
        if self.writer is None:
            return
        
        step = self.global_step
        
        self.writer.add_scalar('eval/win_rate', eval_stats['win_rate'], step)
        self.writer.add_scalar('eval/draw_rate', eval_stats['draw_rate'], step)
        self.writer.add_scalar('eval/avg_game_length', eval_stats['avg_game_length'], step)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f'{name}.pt'
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'ppo_state': {
                'total_updates': self.ppo.total_updates,
                'current_clip_range': self.ppo.current_clip_range,
                'current_entropy_coef': self.ppo.current_entropy_coef
            },
            'global_step': self.global_step,
            'best_win_rate': self.best_win_rate,
            'config': self.config
        }, path)
        
        print(f"💾 Saved checkpoint: {name}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        if 'ppo_state' in checkpoint:
            self.ppo.total_updates = checkpoint['ppo_state']['total_updates']
            self.ppo.current_clip_range = checkpoint['ppo_state']['current_clip_range']
            self.ppo.current_entropy_coef = checkpoint['ppo_state']['current_entropy_coef']
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        
        print(f"📂 Loaded checkpoint from {path}")
        print(f"   Global step: {self.global_step}")
        print(f"   Best win rate: {self.best_win_rate:.2%}")
    
    def close(self):
        """Cleanup resources."""
        if self.writer is not None:
            self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dari YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
