"""
Replay Buffer dan Rollout Buffer
=================================
Buffer untuk menyimpan experience selama training RL.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    legal_mask: np.ndarray


class RolloutBuffer:
    """
    Buffer untuk menyimpan rollout data untuk PPO training.
    
    Menyimpan trajectory dari satu rollout phase dan menghitung
    advantages menggunakan GAE.
    """
    
    def __init__(
        self,
        buffer_size: int,
        state_shape: Tuple[int, ...],
        action_size: int,
        device: torch.device = torch.device('cpu'),
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Inisialisasi Rollout Buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            state_shape: Shape of state tensor
            action_size: Size of action space
            device: Torch device
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
    
    def reset(self):
        """Reset buffer ke empty state."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.legal_masks = []
        
        self.advantages = None
        self.returns = None
        self.ptr = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        legal_mask: Optional[np.ndarray] = None
    ):
        """
        Tambah single experience ke buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            done: Episode done flag
            log_prob: Log probability of action
            value: Value estimate
            legal_mask: Legal action mask
        """
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        if legal_mask is not None:
            self.legal_masks.append(legal_mask.copy())
        
        self.ptr += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        last_done: bool = True
    ):
        """
        Compute returns dan advantages menggunakan GAE.
        
        Args:
            last_value: Value estimate untuk state terakhir
            last_done: Apakah episode terakhir selesai
        """
        n = len(self.rewards)
        
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        last_gae_lam = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_value = self.values[t + 1]
            
            delta = (self.rewards[t] + 
                    self.gamma * next_value * next_non_terminal - 
                    self.values[t])
            
            advantages[t] = last_gae_lam = (
                delta + 
                self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        returns = advantages + np.array(self.values)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate batches untuk training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Dictionary dengan batch data
        """
        n = len(self.states)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                'states': torch.FloatTensor(
                    np.array([self.states[i] for i in batch_indices])
                ).to(self.device),
                'actions': torch.LongTensor(
                    [self.actions[i] for i in batch_indices]
                ).to(self.device),
                'old_log_probs': torch.FloatTensor(
                    [self.log_probs[i] for i in batch_indices]
                ).to(self.device),
                'advantages': torch.FloatTensor(
                    self.advantages[batch_indices]
                ).to(self.device),
                'returns': torch.FloatTensor(
                    self.returns[batch_indices]
                ).to(self.device),
                'old_values': torch.FloatTensor(
                    [self.values[i] for i in batch_indices]
                ).to(self.device)
            }
            
            if self.legal_masks:
                batch['legal_masks'] = torch.BoolTensor(
                    np.array([self.legal_masks[i] for i in batch_indices])
                ).to(self.device)
            
            yield batch
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get semua data sebagai single batch."""
        data = {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device),
            'old_values': torch.FloatTensor(self.values).to(self.device)
        }
        
        if self.legal_masks:
            data['legal_masks'] = torch.BoolTensor(
                np.array(self.legal_masks)
            ).to(self.device)
        
        return data
    
    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    """
    Simple Replay Buffer untuk experience replay.
    
    Berbeda dari RolloutBuffer, ini menyimpan experiences secara permanent
    dan bisa di-sample secara random.
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        device: torch.device = torch.device('cpu')
    ):
        """
        Inisialisasi Replay Buffer.
        
        Args:
            capacity: Maximum capacity
            state_shape: Shape of state
            device: Torch device
        """
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience ke buffer."""
        idx = self.ptr % self.capacity
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch dari buffer."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device)
        }
    
    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Memprioritaskan experiences dengan TD-error yang tinggi.
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        device: torch.device = torch.device('cpu'),
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Initial importance sampling weight
        beta_increment: float = 0.001
    ):
        """Inisialisasi PER buffer."""
        super().__init__(capacity, state_shape, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add dengan max priority."""
        idx = self.ptr % self.capacity
        
        super().add(state, action, reward, next_state, done)
        
        # Set priority ke max
        self.priorities[idx] = self.max_priority ** self.alpha
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample dengan prioritization.
        
        Returns:
            batch: Data batch
            indices: Indices yang di-sample
            weights: Importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        
        # Compute importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device)
        }
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities berdasarkan TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
