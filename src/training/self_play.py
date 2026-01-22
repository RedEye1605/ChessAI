"""
Self-Play Training Manager
==========================
Mengelola self-play training untuk agen catur.
Agen bermain melawan dirinya sendiri dan historical versions.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import copy
import random

from ..environment import ChessEnv
from ..algorithms.replay_buffer import RolloutBuffer


@dataclass
class GameResult:
    """Hasil satu game self-play."""
    winner: Optional[str]  # 'white', 'black', atau None untuk draw
    num_moves: int
    final_reward: float
    trajectory: List[Tuple[np.ndarray, int, float]]  # (state, action, reward)


class SelfPlayManager:
    """
    Manager untuk self-play training.
    
    Fitur:
    - Pool of historical opponents
    - Parallel game generation
    - Experience collection untuk training
    - ELO tracking
    """
    
    def __init__(
        self,
        policy: nn.Module,
        env: ChessEnv,
        device: torch.device,
        opponent_pool_size: int = 10,
        self_play_prob: float = 0.8,
        save_trajectories: bool = True
    ):
        """
        Inisialisasi Self-Play Manager.
        
        Args:
            policy: Policy network
            env: Chess environment
            device: Torch device
            opponent_pool_size: Ukuran pool untuk historical opponents
            self_play_prob: Probabilitas bermain melawan current policy
            save_trajectories: Apakah menyimpan full trajectories
        """
        self.policy = policy
        self.env = env
        self.device = device
        self.opponent_pool_size = opponent_pool_size
        self.self_play_prob = self_play_prob
        self.save_trajectories = save_trajectories
        
        # Opponent pool (historical versions)
        self.opponent_pool: deque = deque(maxlen=opponent_pool_size)
        
        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        
        # Current opponent
        self.current_opponent: Optional[nn.Module] = None
    
    def update_opponent_pool(self):
        """Tambahkan current policy ke opponent pool."""
        # Deep copy policy
        opponent_copy = copy.deepcopy(self.policy)
        opponent_copy.eval()
        self.opponent_pool.append(opponent_copy)
    
    def select_opponent(self) -> nn.Module:
        """Select opponent untuk game berikutnya."""
        if random.random() < self.self_play_prob or len(self.opponent_pool) == 0:
            # Play against self
            return self.policy
        else:
            # Play against random historical opponent
            return random.choice(list(self.opponent_pool))
    
    def play_game(
        self,
        player_white: nn.Module,
        player_black: nn.Module,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> GameResult:
        """
        Play satu game antara dua players.
        
        Args:
            player_white: Policy untuk white
            player_black: Policy untuk black
            deterministic: Gunakan greedy action selection
            temperature: Temperature untuk action sampling
            
        Returns:
            GameResult dengan hasil game
        """
        state, _ = self.env.reset()
        trajectory = []
        done = False
        
        while not done:
            # Determine current player
            is_white_turn = self.env.board.turn == chess.WHITE
            current_player = player_white if is_white_turn else player_black
            
            # Get legal mask
            legal_mask = self.env.get_legal_action_mask()
            
            # Select action
            action, log_prob, value = self._select_action(
                current_player, state, legal_mask, deterministic, temperature
            )
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Save trajectory
            if self.save_trajectories:
                trajectory.append((state.copy(), action, reward))
            
            state = next_state
        
        # Determine winner
        result = self.env.board.result()
        if result == '1-0':
            winner = 'white'
            self.white_wins += 1
        elif result == '0-1':
            winner = 'black'
            self.black_wins += 1
        else:
            winner = None
            self.draws += 1
        
        self.games_played += 1
        
        return GameResult(
            winner=winner,
            num_moves=self.env.move_count,
            final_reward=reward,
            trajectory=trajectory
        )
    
    def _select_action(
        self,
        policy: nn.Module,
        state: np.ndarray,
        legal_mask: np.ndarray,
        deterministic: bool,
        temperature: float
    ) -> Tuple[int, float, float]:
        """Select action dari policy."""
        policy.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)
            
            log_probs, value = policy(state_tensor, mask_tensor)
            
            if temperature != 1.0:
                log_probs = log_probs / temperature
                log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
            
            probs = torch.exp(log_probs)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                from torch.distributions import Categorical
                dist = Categorical(probs)
                action = dist.sample().item()
            
            log_prob = log_probs[0, action].item()
            value = value.item()
        
        return action, log_prob, value
    
    def generate_games(
        self,
        num_games: int,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> List[GameResult]:
        """
        Generate multiple games untuk training data.
        
        Args:
            num_games: Jumlah games yang akan di-generate
            deterministic: Use greedy selection
            temperature: Sampling temperature
            
        Returns:
            List of GameResults
        """
        results = []
        
        for _ in range(num_games):
            # Select opponent
            opponent = self.select_opponent()
            
            # Randomly assign colors
            if random.random() < 0.5:
                player_white = self.policy
                player_black = opponent
            else:
                player_white = opponent
                player_black = self.policy
            
            # Play game
            result = self.play_game(
                player_white, player_black, deterministic, temperature
            )
            results.append(result)
        
        return results
    
    def collect_training_data(
        self,
        num_games: int,
        state_shape: Tuple[int, ...],
        action_size: int = 4672
    ) -> RolloutBuffer:
        """
        Collect training data dari self-play games.
        
        Args:
            num_games: Number of games
            state_shape: Shape of state
            action_size: Action space size
            
        Returns:
            RolloutBuffer dengan training data
        """
        # Generate games
        game_results = self.generate_games(num_games)
        
        # Collect all experiences
        all_states = []
        all_actions = []
        all_rewards = []
        
        for result in game_results:
            for state, action, reward in result.trajectory:
                all_states.append(state)
                all_actions.append(action)
                all_rewards.append(reward)
        
        # Create buffer
        buffer = RolloutBuffer(
            buffer_size=len(all_states),
            state_shape=state_shape,
            action_size=action_size,
            device=self.device
        )
        
        # Add experiences
        for i, (state, action, reward) in enumerate(zip(all_states, all_actions, all_rewards)):
            done = (i == len(all_states) - 1)
            
            # Get value estimate
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, value = self.policy(state_tensor)
                value = value.item()
            
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=0.0,  # Will be computed during training
                value=value
            )
        
        # Compute returns
        buffer.compute_returns_and_advantages()
        
        return buffer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play statistics."""
        total = max(1, self.games_played)
        return {
            'games_played': self.games_played,
            'white_win_rate': self.white_wins / total,
            'black_win_rate': self.black_wins / total,
            'draw_rate': self.draws / total,
            'opponent_pool_size': len(self.opponent_pool)
        }
    
    def reset_statistics(self):
        """Reset game statistics."""
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0


class ArenaEvaluator:
    """
    Arena untuk evaluasi model melalui tournament.
    """
    
    def __init__(
        self,
        env: ChessEnv,
        device: torch.device,
        games_per_match: int = 10
    ):
        """
        Inisialisasi arena.
        
        Args:
            env: Chess environment
            device: Torch device
            games_per_match: Games per side dalam match
        """
        self.env = env
        self.device = device
        self.games_per_match = games_per_match
        
        self.self_play_manager = SelfPlayManager(
            policy=None,  # Will be set during matches
            env=env,
            device=device,
            save_trajectories=False
        )
    
    def play_match(
        self,
        player1: nn.Module,
        player2: nn.Module
    ) -> Dict[str, Any]:
        """
        Play match antara dua players.
        
        Args:
            player1: First player
            player2: Second player
            
        Returns:
            Match results
        """
        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        # Player 1 as white
        for _ in range(self.games_per_match):
            result = self.self_play_manager.play_game(
                player1, player2, deterministic=True
            )
            if result.winner == 'white':
                p1_wins += 1
            elif result.winner == 'black':
                p2_wins += 1
            else:
                draws += 1
        
        # Player 2 as white
        for _ in range(self.games_per_match):
            result = self.self_play_manager.play_game(
                player2, player1, deterministic=True
            )
            if result.winner == 'white':
                p2_wins += 1
            elif result.winner == 'black':
                p1_wins += 1
            else:
                draws += 1
        
        total_games = 2 * self.games_per_match
        
        return {
            'player1_wins': p1_wins,
            'player2_wins': p2_wins,
            'draws': draws,
            'player1_win_rate': p1_wins / total_games,
            'player2_win_rate': p2_wins / total_games
        }
