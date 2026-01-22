"""
Evaluator untuk Chess RL Agent
==============================
Modul untuk mengevaluasi performa agen catur.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from ..environment import ChessEnv


@dataclass
class GameStats:
    """Statistics untuk satu game."""
    winner: Optional[str]
    num_moves: int
    opening_name: Optional[str]
    final_material: Dict[str, int]
    checkmate: bool
    stalemate: bool


class Evaluator:
    """
    Evaluator untuk menilai performa agen catur.
    
    Metrics:
    - Win/Draw/Loss rate
    - Average game length
    - Opening analysis
    - Move quality
    - ELO estimation
    """
    
    def __init__(
        self,
        env: ChessEnv,
        device: torch.device
    ):
        """
        Inisialisasi Evaluator.
        
        Args:
            env: Chess environment
            device: Torch device
        """
        self.env = env
        self.device = device
        
        # Results storage
        self.game_results: List[GameStats] = []
        
        # ELO tracking
        self.elo_rating = 1000  # Starting ELO
    
    def evaluate_agent(
        self,
        agent: nn.Module,
        num_games: int = 100,
        opponent: Optional[nn.Module] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent melalui multiple games.
        
        Args:
            agent: Agent untuk dievaluasi
            num_games: Jumlah games
            opponent: Lawan (default: random)
            verbose: Print progress
            
        Returns:
            Evaluation results
        """
        agent.eval()
        
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        checkmates = 0
        
        for i in range(num_games):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Game {i + 1}/{num_games}")
            
            result = self._play_game(agent, opponent)
            
            if result.winner == 'white':  # Assume agent is white
                wins += 1
            elif result.winner == 'black':
                losses += 1
            else:
                draws += 1
            
            total_moves += result.num_moves
            if result.checkmate:
                checkmates += 1
            
            self.game_results.append(result)
        
        agent.train()
        
        return {
            'num_games': num_games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games,
            'avg_game_length': total_moves / num_games,
            'checkmate_rate': checkmates / num_games
        }
    
    def _play_game(
        self,
        agent: nn.Module,
        opponent: Optional[nn.Module] = None
    ) -> GameStats:
        """Play satu game dan collect statistics."""
        state, _ = self.env.reset()
        done = False
        
        while not done:
            is_white = self.env.board.turn == chess.WHITE
            
            if is_white:
                action = self._select_action(agent, state)
            else:
                if opponent is not None:
                    action = self._select_action(opponent, state)
                else:
                    action = self._random_action()
            
            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        
        # Collect stats
        result = self.env.board.result()
        
        if result == '1-0':
            winner = 'white'
        elif result == '0-1':
            winner = 'black'
        else:
            winner = None
        
        return GameStats(
            winner=winner,
            num_moves=self.env.move_count,
            opening_name=None,
            final_material=self._count_material(),
            checkmate=self.env.board.is_checkmate(),
            stalemate=self.env.board.is_stalemate()
        )
    
    def _select_action(self, agent: nn.Module, state: np.ndarray) -> int:
        """Select action dari agent."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            legal_mask = torch.BoolTensor(
                self.env.get_legal_action_mask()
            ).unsqueeze(0).to(self.device)
            
            log_probs, _ = agent(state_tensor, legal_mask)
            action = log_probs.argmax(dim=-1).item()
        
        return action
    
    def _random_action(self) -> int:
        """Select random legal action."""
        legal_moves = list(self.env.board.legal_moves)
        if not legal_moves:
            return 0
        
        move = np.random.choice(legal_moves)
        return self.env.action_space_handler.encode_move(move)
    
    def _count_material(self) -> Dict[str, int]:
        """Count material di final position."""
        material = {'white': 0, 'black': 0}
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        for square in chess.SQUARES:
            piece = self.env.board.piece_at(square)
            if piece and piece.piece_type in piece_values:
                color = 'white' if piece.color else 'black'
                material[color] += piece_values[piece.piece_type]
        
        return material
    
    def estimate_elo(
        self,
        against_rating: int = 1200,
        games_won: int = 0,
        games_total: int = 0
    ) -> int:
        """
        Estimate ELO rating berdasarkan performance.
        
        Args:
            against_rating: Rating dari lawan
            games_won: Games yang dimenangkan
            games_total: Total games
            
        Returns:
            Estimated ELO rating
        """
        if games_total == 0:
            return self.elo_rating
        
        win_rate = games_won / games_total
        
        # ELO formula
        # Expected score = 1 / (1 + 10^((opponent_rating - player_rating) / 400))
        # Solve for player_rating given win_rate
        
        if win_rate == 0:
            return max(100, against_rating - 400)
        elif win_rate == 1:
            return against_rating + 400
        else:
            rating_diff = -400 * np.log10(1 / win_rate - 1)
            return int(against_rating + rating_diff)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.game_results:
            return {'num_games': 0}
        
        wins = sum(1 for g in self.game_results if g.winner == 'white')
        draws = sum(1 for g in self.game_results if g.winner is None)
        losses = sum(1 for g in self.game_results if g.winner == 'black')
        
        total = len(self.game_results)
        avg_length = sum(g.num_moves for g in self.game_results) / total
        
        return {
            'total_games': total,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / total,
            'avg_game_length': avg_length,
            'estimated_elo': self.elo_rating
        }


class MoveQualityAnalyzer:
    """
    Analyze kualitas moves dari agent.
    """
    
    def __init__(self, env: ChessEnv):
        """Initialize analyzer."""
        self.env = env
    
    def analyze_game(
        self,
        moves: List[chess.Move],
        agent_values: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze kualitas moves dalam satu game.
        
        Args:
            moves: List of moves
            agent_values: Value estimates dari agent
            
        Returns:
            Analysis results
        """
        blunders = 0  # Large value drop
        mistakes = 0  # Medium value drop
        inaccuracies = 0  # Small value drop
        
        for i in range(1, len(agent_values)):
            value_change = agent_values[i] - agent_values[i-1]
            
            # Dari perspektif player yang move
            if i % 2 == 1:  # White move
                if value_change < -0.3:
                    blunders += 1
                elif value_change < -0.1:
                    mistakes += 1
                elif value_change < -0.05:
                    inaccuracies += 1
        
        return {
            'blunders': blunders,
            'mistakes': mistakes,
            'inaccuracies': inaccuracies,
            'accuracy': 1 - (blunders * 0.5 + mistakes * 0.3 + inaccuracies * 0.1) / max(1, len(moves))
        }
