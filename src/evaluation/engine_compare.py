"""
Stockfish Integration untuk Evaluasi
====================================
Modul untuk membandingkan agen dengan Stockfish engine.
"""

import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import subprocess
import os

try:
    from stockfish import Stockfish
    HAS_STOCKFISH = True
except ImportError:
    HAS_STOCKFISH = False

from ..environment import ChessEnv


@dataclass
class StockfishAnalysis:
    """Hasil analisis dari Stockfish."""
    best_move: str
    evaluation: float  # Centipawns
    mate_in: Optional[int]
    top_moves: List[Tuple[str, float]]


class StockfishEvaluator:
    """
    Evaluator menggunakan Stockfish engine.
    
    Fitur:
    - Play games melawan Stockfish
    - Analyze move quality
    - ELO estimation
    """
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        depth: int = 10,
        elo_limit: Optional[int] = None
    ):
        """
        Inisialisasi Stockfish Evaluator.
        
        Args:
            stockfish_path: Path ke Stockfish executable
            depth: Depth untuk analysis
            elo_limit: Limit ELO Stockfish (untuk fair play)
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.elo_limit = elo_limit
        
        if not HAS_STOCKFISH:
            print("⚠️ Stockfish Python package not installed. Install with: pip install stockfish")
            self.engine = None
        else:
            try:
                self.engine = Stockfish(path=stockfish_path)
                self.engine.set_depth(depth)
                
                if elo_limit:
                    self.engine.set_elo_rating(elo_limit)
                
                print(f"✅ Stockfish initialized (depth={depth}, elo_limit={elo_limit})")
            except Exception as e:
                print(f"⚠️ Failed to initialize Stockfish: {e}")
                self.engine = None
    
    def is_available(self) -> bool:
        """Check apakah Stockfish available."""
        return self.engine is not None
    
    def get_best_move(self, fen: str) -> Optional[str]:
        """
        Get best move dari Stockfish.
        
        Args:
            fen: FEN string dari posisi
            
        Returns:
            Best move dalam UCI format
        """
        if not self.is_available():
            return None
        
        self.engine.set_fen_position(fen)
        return self.engine.get_best_move()
    
    def get_evaluation(self, fen: str) -> float:
        """
        Get evaluation dari Stockfish.
        
        Args:
            fen: FEN string
            
        Returns:
            Evaluation dalam centipawns
        """
        if not self.is_available():
            return 0.0
        
        self.engine.set_fen_position(fen)
        evaluation = self.engine.get_evaluation()
        
        if evaluation['type'] == 'cp':
            return evaluation['value'] / 100.0  # Convert to pawns
        elif evaluation['type'] == 'mate':
            return 100.0 if evaluation['value'] > 0 else -100.0
        
        return 0.0
    
    def analyze_position(self, fen: str) -> Optional[StockfishAnalysis]:
        """
        Analyze posisi dengan Stockfish.
        
        Args:
            fen: FEN string
            
        Returns:
            StockfishAnalysis atau None
        """
        if not self.is_available():
            return None
        
        self.engine.set_fen_position(fen)
        
        best_move = self.engine.get_best_move()
        evaluation = self.engine.get_evaluation()
        top_moves = self.engine.get_top_moves(3)
        
        eval_value = 0.0
        mate_in = None
        
        if evaluation['type'] == 'cp':
            eval_value = evaluation['value'] / 100.0
        elif evaluation['type'] == 'mate':
            mate_in = evaluation['value']
            eval_value = 100.0 if mate_in > 0 else -100.0
        
        return StockfishAnalysis(
            best_move=best_move,
            evaluation=eval_value,
            mate_in=mate_in,
            top_moves=[(m['Move'], m.get('Centipawn', 0) / 100.0) for m in top_moves]
        )
    
    def play_game(
        self,
        agent,
        env: ChessEnv,
        device,
        agent_color: chess.Color = chess.WHITE
    ) -> Dict[str, Any]:
        """
        Play game antara agent dan Stockfish.
        
        Args:
            agent: RL agent
            env: Chess environment
            device: Torch device
            agent_color: Warna agent (WHITE atau BLACK)
            
        Returns:
            Game result dan statistics
        """
        import torch
        
        if not self.is_available():
            return {'error': 'Stockfish not available'}
        
        state, _ = env.reset()
        done = False
        agent_moves = []
        stockfish_moves = []
        
        while not done:
            is_agent_turn = (env.board.turn == agent_color)
            
            if is_agent_turn:
                # Agent move
                agent.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    legal_mask = torch.BoolTensor(
                        env.get_legal_action_mask()
                    ).unsqueeze(0).to(device)
                    
                    log_probs, _ = agent(state_tensor, legal_mask)
                    action = log_probs.argmax(dim=-1).item()
                
                move = env.action_space_handler.decode_action(action)
                agent_moves.append(move.uci())
            else:
                # Stockfish move
                fen = env.board.fen()
                best_move = self.get_best_move(fen)
                
                if best_move:
                    move = chess.Move.from_uci(best_move)
                    action = env.action_space_handler.encode_move(move)
                    stockfish_moves.append(best_move)
                else:
                    # Fallback ke random
                    action = env.action_space_handler.sample_legal_action(env.board)
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # Determine result
        result = env.board.result()
        
        if result == '1-0':
            winner = 'white'
        elif result == '0-1':
            winner = 'black'
        else:
            winner = 'draw'
        
        agent_won = (winner == 'white' and agent_color == chess.WHITE) or \
                    (winner == 'black' and agent_color == chess.BLACK)
        
        return {
            'winner': winner,
            'agent_won': agent_won,
            'num_moves': env.move_count,
            'agent_color': 'white' if agent_color == chess.WHITE else 'black',
            'agent_moves': agent_moves,
            'stockfish_moves': stockfish_moves,
            'final_fen': env.board.fen()
        }
    
    def evaluate_against_stockfish(
        self,
        agent,
        env: ChessEnv,
        device,
        num_games: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate agent melawan Stockfish dalam multiple games.
        
        Args:
            agent: RL agent
            env: Chess environment
            device: Torch device
            num_games: Number of games (total = num_games * 2 untuk kedua sisi)
            
        Returns:
            Evaluation results
        """
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        print(f"\n🏆 Evaluating against Stockfish ({num_games} games per side)")
        
        # Play as white
        print("  Playing as White...")
        for i in range(num_games):
            result = self.play_game(agent, env, device, chess.WHITE)
            
            if result.get('agent_won'):
                wins += 1
            elif result.get('winner') == 'draw':
                draws += 1
            else:
                losses += 1
            
            total_moves += result.get('num_moves', 0)
        
        # Play as black
        print("  Playing as Black...")
        for i in range(num_games):
            result = self.play_game(agent, env, device, chess.BLACK)
            
            if result.get('agent_won'):
                wins += 1
            elif result.get('winner') == 'draw':
                draws += 1
            else:
                losses += 1
            
            total_moves += result.get('num_moves', 0)
        
        total_games = num_games * 2
        
        # Estimate ELO based on performance
        stockfish_elo = self.elo_limit if self.elo_limit else 2000
        estimated_elo = self._estimate_elo(wins / total_games, stockfish_elo)
        
        return {
            'total_games': total_games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / total_games,
            'draw_rate': draws / total_games,
            'avg_game_length': total_moves / total_games,
            'stockfish_elo': stockfish_elo,
            'estimated_elo': estimated_elo
        }
    
    def _estimate_elo(self, win_rate: float, opponent_elo: int) -> int:
        """Estimate ELO dari win rate."""
        if win_rate == 0:
            return max(100, opponent_elo - 400)
        elif win_rate == 1:
            return opponent_elo + 400
        else:
            rating_diff = -400 * np.log10(1 / max(0.01, min(0.99, win_rate)) - 1)
            return int(opponent_elo + rating_diff)
    
    def analyze_agent_moves(
        self,
        agent,
        env: ChessEnv,
        device,
        num_positions: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze kualitas moves dari agent dibanding Stockfish.
        
        Args:
            agent: RL agent
            env: Environment
            device: Torch device
            num_positions: Number of positions untuk analyze
            
        Returns:
            Analysis results
        """
        import torch
        
        if not self.is_available():
            return {'error': 'Stockfish not available'}
        
        matches = 0  # Agent pilih move yang sama dengan Stockfish
        near_matches = 0  # Agent pilih salah satu dari top 3
        total = 0
        eval_differences = []
        
        for _ in range(num_positions):
            state, _ = env.reset()
            
            # Play beberapa moves random
            for _ in range(np.random.randint(5, 30)):
                if env.board.is_game_over():
                    break
                legal = list(env.board.legal_moves)
                move = np.random.choice(legal)
                env.board.push(move)
            
            if env.board.is_game_over():
                continue
            
            # Get agent's move
            state = env.state_encoder.encode(env.board)
            agent.eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                legal_mask = torch.BoolTensor(
                    env.get_legal_action_mask()
                ).unsqueeze(0).to(device)
                
                log_probs, _ = agent(state_tensor, legal_mask)
                action = log_probs.argmax(dim=-1).item()
            
            agent_move = env.action_space_handler.decode_action(action)
            
            # Get Stockfish analysis
            analysis = self.analyze_position(env.board.fen())
            
            if analysis:
                total += 1
                
                if agent_move.uci() == analysis.best_move:
                    matches += 1
                    near_matches += 1
                elif agent_move.uci() in [m[0] for m in analysis.top_moves]:
                    near_matches += 1
                
                # Compute eval difference
                env.board.push(agent_move)
                agent_eval = self.get_evaluation(env.board.fen())
                env.board.pop()
                
                env.board.push(chess.Move.from_uci(analysis.best_move))
                best_eval = self.get_evaluation(env.board.fen())
                env.board.pop()
                
                eval_differences.append(abs(agent_eval - best_eval))
        
        return {
            'positions_analyzed': total,
            'exact_match_rate': matches / max(1, total),
            'top3_match_rate': near_matches / max(1, total),
            'avg_eval_difference': np.mean(eval_differences) if eval_differences else 0,
            'max_eval_difference': np.max(eval_differences) if eval_differences else 0
        }
    
    def close(self):
        """Close Stockfish engine."""
        if self.engine:
            del self.engine
            self.engine = None
