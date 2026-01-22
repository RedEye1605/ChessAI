#!/usr/bin/env python3
"""
=============================================================================
Play Chess Against Your Kaggle-Trained Model
=============================================================================
Script untuk bermain melawan model yang di-training menggunakan 
complete_chess_training.py di Kaggle.

Penggunaan:
    python play_kaggle_model.py --checkpoint checkpoints/chess_model_best.pt
    python play_kaggle_model.py --checkpoint checkpoints/chess_model_best.pt --color black
    python play_kaggle_model.py --checkpoint checkpoints/chess_model_best.pt --demo

Author: Chess RL Project
=============================================================================
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
from pathlib import Path
from typing import Optional, Dict, Tuple
from torch.distributions import Categorical


# =============================================================================
# Chess Environment (Compatible with Kaggle Training - Version 1)
# =============================================================================

class ChessEnv:
    """
    Chess Environment dengan 18-channel state encoding.
    Kompatibel dengan complete_chess_training.py versi 1
    
    State: 18 channels x 8 x 8
    - 12 piece planes (6 white + 6 black)
    - 1 turn indicator
    - 4 castling rights
    - 1 en passant
    """
    
    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    def __init__(self, max_moves=200):
        self.board = chess.Board()
        self.max_moves = max_moves
        self.move_count = 0
        self._init_move_encoding()
    
    def _init_move_encoding(self):
        """Initialize AlphaZero-style action encoding."""
        self.action_to_move = {}
        self.move_to_action = {}
        
        # Direction vectors untuk queen-like moves
        directions = []
        for d in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            for dist in range(1, 8):
                directions.append((d[0]*dist, d[1]*dist))
        
        # Knight moves
        for d in [(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)]:
            directions.append(d)
        
        action = 0
        for sq in range(64):
            from_rank, from_file = sq // 8, sq % 8
            
            for dx, dy in directions:
                to_rank = from_rank + dy
                to_file = from_file + dx
                
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move = chess.Move(sq, to_sq)
                    self.action_to_move[action] = move
                    self.move_to_action[move.uci()] = action
                action += 1
            
            # Underpromotions
            if from_rank == 6:
                for dx in [-1, 0, 1]:
                    for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                        to_file = from_file + dx
                        if 0 <= to_file < 8:
                            to_sq = 7 * 8 + to_file
                            move = chess.Move(sq, to_sq, promotion=promo)
                            self.action_to_move[action] = move
                            self.move_to_action[move.uci()] = action
                        action += 1
    
    def encode_state(self) -> np.ndarray:
        """Encode board state as 18-channel numpy array."""
        state = np.zeros((18, 8, 8), dtype=np.float32)
        
        piece_to_channel = {
            (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
            (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
            (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
            (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11
        }
        
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                rank, file = sq // 8, sq % 8
                ch = piece_to_channel[(piece.piece_type, piece.color)]
                state[ch, rank, file] = 1.0
        
        # Turn indicator
        state[12, :, :] = 1.0 if self.board.turn else 0.0
        
        # Castling rights
        state[13, 0, :] = float(self.board.has_kingside_castling_rights(True))
        state[14, 0, :] = float(self.board.has_queenside_castling_rights(True))
        state[15, 0, :] = float(self.board.has_kingside_castling_rights(False))
        state[16, 0, :] = float(self.board.has_queenside_castling_rights(False))
        
        # En passant
        if self.board.ep_square:
            ep_rank, ep_file = self.board.ep_square // 8, self.board.ep_square % 8
            state[17, ep_rank, ep_file] = 1.0
        
        return state
    
    def get_legal_action_mask(self) -> np.ndarray:
        """Get binary mask for legal actions."""
        mask = np.zeros(4672, dtype=bool)
        for move in self.board.legal_moves:
            uci = move.uci()
            if uci in self.move_to_action:
                mask[self.move_to_action[uci]] = True
            # Handle queen promotions (default)
            elif len(uci) == 5 and uci[4] == 'q':
                base_uci = uci[:4]
                if base_uci in self.move_to_action:
                    mask[self.move_to_action[base_uci]] = True
        return mask
    
    def reset(self):
        """Reset the board to initial position."""
        self.board = chess.Board()
        self.move_count = 0
        return self.encode_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state."""
        if action not in self.action_to_move:
            return self.encode_state(), -1.0, True, {'illegal': True}
        
        move = self.action_to_move[action]
        
        # Find matching legal move
        executed = False
        for legal in self.board.legal_moves:
            if legal.uci()[:4] == move.uci()[:4]:
                self.board.push(legal)
                self.move_count += 1
                executed = True
                break
        
        if not executed:
            return self.encode_state(), -1.0, True, {'illegal': True}
        
        done = self.board.is_game_over() or self.move_count >= self.max_moves
        reward = 0.0
        
        if self.board.is_checkmate():
            reward = 1.0 if not self.board.turn else -1.0
        
        return self.encode_state(), reward, done, {}


# =============================================================================
# Neural Network (Compatible with Kaggle Training - Version 1)
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block untuk channel attention."""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block dengan optional SE attention."""
    
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class ChessNetwork(nn.Module):
    """
    Policy-Value Network untuk Chess RL.
    Kompatibel dengan complete_chess_training.py versi 1
    
    Architecture:
    - Input: (batch, 18, 8, 8)
    - Residual backbone dengan SE blocks (every other block)
    - Policy head: 80 channels → 4672 actions
    - Value head: 32 channels → 1 value
    """
    
    def __init__(self, input_channels=18, num_filters=256, num_blocks=12, action_size=4672):
        super().__init__()
        
        self.action_size = action_size
        
        # Input conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower dengan SE blocks (every other block)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, use_se=(i % 2 == 0))
            for i in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 80, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(80)
        self.policy_fc = nn.Linear(80 * 64, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x, legal_mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 18, 8, 8)
            legal_mask: Boolean mask for legal actions (batch, 4672)
            
        Returns:
            log_probs: Log probabilities (batch, action_size)
            value: Value estimate (batch, 1)
        """
        # Backbone
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Apply legal action mask
        if legal_mask is not None:
            policy_logits = policy_logits.float()
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_probs, value


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> Optional[ChessNetwork]:
    """Load model from Kaggle checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
        return None
    
    print(f"📂 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create network with matching architecture
    network = ChessNetwork(
        input_channels=config.get('input_channels', 18),
        num_filters=config.get('num_filters', 256),
        num_blocks=config.get('num_blocks', 12),
        action_size=4672
    )
    
    # Load weights
    network.load_state_dict(checkpoint['network_state_dict'])
    network = network.to(device)
    network.eval()
    
    # Print info
    best_win_rate = checkpoint.get('best_win_rate', 0)
    update = checkpoint.get('update', 0)
    print(f"✅ Model loaded successfully!")
    print(f"   Update: {update}")
    print(f"   Best win rate: {best_win_rate:.1%}")
    print(f"   Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    return network


# =============================================================================
# AI Agent
# =============================================================================

class ChessAI:
    """Chess AI agent using loaded network."""
    
    def __init__(self, network: ChessNetwork, device: torch.device):
        self.network = network
        self.device = device
    
    def select_action(self, env: ChessEnv, deterministic: bool = True) -> Tuple[int, float]:
        """Select action using the network."""
        state = env.encode_state()
        legal_mask = env.get_legal_action_mask()
        
        if legal_mask.sum() == 0:
            return None, 0.0
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)
            
            log_probs, value = self.network(state_t, mask_t)
            
            if deterministic:
                action = log_probs.argmax(dim=-1).item()
            else:
                dist = Categorical(logits=log_probs)
                action = dist.sample().item()
        
        return action, value.item()
    
    def get_move_uci(self, env: ChessEnv, deterministic: bool = True) -> Optional[str]:
        """Get move in UCI format."""
        action, _ = self.select_action(env, deterministic)
        if action is None:
            return None
        
        base_move = env.action_to_move.get(action)
        if base_move is None:
            return None
        
        # Find matching legal move
        for legal in env.board.legal_moves:
            if legal.uci()[:4] == base_move.uci()[:4]:
                return legal.uci()
        
        return None


# =============================================================================
# Play Functions
# =============================================================================

def play_terminal(ai: ChessAI, env: ChessEnv, player_color: str):
    """Play in terminal mode."""
    print("\n" + "=" * 60)
    print("🎯 Chess RL - Play Against Your AI")
    print("=" * 60)
    print(f"\n👤 You are playing as: {player_color.upper()}")
    print("\nCommands:")
    print("  - Type move in UCI format (e.g., e2e4)")
    print("  - 'quit' to exit")
    print("  - 'board' to show board")
    print("  - 'legal' to show legal moves")
    print("  - 'undo' to undo last move pair")
    print("=" * 60 + "\n")
    
    env.reset()
    player_is_white = (player_color == 'white')
    
    while not env.board.is_game_over():
        print(env.board)
        print(f"\nTurn: {'White' if env.board.turn else 'Black'}")
        
        is_player_turn = (env.board.turn == chess.WHITE) == player_is_white
        
        if is_player_turn:
            # Player's turn
            while True:
                cmd = input("\n🎮 Your move: ").strip().lower()
                
                if cmd == 'quit':
                    print("\n👋 Thanks for playing!")
                    return
                elif cmd == 'board':
                    print(env.board)
                    continue
                elif cmd == 'legal':
                    legal_moves = [m.uci() for m in env.board.legal_moves]
                    print(f"Legal moves: {', '.join(legal_moves)}")
                    continue
                elif cmd == 'undo':
                    if len(env.board.move_stack) >= 2:
                        env.board.pop()
                        env.board.pop()
                        print("⬅️ Move undone")
                        print(env.board)
                    else:
                        print("⚠️ No moves to undo")
                    continue
                
                try:
                    move = chess.Move.from_uci(cmd)
                    if move in env.board.legal_moves:
                        env.board.push(move)
                        print(f"✓ Played: {cmd}")
                        break
                    else:
                        # Check for promotion
                        found = False
                        for legal in env.board.legal_moves:
                            if legal.uci()[:4] == cmd[:4]:
                                env.board.push(legal)
                                print(f"✓ Played: {legal.uci()}")
                                found = True
                                break
                        if found:
                            break
                        print("❌ Illegal move! Try again.")
                except:
                    print("❌ Invalid format! Use UCI (e.g., e2e4)")
        else:
            # AI's turn
            print("\n🤖 AI is thinking...")
            
            move_uci = ai.get_move_uci(env)
            if move_uci:
                action, value = ai.select_action(env)
                move = chess.Move.from_uci(move_uci)
                san = env.board.san(move)
                env.board.push(move)
                print(f"🤖 AI plays: {san} ({move_uci}) [eval: {value:.2f}]")
            else:
                print("⚠️ AI has no legal moves!")
                break
    
    # Game over
    print("\n" + "=" * 60)
    print(env.board)
    result = env.board.result()
    
    if result == '1-0':
        winner = "White wins!"
    elif result == '0-1':
        winner = "Black wins!"
    else:
        winner = "Draw!"
    
    print(f"\n🏆 Game Over: {winner}")
    print(f"   Result: {result}")
    
    if player_is_white:
        if result == '1-0':
            print("   🎉 YOU WIN!")
        elif result == '0-1':
            print("   😢 AI wins...")
    else:
        if result == '0-1':
            print("   🎉 YOU WIN!")
        elif result == '1-0':
            print("   😢 AI wins...")
    
    print("=" * 60)


def play_demo(ai: ChessAI, env: ChessEnv, max_moves: int = 100):
    """Watch AI play a demo game against random opponent."""
    print("\n" + "=" * 60)
    print("🎬 Demo Game: AI (White) vs Random (Black)")
    print("=" * 60 + "\n")
    
    env.reset()
    move_history = []
    
    print("Initial position:")
    print(env.board)
    print()
    
    for move_num in range(max_moves):
        if env.board.is_game_over():
            break
        
        if env.board.turn == chess.WHITE:
            # AI plays white
            move_uci = ai.get_move_uci(env)
            if move_uci:
                action, value = ai.select_action(env)
                move = chess.Move.from_uci(move_uci)
                san = env.board.san(move)
                env.board.push(move)
                move_history.append(san)
                print(f"Move {len(move_history)}: AI plays {san} [eval: {value:.2f}]")
            else:
                break
        else:
            # Random plays black
            legal_moves = list(env.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                san = env.board.san(move)
                env.board.push(move)
                move_history.append(san)
                print(f"Move {len(move_history)}: Random plays {san}")
            else:
                break
    
    print("\n" + "=" * 60)
    print("Final position:")
    print(env.board)
    print(f"\nResult: {env.board.result()}")
    
    # Print PGN
    print("\n📜 Move history (PGN format):")
    pgn = ""
    for i, move in enumerate(move_history):
        if i % 2 == 0:
            pgn += f"{i // 2 + 1}. "
        pgn += move + " "
    print(pgn)
    print("=" * 60)


def evaluate_vs_random(ai: ChessAI, env: ChessEnv, n_games: int = 20):
    """Evaluate AI against random player."""
    print(f"\n📊 Evaluating AI vs Random ({n_games} games)...")
    
    wins, draws, losses = 0, 0, 0
    
    for game in range(n_games):
        env.reset()
        
        while not env.board.is_game_over() and env.move_count < 200:
            if env.board.turn == chess.WHITE:
                # AI plays white
                move_uci = ai.get_move_uci(env, deterministic=True)
                if move_uci:
                    move = chess.Move.from_uci(move_uci)
                    env.board.push(move)
                    env.move_count += 1
                else:
                    break
            else:
                # Random plays black
                legal_moves = list(env.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    env.board.push(move)
                    env.move_count += 1
                else:
                    break
        
        result = env.board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
        
        print(f"  Game {game + 1}/{n_games}: {result}", end='\r')
    
    print(f"\n\n📈 Results:")
    print(f"   Wins:   {wins} ({wins/n_games:.1%})")
    print(f"   Draws:  {draws} ({draws/n_games:.1%})")
    print(f"   Losses: {losses} ({losses/n_games:.1%})")
    
    return {'wins': wins, 'draws': draws, 'losses': losses}


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Play Chess Against Your Kaggle-Trained Model'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='checkpoints/chess_model_best.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='white',
        choices=['white', 'black', 'random'],
        help='Your color (default: white)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Watch AI play a demo game'
    )
    
    parser.add_argument(
        '--eval',
        type=int,
        default=0,
        metavar='N',
        help='Evaluate AI against random (N games)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Device: {device}")
    
    # Load model
    network = load_model(args.checkpoint, device)
    if network is None:
        return
    
    # Create environment and AI
    env = ChessEnv()
    ai = ChessAI(network, device)
    
    # Run requested mode
    if args.eval > 0:
        evaluate_vs_random(ai, env, args.eval)
    elif args.demo:
        play_demo(ai, env)
    else:
        player_color = args.color
        if player_color == 'random':
            player_color = random.choice(['white', 'black'])
        play_terminal(ai, env, player_color)


if __name__ == '__main__':
    main()
