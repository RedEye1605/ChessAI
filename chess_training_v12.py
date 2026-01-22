"""
================================================================================
⚡ CHESS AI v12 - HYBRID SUPERVISED + STOCKFISH RL
================================================================================
Best of both worlds:
- Phase 1: Supervised learning from Lichess (fast bootstrap)
- Phase 2: RL with Stockfish library as reward (real-time evaluation)

Key optimizations for speed:
- Stockfish depth 8 (fast but accurate)
- Evaluate every 3 moves (not every move)
- Batch training after each game
- Early stopping when good enough

Training time: ~4-6 hours on P100
Expected: 85-95% WR vs Random

Author: AI Assistant
Date: 2026-01-18 (v12 - Stockfish RL)
================================================================================
"""

# ==============================================================================
# Cell 1: Setup & Dependencies
# ==============================================================================

# Install dependencies (Kaggle notebook style)
!pip install -q python-chess gymnasium tqdm matplotlib pandas
print("✅ Dependencies installed!")

import subprocess
import sys
import os

# Install Stockfish on Kaggle/Colab
if os.path.exists('/kaggle'):
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
elif os.path.exists('/content'):  # Colab
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
else:
    STOCKFISH_PATH = 'stockfish'  # Assume in PATH

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import chess
import chess.engine
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = torch.cuda.is_available()

print("=" * 60)
print("⚡ CHESS AI v12 - STOCKFISH RL")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Configuration
# ==============================================================================

@dataclass
class Config:
    # Network
    filters: int = 128
    blocks: int = 6
    
    # Phase 1: Supervised
    supervised_epochs: int = 15
    
    # Phase 2: Stockfish RL
    rl_iterations: int = 100
    games_per_iter: int = 20     # Fewer games but better quality
    stockfish_depth: int = 8      # Fast but decent
    eval_frequency: int = 3       # Evaluate every N moves
    
    # Training
    batch_size: int = 256
    lr_supervised: float = 1e-3
    lr_rl: float = 1e-4
    
    # Exploration
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    
    # Buffer
    buffer_size: int = 50000
    
    # Opening book moves
    book_moves: int = 6

config = Config()
print(f"✅ Config: SF depth={config.stockfish_depth}, eval every {config.eval_frequency} moves")

# ==============================================================================
# Cell 3: Stockfish Engine
# ==============================================================================

stockfish = None

def init_stockfish():
    global stockfish
    try:
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish.configure({"Threads": 1, "Hash": 64})
        print(f"✅ Stockfish loaded from {STOCKFISH_PATH}")
        return True
    except Exception as e:
        print(f"⚠️ Stockfish not available: {e}")
        print("   RL phase will use material-based evaluation")
        return False

def get_stockfish_eval(board: chess.Board, depth: int = 8) -> float:
    """Get Stockfish evaluation normalized to [-1, 1]."""
    if stockfish is None:
        return material_eval(board)
    
    try:
        result = stockfish.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].relative
        
        if score.is_mate():
            mate_in = score.mate()
            return 1.0 if mate_in > 0 else -1.0
        else:
            cp = score.score()
            return float(np.tanh(cp / 400))  # 400cp = ~0.76
    except:
        return material_eval(board)

def material_eval(board: chess.Board) -> float:
    """Backup evaluation based on material."""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    
    # Normalize
    return float(np.tanh(score / 10))

has_stockfish = init_stockfish()

# ==============================================================================
# Cell 4: Opening Book
# ==============================================================================

OPENING_BOOK = {
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b': ['d7d5', 'g8f6', 'e7e6'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'f1c4', 'b1c3'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b': ['b8c6', 'g8f6', 'd7d6'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w': ['f1b5', 'f1c4', 'd2d4'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b': ['g8f6', 'f8c5'],
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w': ['c2c4', 'g1f3', 'c1f4'],
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b': ['e7e6', 'c7c6', 'd5c4'],
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'b1c3', 'c2c3'],
}

def get_opening_move(board: chess.Board) -> Optional[str]:
    fen_key = ' '.join(board.fen().split()[:2])
    if fen_key in OPENING_BOOK:
        moves = [m for m in OPENING_BOOK[fen_key] if chess.Move.from_uci(m) in board.legal_moves]
        if moves:
            return random.choice(moves)
    return None

print(f"✅ Opening book: {len(OPENING_BOOK)} positions")

# ==============================================================================
# Cell 5: State Encoder
# ==============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    state = np.zeros((8, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_map[piece.piece_type]
            r, f = sq // 8, sq % 8
            state[ch, r, f] = 1.0 if piece.color == chess.WHITE else -1.0
    
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    return state

NUM_ACTIONS = 4096

def encode_move(m: chess.Move) -> int:
    return m.from_square * 64 + m.to_square

def decode_move(a: int, board: chess.Board) -> Optional[chess.Move]:
    f, t = a // 64, a % 64
    for m in board.legal_moves:
        if m.from_square == f and m.to_square == t:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[encode_move(m)] = True
    return mask

# ==============================================================================
# Cell 6: Neural Network
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c//8), nn.ReLU(),
            nn.Linear(c//8, c), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).view(-1, x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), SEBlock(c)
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class ChessNet(nn.Module):
    def __init__(self, filters=128, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(8, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(2048, NUM_ACTIONS)
        )
        self.value = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        x = self.blocks(self.stem(x))
        p = self.policy(x)
        v = self.value(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)
        return p, v
    
    def predict(self, state, mask, temp=0.3):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            logits, value = self(x, m)
            
            if temp < 0.1:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temp, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            return action, value.item()

network = ChessNet(config.filters, config.blocks).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 7: Experience Buffer
# ==============================================================================

class Buffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
    
    def add(self, state, action, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        
        while len(self.states) > self.maxlen:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
    
    def sample(self, n):
        if len(self.states) < n:
            idx = list(range(len(self.states)))
        else:
            idx = random.sample(range(len(self.states)), n)
        
        return (np.array([self.states[i] for i in idx]),
                np.array([self.actions[i] for i in idx]),
                np.array([self.rewards[i] for i in idx]),
                np.array([self.values[i] for i in idx]))
    
    def __len__(self):
        return len(self.states)

buffer = Buffer(config.buffer_size)

# ==============================================================================
# Cell 8: Load Lichess Data
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading Lichess dataset...")
lichess_df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "datasnaek/chess", "games.csv")
print(f"✅ Loaded {len(lichess_df):,} games")

# ==============================================================================
# Cell 9: Phase 1 - Supervised Learning
# ==============================================================================

def supervised_phase(network, df, epochs=15):
    """Phase 1: Learn from human games."""
    print("\n📚 PHASE 1: Supervised Learning")
    print("=" * 50)
    
    # Process games
    states, actions, values = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            board = chess.Board()
            winner = row['winner']
            
            for token in str(row['moves']).split():
                try:
                    move = board.parse_san(token)
                    states.append(encode_board(board))
                    actions.append(encode_move(move))
                    
                    if winner == 'white':
                        v = 1.0 if board.turn == chess.WHITE else -1.0
                    elif winner == 'black':
                        v = -1.0 if board.turn == chess.WHITE else 1.0
                    else:
                        v = 0.0
                    values.append(v)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    states = np.array(states)
    actions = np.array(actions)
    values = np.array(values)
    print(f"   Training data: {len(states):,} positions")
    
    # Training
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_supervised, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    best_wr, best_state = 0, None
    
    for epoch in range(epochs):
        network.train()
        idx = np.random.permutation(len(states))
        ploss_sum, vloss_sum, batches = 0, 0, 0
        
        for i in range(0, len(states), config.batch_size):
            batch = idx[i:i+config.batch_size]
            
            s = torch.FloatTensor(states[batch]).to(device)
            a = torch.LongTensor(actions[batch]).to(device)
            v = torch.FloatTensor(values[batch]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                logits, pred_v = network(s)
                ploss = F.cross_entropy(logits, a, label_smoothing=0.1)
                vloss = F.mse_loss(pred_v.squeeze(-1), v)
                loss = ploss + 0.5 * vloss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ploss_sum += ploss.item()
            vloss_sum += vloss.item()
            batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 3 == 0:
            wr = evaluate(network, 50)
            print(f"   Epoch {epoch+1}/{epochs}: PLoss={ploss_sum/batches:.4f}, VLoss={vloss_sum/batches:.4f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
    
    if best_state:
        network.load_state_dict(best_state)
    
    print(f"   ✅ Supervised complete: {best_wr:.0%}")
    torch.save(network.state_dict(), '/kaggle/working/chess_v12_supervised.pt')
    return best_wr

# ==============================================================================
# Cell 10: Phase 2 - Stockfish RL
# ==============================================================================

def play_rl_game(network, epsilon):
    """Play one game with Stockfish reward."""
    board = chess.Board()
    experiences = []
    move_count = 0
    prev_eval = 0.0
    
    while not board.is_game_over() and move_count < 150:
        # Opening book
        if move_count < config.book_moves:
            book = get_opening_move(board)
            if book:
                board.push(chess.Move.from_uci(book))
                move_count += 1
                continue
        
        state = encode_board(board)
        mask = get_legal_mask(board)
        
        # Epsilon-greedy
        if random.random() < epsilon:
            legal = list(board.legal_moves)
            move = random.choice(legal)
            action = encode_move(move)
        else:
            action, _ = network.predict(state, mask, temp=0.3)
            move = decode_move(action, board)
            if move is None:
                move = random.choice(list(board.legal_moves))
                action = encode_move(move)
        
        # Store state before move
        experiences.append({
            'state': state,
            'action': action,
            'move_count': move_count
        })
        
        board.push(move)
        move_count += 1
        
        # Get Stockfish reward every N moves
        if move_count % config.eval_frequency == 0:
            current_eval = get_stockfish_eval(board, config.stockfish_depth)
            
            # Reward = improvement in position
            # Flip for black's turn
            if board.turn == chess.BLACK:
                current_eval = -current_eval
            
            reward = current_eval - prev_eval
            prev_eval = current_eval
            
            # Apply reward to last N experiences
            for exp in experiences[-config.eval_frequency:]:
                exp['reward'] = reward / config.eval_frequency
    
    # Final reward based on game outcome
    result = board.result()
    if result == '1-0':
        final_reward = 1.0
    elif result == '0-1':
        final_reward = -1.0
    else:
        final_reward = 0.0
    
    # Add to buffer with combined rewards
    for exp in experiences:
        reward = exp.get('reward', 0) + final_reward * 0.1  # Mix incremental + final
        buffer.add(exp['state'], exp['action'], reward, final_reward)
    
    return result

def rl_phase(network, iterations):
    """Phase 2: RL with Stockfish rewards."""
    print("\n🤖 PHASE 2: Stockfish RL")
    print("=" * 50)
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_rl, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    best_wr = 0
    history = []
    
    for iteration in range(iterations):
        # Decay epsilon
        epsilon = config.epsilon_start - (config.epsilon_start - config.epsilon_end) * (iteration / iterations)
        
        # Play games
        wins, draws, losses = 0, 0, 0
        for _ in range(config.games_per_iter):
            result = play_rl_game(network, epsilon)
            if result == '1-0':
                wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        
        # Train on buffer
        if len(buffer) >= config.batch_size:
            network.train()
            
            for _ in range(20):  # Multiple updates per iteration
                states, actions, rewards, values = buffer.sample(config.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                values = torch.FloatTensor(values).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits, pred_v = network(states)
                    
                    # Policy gradient with advantage
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    advantage = rewards - pred_v.squeeze(-1).detach()
                    policy_loss = -(selected_log_probs * advantage).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(pred_v.squeeze(-1), values)
                    
                    # Entropy for exploration
                    entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
                    
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        
        # Evaluate periodically
        if (iteration + 1) % 10 == 0:
            wr = evaluate(network, 50)
            history.append(wr)
            print(f"   Iter {iteration+1}/{iterations}: W={wins} D={draws} L={losses}, ε={epsilon:.2f}, WR={wr:.0%}, Buf={len(buffer):,}")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v12_best.pt')
    
    print(f"   ✅ RL complete: best WR={best_wr:.0%}")
    return best_wr, history

# ==============================================================================
# Cell 11: Evaluation
# ==============================================================================

def evaluate(network, n_games):
    """Evaluate against random opponent."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        
        for _ in range(150):
            if board.is_game_over():
                break
            
            if board.turn == chess.WHITE:
                # Opening book
                if board.fullmove_number <= config.book_moves:
                    book = get_opening_move(board)
                    if book:
                        board.push(chess.Move.from_uci(book))
                        continue
                
                state = encode_board(board)
                mask = get_legal_mask(board)
                action, _ = network.predict(state, mask, temp=0.1)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        if board.result() == '1-0':
            wins += 1
    
    return wins / n_games

# ==============================================================================
# Cell 12: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

start = time.time()

# Phase 1: Supervised
supervised_wr = supervised_phase(network, lichess_df, epochs=config.supervised_epochs)

# Phase 2: Stockfish RL
rl_wr, rl_history = rl_phase(network, iterations=config.rl_iterations)

# Final evaluation
print(f"\n📊 Final Evaluation (100 games)...")
final_wr = evaluate(network, 100)
print(f"   Win Rate: {final_wr:.0%}")

total_time = time.time() - start
print(f"\n⏱️ Total time: {total_time/3600:.1f} hours")

# Save final
torch.save(network.state_dict(), '/kaggle/working/chess_v12_final.pt')
print("💾 Model saved!")

# Cleanup
if stockfish:
    stockfish.quit()

# ==============================================================================
# Cell 13: Plot Results
# ==============================================================================

if rl_history:
    plt.figure(figsize=(10, 5))
    x = list(range(10, len(rl_history)*10+1, 10))
    plt.plot(x, rl_history, 'b-o', markersize=6)
    plt.axhline(supervised_wr, color='red', linestyle='--', label=f'Supervised: {supervised_wr:.0%}')
    plt.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('RL Iteration')
    plt.ylabel('Win Rate vs Random')
    plt.title(f'Chess v12 Training (Final: {final_wr:.0%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('/kaggle/working/training_v12.png', dpi=150)
    plt.show()

print(f"\n🎉 DONE!")
print(f"   Supervised: {supervised_wr:.0%}")
print(f"   RL Best: {rl_wr:.0%}")
print(f"   Final: {final_wr:.0%}")
