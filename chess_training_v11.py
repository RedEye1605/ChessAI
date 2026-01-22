"""
================================================================================
⚡ CHESS AI v11 - SUPERVISED ONLY (STABLE)
================================================================================
Simplified approach:
- ONLY Lichess games (no Stockfish evals that don't match)
- Value = game outcome (win/loss/draw)
- No self-play (which corrupts the model)
- Focus on opening book and strong supervised learning

Training time: ~2-3 hours on P100
Expected: 75-85% WR vs Random

Author: AI Assistant
Date: 2026-01-18 (v11 - Supervised Stable)
================================================================================
"""

# ==============================================================================
# Cell 1: Setup
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import time
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Install dependencies
try:
    import chess
except:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'python-chess', 'tqdm', 'matplotlib'])
    import chess

from tqdm import tqdm
import matplotlib.pyplot as plt

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
print("⚡ CHESS AI v11 - SUPERVISED ONLY (STABLE)")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Opening Book (Besar)
# ==============================================================================

# Comprehensive opening book untuk develop pieces dengan benar
OPENING_BOOK = {
    # === STARTING POSITION ===
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    
    # === AFTER 1.e4 ===
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b': ['e7e5', 'c7c5', 'e7e6', 'c7c6', 'd7d5', 'g8f6'],
    
    # === AFTER 1.d4 ===
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b': ['d7d5', 'g8f6', 'e7e6', 'f7f5'],
    
    # === AFTER 1.c4 ===
    'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b': ['e7e5', 'c7c5', 'g8f6', 'e7e6'],
    
    # === AFTER 1.Nf3 ===
    'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b': ['d7d5', 'g8f6', 'c7c5', 'e7e6'],
    
    # === 1.e4 e5 - Open Games ===
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'f1c4', 'b1c3', 'f2f4'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b': ['b8c6', 'g8f6', 'd7d6'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w': ['f1b5', 'f1c4', 'd2d4', 'b1c3'],
    
    # === Italian Game ===
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b': ['g8f6', 'f8c5', 'f8e7', 'd7d6'],
    'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w': ['d2d3', 'b1c3', 'c2c3', 'o-o'],
    'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w': ['c2c3', 'd2d3', 'b1c3', 'o-o'],
    
    # === Ruy Lopez ===
    'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b': ['a7a6', 'g8f6', 'f8c5', 'd7d6'],
    'r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w': ['f1a4', 'b5c6', 'b5a4'],
    
    # === Sicilian Defense ===
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'b1c3', 'c2c3', 'd2d4'],
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b': ['d7d6', 'b8c6', 'e7e6', 'g8f6'],
    
    # === French Defense ===
    'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w': ['d2d4', 'b1c3', 'd2d3', 'g1f3'],
    'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b': ['d7d5', 'c7c5', 'g8f6'],
    
    # === Caro-Kann ===
    'rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w': ['d2d4', 'b1c3', 'g1f3', 'c2c4'],
    
    # === Queen's Gambit ===
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w': ['c2c4', 'g1f3', 'c1f4', 'b1c3'],
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b': ['e7e6', 'c7c6', 'd5c4', 'e7e5'],
    'rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w': ['b1c3', 'g1f3', 'c1g5'],
    
    # === Indian Defenses ===
    'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w': ['c2c4', 'g1f3', 'c1f4', 'b1c3'],
    'rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b': ['e7e6', 'g7g6', 'c7c5', 'd7d6'],
    
    # === King's Indian ===
    'rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w': ['b1c3', 'g1f3', 'g2g3'],
    'rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b': ['o-o', 'd7d6', 'c7c5'],
    
    # === Development moves after castling ===
    'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b': ['o-o', 'd7d6', 'a7a6'],
    'r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w': ['d2d3', 'c2c3', 'b1c3', 'h2h3'],
}

def get_opening_move(board: chess.Board) -> Optional[str]:
    """Get move from opening book."""
    fen_key = ' '.join(board.fen().split()[:2])
    if fen_key in OPENING_BOOK:
        moves = [m for m in OPENING_BOOK[fen_key] 
                 if chess.Move.from_uci(m) in board.legal_moves]
        if moves:
            return random.choice(moves)
    return None

print(f"✅ Opening book: {len(OPENING_BOOK)} positions")

# ==============================================================================
# Cell 3: State Encoder
# ==============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    """8-channel board encoding."""
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
# Cell 4: Neural Network
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
                probs = torch.zeros_like(logits)
                probs[0, logits.argmax()] = 1.0
            else:
                probs = F.softmax(logits / temp, dim=-1)
            
            return probs.squeeze(0).cpu().numpy(), value.item()

network = ChessNet(filters=128, blocks=6).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 5: Load Lichess Dataset
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading Lichess dataset...")

lichess_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "datasnaek/chess",
    "games.csv"
)
print(f"✅ Loaded {len(lichess_df):,} games")

# ==============================================================================
# Cell 6: Process Games into Training Data
# ==============================================================================

def process_games(df, min_rating=1000):
    """Convert Lichess games to training data (state, action, value)."""
    states, actions, values = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # Filter by rating
            w_rating = row.get('white_rating', 0)
            b_rating = row.get('black_rating', 0)
            if max(w_rating, b_rating) < min_rating:
                continue
            
            board = chess.Board()
            winner = row['winner']
            moves_str = row['moves']
            
            if not isinstance(moves_str, str):
                continue
            
            for token in moves_str.split():
                if board.is_game_over():
                    break
                try:
                    move = board.parse_san(token)
                    
                    # State
                    state = encode_board(board)
                    
                    # Action
                    action = encode_move(move)
                    
                    # Value from game outcome (from current player's perspective)
                    if winner == 'white':
                        v = 1.0 if board.turn == chess.WHITE else -1.0
                    elif winner == 'black':
                        v = -1.0 if board.turn == chess.WHITE else 1.0
                    else:  # draw
                        v = 0.0
                    
                    states.append(state)
                    actions.append(action)
                    values.append(v)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    return np.array(states), np.array(actions), np.array(values)

print("\n📦 Processing games to training data...")
states, actions, values = process_games(lichess_df)
print(f"✅ Training data: {len(states):,} positions")
print(f"   Value distribution: +1={np.mean(values > 0):.1%}, 0={np.mean(values == 0):.1%}, -1={np.mean(values < 0):.1%}")

# Data augmentation: horizontal flip
def augment_data(states, actions, values):
    """Flip board horizontally for more data."""
    aug_states = np.flip(states, axis=3).copy()  # Flip file axis
    
    aug_actions = []
    for a in actions:
        from_sq, to_sq = a // 64, a % 64
        from_r, from_f = from_sq // 8, from_sq % 8
        to_r, to_f = to_sq // 8, to_sq % 8
        new_from = from_r * 8 + (7 - from_f)
        new_to = to_r * 8 + (7 - to_f)
        aug_actions.append(new_from * 64 + new_to)
    
    return aug_states, np.array(aug_actions), values.copy()

print("   Augmenting data...")
aug_states, aug_actions, aug_values = augment_data(states, actions, values)
states = np.concatenate([states, aug_states])
actions = np.concatenate([actions, aug_actions])
values = np.concatenate([values, aug_values])
print(f"✅ Total after augmentation: {len(states):,} positions")

# ==============================================================================
# Cell 7: Training
# ==============================================================================

def train(network, states, actions, values, epochs=20, batch_size=256, lr=1e-3):
    """Supervised training."""
    print(f"\n🎓 Training ({epochs} epochs)...")
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    n = len(states)
    history = {'ploss': [], 'vloss': [], 'wr': []}
    best_wr, best_state = 0, None
    patience, no_improve = 5, 0
    
    for epoch in range(epochs):
        network.train()
        indices = np.random.permutation(n)
        
        ploss_sum, vloss_sum, batches = 0, 0, 0
        
        for i in range(0, n, batch_size):
            batch = indices[i:i+batch_size]
            
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
        
        avg_p = ploss_sum / batches
        avg_v = vloss_sum / batches
        history['ploss'].append(avg_p)
        history['vloss'].append(avg_v)
        
        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0:
            wr = evaluate(network, 50)
            history['wr'].append(wr)
            print(f"   Epoch {epoch+1}/{epochs}: PLoss={avg_p:.4f}, VLoss={avg_v:.4f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
                no_improve = 0
                print(f"   🏆 New best!")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"   ⏹️ Early stopping")
                    break
    
    if best_state:
        network.load_state_dict(best_state)
        print(f"   ✅ Restored best (WR={best_wr:.0%})")
    
    return history, best_wr

# ==============================================================================
# Cell 8: Evaluation
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
                # Opening book first
                if board.fullmove_number <= 8:
                    book = get_opening_move(board)
                    if book:
                        board.push(chess.Move.from_uci(book))
                        continue
                
                # Network
                state = encode_board(board)
                mask = get_legal_mask(board)
                probs, _ = network.predict(state, mask, temp=0.1)
                move = decode_move(int(np.argmax(probs)), board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        if board.result() == '1-0':
            wins += 1
    
    return wins / n_games

# ==============================================================================
# Cell 9: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

start = time.time()

history, best_wr = train(network, states, actions, values, 
                          epochs=25, batch_size=256, lr=1e-3)

# Final evaluation
print(f"\n📊 Final Evaluation (100 games)...")
final_wr = evaluate(network, 100)
print(f"   Win Rate: {final_wr:.0%}")

total_time = time.time() - start
print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")

# Save model
torch.save(network.state_dict(), '/kaggle/working/chess_v11_final.pt')
print(f"💾 Model saved!")

# ==============================================================================
# Cell 10: Plot Results
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(history['ploss'], 'b-')
axes[0].set_title('Policy Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['vloss'], 'g-')
axes[1].set_title('Value Loss')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

if history['wr']:
    x = list(range(2, len(history['wr'])*2+1, 2))
    axes[2].plot(x, history['wr'], 'r-o', markersize=6)
    axes[2].axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title(f'Win Rate (Best: {best_wr:.0%})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_v11.png', dpi=150)
plt.show()

print(f"\n🎉 DONE!")
print(f"   Best Win Rate: {best_wr:.0%}")
print(f"   Final Win Rate: {final_wr:.0%}")
