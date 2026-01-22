"""
================================================================================
⚡ CHESS RL FINE-TUNING v8 - STOCKFISH EVALUATIONS
================================================================================
Fine-tune model 80% dengan:
1. Value Head dari 16M Stockfish evaluations (depth 22)
2. Policy dari Tactic puzzles (best moves)

Prerequisite: Model supervised sudah trained (chess_lichess_v7.pt)

Target: 85-95% vs Random

Author: AI Assistant
Date: 2026-01-16 (v8 - Stockfish RL)
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
print("⚡ CHESS RL v8 - STOCKFISH EVALUATIONS")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Dependencies
# ==============================================================================

try:
    import chess
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    print("✅ Dependencies loaded!")
except ImportError:
    print("Run: pip install python-chess tqdm matplotlib pandas")

# ==============================================================================
# Cell 3: Load Stockfish Evaluations Dataset
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading Stockfish Evaluations dataset...")

# Load tactic evaluations (has best moves!)
print("   Loading tactic_evals.csv...")
tactics_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ronakbadhe/chess-evaluations",
    "tactic_evals.csv"
)
print(f"   ✅ Tactics: {len(tactics_df):,} positions with best moves")

# Load main evaluations (sample for value training)
print("   Loading chessData.csv (sampling 500K)...")
evals_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ronakbadhe/chess-evaluations",
    "chessData.csv"
)
# Sample 500K for memory efficiency
if len(evals_df) > 500000:
    evals_df = evals_df.sample(500000, random_state=42)
print(f"   ✅ Evaluations: {len(evals_df):,} positions")

print("\n📊 Dataset samples:")
print(tactics_df.head(3))

# ==============================================================================
# Cell 4: State Encoder (from FEN)
# ==============================================================================

def fen_to_board(fen: str) -> Optional[chess.Board]:
    """Parse FEN to board."""
    try:
        return chess.Board(fen)
    except:
        return None

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board to 8x8x8 tensor."""
    state = np.zeros((8, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_to_channel[piece.piece_type]
            rank = sq // 8
            file = sq % 8
            state[ch, rank, file] = 1.0 if piece.color == chess.WHITE else -1.0
    
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    
    return state

# Action space
NUM_ACTIONS = 64 * 64

def encode_move(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    from_sq = action // 64
    to_sq = action % 64
    for m in board.legal_moves:
        if m.from_square == from_sq and m.to_square == to_sq:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask

def parse_uci_move(board: chess.Board, uci: str) -> Optional[chess.Move]:
    """Parse UCI move string."""
    if not uci or uci == '[null]' or pd.isna(uci):
        return None
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
        # Try with promotion
        for promo in ['q', 'r', 'b', 'n']:
            try:
                move = chess.Move.from_uci(uci + promo)
                if move in board.legal_moves:
                    return move
            except:
                pass
    except:
        pass
    return None

# ==============================================================================
# Cell 5: Neural Network with Value Head
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//8), nn.ReLU(), nn.Linear(ch//8, ch), nn.Sigmoid())
    def forward(self, x):
        y = self.pool(x).view(x.size(0), -1)
        return x * self.fc(y).view(x.size(0), -1, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(self.se(out) + x)

class ChessNetWithValue(nn.Module):
    """Network with both Policy and Value heads."""
    def __init__(self, in_ch=8, filters=128, blocks=6, actions=NUM_ACTIONS):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, actions)
        )
        
        # Value head (NEW!)
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()  # Output [-1, 1]
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
        x = self.res_blocks(self.input_conv(x))
        p = self.policy_head(x)
        v = self.value_head(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e4)
        return p, v
    
    def predict(self, state, mask):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            p, v = self(x, m)
            return F.softmax(p, dim=-1).squeeze(0).cpu().numpy(), v.item()

# Create network
network = ChessNetWithValue(in_ch=8, filters=128, blocks=6).to(device)

# Load pretrained weights (policy only)
pretrained_path = '/kaggle/working/chess_lichess_v7.pt'
if os.path.exists(pretrained_path):
    print(f"\n📂 Loading pretrained model from {pretrained_path}...")
    pretrained = torch.load(pretrained_path, map_location=device)
    # Load matching keys
    model_dict = network.state_dict()
    pretrained_filtered = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_filtered)
    network.load_state_dict(model_dict)
    print(f"   ✅ Loaded {len(pretrained_filtered)} layers from pretrained model")
else:
    print(f"   ⚠️ No pretrained model found at {pretrained_path}")

print(f"✅ ChessNetWithValue: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 6: Process Stockfish Data
# ==============================================================================

def parse_evaluation(eval_str) -> Optional[float]:
    """Parse evaluation string to float in [-1, 1] range."""
    if pd.isna(eval_str):
        return None
    eval_str = str(eval_str).strip()
    
    # Handle mate scores
    if eval_str.startswith('#+'):
        return 1.0  # Winning
    elif eval_str.startswith('#-'):
        return -1.0  # Losing
    elif eval_str.startswith('#'):
        return 1.0 if '+' in eval_str else -1.0
    
    try:
        # Centipawns to [-1, 1] with tanh-like scaling
        cp = float(eval_str)
        # Scale: 100cp ~ 0.2, 500cp ~ 0.7, 1000cp ~ 0.9
        return np.tanh(cp / 400)
    except:
        return None

print("\n📦 Processing datasets...")

# Process evaluations for value training
print("   Processing evaluations...")
value_states = []
value_targets = []

for _, row in tqdm(evals_df.iterrows(), total=len(evals_df), desc="Values"):
    board = fen_to_board(row['FEN'])
    if board is None:
        continue
    
    value = parse_evaluation(row['Evaluation'])
    if value is None:
        continue
    
    # Adjust for side to move (evaluation is from White's perspective)
    if board.turn == chess.BLACK:
        value = -value
    
    state = encode_board(board)
    value_states.append(state)
    value_targets.append(value)

value_states = np.array(value_states)
value_targets = np.array(value_targets)
print(f"   ✅ Value data: {len(value_states):,} positions")

# Process tactics for policy training
print("   Processing tactics...")
tactic_states = []
tactic_actions = []

for _, row in tqdm(tactics_df.iterrows(), total=len(tactics_df), desc="Tactics"):
    board = fen_to_board(row['FEN'])
    if board is None:
        continue
    
    move = parse_uci_move(board, row.get('Move', None))
    if move is None:
        continue
    
    state = encode_board(board)
    action = encode_move(move)
    
    tactic_states.append(state)
    tactic_actions.append(action)

tactic_states = np.array(tactic_states)
tactic_actions = np.array(tactic_actions)
print(f"   ✅ Tactic data: {len(tactic_states):,} positions with best moves")

# ==============================================================================
# Cell 7: Fine-tune Training
# ==============================================================================

def finetune(net, value_states, value_targets, tactic_states, tactic_actions,
             epochs=10, batch_size=256, lr=5e-4):
    """Fine-tune with value and policy."""
    print(f"\n🎓 Fine-tuning ({epochs} epochs)...")
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    history = {'value_loss': [], 'policy_loss': [], 'win_rate': []}
    best_wr = 0
    best_state = None
    
    n_value = len(value_states)
    n_policy = len(tactic_states)
    
    for epoch in range(epochs):
        net.train()
        total_vloss, total_ploss = 0, 0
        
        # Value training
        val_indices = np.random.permutation(n_value)
        for i in range(0, min(n_value, 50000), batch_size):  # 50K per epoch
            batch_idx = val_indices[i:i+batch_size]
            
            states = torch.FloatTensor(value_states[batch_idx]).to(device)
            targets = torch.FloatTensor(value_targets[batch_idx]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                _, values = net(states)
                vloss = F.mse_loss(values.squeeze(-1), targets)
            
            optimizer.zero_grad()
            scaler.scale(vloss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_vloss += vloss.item()
        
        # Policy training on tactics
        pol_indices = np.random.permutation(n_policy)
        for i in range(0, n_policy, batch_size):
            batch_idx = pol_indices[i:i+batch_size]
            
            states = torch.FloatTensor(tactic_states[batch_idx]).to(device)
            actions = torch.LongTensor(tactic_actions[batch_idx]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                logits, _ = net(states)
                ploss = F.cross_entropy(logits, actions)
            
            optimizer.zero_grad()
            scaler.scale(ploss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_ploss += ploss.item()
        
        avg_vloss = total_vloss / (min(n_value, 50000) // batch_size)
        avg_ploss = total_ploss / (n_policy // batch_size)
        
        history['value_loss'].append(avg_vloss)
        history['policy_loss'].append(avg_ploss)
        
        # Evaluate
        wr = evaluate(net, 50)
        history['win_rate'].append(wr)
        print(f"   Epoch {epoch+1}/{epochs}: VLoss={avg_vloss:.4f}, PLoss={avg_ploss:.4f}, WR={wr:.0%}")
        
        if wr > best_wr:
            best_wr = wr
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"   🏆 New best: {wr:.0%}")
    
    # Restore best
    if best_state:
        net.load_state_dict(best_state)
        print(f"\n   ✅ Restored best model (WR={best_wr:.0%})")
    
    return history, best_wr

# ==============================================================================
# Cell 8: Evaluation
# ==============================================================================

def evaluate(net, n_games):
    """Evaluate using policy + value for move selection."""
    net.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(150):
            if board.is_game_over():
                break
            
            if board.turn == chess.WHITE:
                state = encode_board(board)
                mask = get_legal_mask(board)
                probs, value = net.predict(state, mask)
                action = int(np.argmax(probs))
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
# Cell 9: Run Fine-tuning
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING FINE-TUNING WITH STOCKFISH DATA")
print("=" * 60)

# Check initial performance
print("\n📊 Initial evaluation...")
initial_wr = evaluate(network, 50)
print(f"   Win Rate before fine-tuning: {initial_wr:.0%}")

start_time = time.time()
history, best_wr = finetune(network, value_states, value_targets, 
                            tactic_states, tactic_actions,
                            epochs=10, batch_size=256, lr=5e-4)

# Final evaluation
print(f"\n📊 Final Evaluation (100 games)...")
final_wr = evaluate(network, 100)
print(f"   Win Rate: {final_wr:.0%}")

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")

# Save
torch.save(network.state_dict(), '/kaggle/working/chess_stockfish_v8.pt')
print("💾 Model saved!")

# ==============================================================================
# Cell 10: Plot
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['value_loss'], 'b-')
axes[0].set_title('Value Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['policy_loss'], 'g-')
axes[1].set_title('Policy Loss (Tactics)')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['win_rate'], 'r-o', markersize=8)
axes[2].axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='Previous best')
axes[2].set_title('Win Rate vs Random')
axes[2].set_xlabel('Epoch')
axes[2].set_ylim(0, 1)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_v8.png', dpi=150)
plt.show()

print(f"\n🎉 DONE! Before: {initial_wr:.0%} → After: {final_wr:.0%}")
