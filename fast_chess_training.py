"""
================================================================================
⚡ CHESS TRAINING v6.1 - LICHESS DATASET (VERIFIED)
================================================================================
Training dari DATA NYATA pemain manusia (1200+ ELO)

Dataset: datasnaek/chess dari Kaggle
- Format moves: Standard Algebraic Notation (SAN)
- Contoh: "e4 e5 Nf3 Nc6 Bb5 a6 ..."

Kolom dataset:
- id: Game ID
- rated: Boolean
- turns: Number of half-moves
- victory_status: 'resign', 'mate', 'outoftime', 'draw'
- winner: 'white', 'black', 'draw'
- white_rating, black_rating: ELO ratings
- moves: Space-separated SAN notation
- opening_eco: ECO code (e.g., "B00", "C50")
- opening_name: Opening name

Author: AI Assistant  
Date: 2026-01-16 (v6.1 - Verified)
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
print("⚡ CHESS TRAINING v6.1 - LICHESS DATA (VERIFIED)")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Dependencies & Download
# ==============================================================================

try:
    import chess
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    print("✅ Dependencies loaded!")
except ImportError:
    print("Run: pip install python-chess tqdm matplotlib pandas")

# Load dataset using kagglehub (works in non-interactive sessions)
import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading Lichess dataset...")
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "datasnaek/chess",
    "games.csv"  # Specify the file directly
)
print(f"✅ Dataset loaded: {len(df)} games")

# ==============================================================================
# Cell 3: Explore Dataset (Verification)
# ==============================================================================

print(f"\n📊 Dataset Info:")
print(f"   Total games: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

print(f"\n📊 Sample row:")
sample = df.iloc[0]
print(f"   ID: {sample['id']}")
print(f"   Winner: {sample['winner']}")
print(f"   White rating: {sample['white_rating']}")
print(f"   Black rating: {sample['black_rating']}")
print(f"   Turns: {sample['turns']}")
print(f"   Victory: {sample['victory_status']}")
print(f"   Opening: {sample['opening_name']} ({sample['opening_eco']})")
print(f"   Moves (first 100 chars): {str(sample['moves'])[:100]}...")

print(f"\n📊 Winner distribution:")
print(df['winner'].value_counts())

print(f"\n📊 Victory status:")
print(df['victory_status'].value_counts())

# ==============================================================================
# Cell 4: State Encoder (8x8x12 - AlphaZero style)
# ==============================================================================

# Piece type to channel mapping (separate channels for each color)
# Channels 0-5: White pieces (P, N, B, R, Q, K)
# Channels 6-11: Black pieces (p, n, b, r, q, k)

def encode_board_12ch(board: chess.Board) -> np.ndarray:
    """
    Encode board to 12x8x8 tensor (AlphaZero style).
    - Channels 0-5: White P, N, B, R, Q, K
    - Channels 6-11: Black P, N, B, R, Q, K
    """
    state = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_to_channel[(piece.piece_type, piece.color)]
            # Chess squares: a1=0, h8=63
            # Rank (row) = sq // 8, File (col) = sq % 8
            rank = sq // 8  # 0-7 (rank 1-8)
            file = sq % 8   # 0-7 (file a-h)
            state[ch, rank, file] = 1.0
    
    return state

def encode_board_8ch(board: chess.Board) -> np.ndarray:
    """
    Encode board to 8x8x8 tensor (simplified).
    - Channels 0-5: Pieces (P, N, B, R, Q, K) with +1 white, -1 black
    - Channel 6: Turn (1=white, -1=black)
    - Channel 7: Move number (normalized)
    """
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
    
    # Turn
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    
    # Move number (normalized)
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    
    return state

# Use 8-channel encoding
encode_board = encode_board_8ch
INPUT_CHANNELS = 8

# ==============================================================================
# Cell 5: Action Space (from_sq, to_sq, promotion)
# ==============================================================================

# Simple action space: from_square * 64 + to_square = 4096 actions
# (promotions handled by finding matching legal move)
NUM_ACTIONS = 64 * 64

def encode_move(move: chess.Move) -> int:
    """Encode move to action index."""
    return move.from_square * 64 + move.to_square

def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    """Decode action to legal move (handles promotions)."""
    from_sq = action // 64
    to_sq = action % 64
    
    # Find matching legal move
    for m in board.legal_moves:
        if m.from_square == from_sq and m.to_square == to_sq:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    """Get mask of legal actions."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask

print(f"✅ Action space: {NUM_ACTIONS} actions")

# ==============================================================================
# Cell 6: Parse Lichess Games
# ==============================================================================

def parse_game_moves(moves_str: str, board: chess.Board = None) -> List[chess.Move]:
    """
    Parse SAN moves string into list of chess.Move objects.
    Returns moves up to where parsing succeeds.
    """
    if not isinstance(moves_str, str) or not moves_str.strip():
        return []
    
    if board is None:
        board = chess.Board()
    
    moves = []
    tokens = moves_str.split()
    
    for token in tokens:
        # Skip move numbers like "1." or "23."
        if token.endswith('.') or token[0].isdigit() and '.' in token:
            continue
        
        # Skip annotations like "?", "!", "??"
        if token in ['?', '!', '??', '!!', '?!', '!?']:
            continue
        
        try:
            move = board.parse_san(token)
            moves.append(move)
            board.push(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            # Stop parsing on error
            break
    
    return moves

def mirror_state(state: np.ndarray) -> np.ndarray:
    """Mirror board horizontally (left-right flip)."""
    return np.flip(state, axis=2).copy()  # Flip along file axis

def mirror_action(action: int) -> int:
    """Mirror action horizontally."""
    from_sq = action // 64
    to_sq = action % 64
    # Mirror file: 0->7, 1->6, etc.
    from_file = from_sq % 8
    from_rank = from_sq // 8
    to_file = to_sq % 8
    to_rank = to_sq // 8
    
    new_from = from_rank * 8 + (7 - from_file)
    new_to = to_rank * 8 + (7 - to_file)
    return new_from * 64 + new_to

def game_to_training_data(moves: List[chess.Move], 
                          collect_white: bool = True,
                          collect_black: bool = False,
                          augment: bool = True) -> List[Tuple[np.ndarray, int]]:
    """
    Convert game moves to (state, action) training pairs.
    With data augmentation (horizontal mirror).
    """
    board = chess.Board()
    data = []
    
    for move in moves:
        should_collect = (board.turn == chess.WHITE and collect_white) or \
                         (board.turn == chess.BLACK and collect_black)
        
        if should_collect:
            state = encode_board(board)
            action = encode_move(move)
            data.append((state, action))
            
            # Data augmentation: add mirrored version
            if augment:
                data.append((mirror_state(state), mirror_action(action)))
        
        board.push(move)
    
    return data

# Test parsing
print("\n🔍 Testing move parsing...")
test_moves = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O"
test_board = chess.Board()
parsed = parse_game_moves(test_moves, test_board.copy())
print(f"   Input: {test_moves}")
print(f"   Parsed: {len(parsed)} moves")
print(f"   Moves: {[m.uci() for m in parsed[:5]]}...")

# ==============================================================================
# Cell 7: Load Dataset
# ==============================================================================

def load_training_data(df: pd.DataFrame, 
                       max_games: int = None,
                       min_rating: int = 1200,
                       winner_filter: str = 'white') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process games into training data.
    
    Args:
        df: DataFrame from games.csv
        max_games: Maximum games to process
        min_rating: Minimum rating filter
        winner_filter: 'white', 'black', or 'all'
    """
    # Filter by winner
    if winner_filter == 'white':
        filtered = df[df['winner'] == 'white']
    elif winner_filter == 'black':
        filtered = df[df['winner'] == 'black']
    else:
        filtered = df[df['winner'] != 'draw']
    
    print(f"\n📊 After winner filter ({winner_filter}): {len(filtered)} games")
    
    # Filter by rating
    if 'white_rating' in filtered.columns:
        filtered = filtered[
            (filtered['white_rating'] >= min_rating) | 
            (filtered['black_rating'] >= min_rating)
        ]
        print(f"   After rating filter (>={min_rating}): {len(filtered)} games")
    
    # Limit games
    if max_games and len(filtered) > max_games:
        filtered = filtered.sample(max_games, random_state=42)
        print(f"   Sampled: {max_games} games")
    
    # Process games
    all_states = []
    all_actions = []
    success_games = 0
    
    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Processing"):
        moves_str = row['moves']
        winner = row['winner']
        
        # Parse moves
        moves = parse_game_moves(moves_str)
        
        if len(moves) < 10:  # Skip very short games
            continue
        
        # Collect data for winning side
        collect_white = (winner == 'white')
        collect_black = (winner == 'black')
        
        data = game_to_training_data(moves, collect_white, collect_black)
        
        if data:
            success_games += 1
            for state, action in data:
                all_states.append(state)
                all_actions.append(action)
    
    print(f"\n✅ Processed {success_games} games successfully")
    print(f"   Total positions: {len(all_states):,}")
    
    return np.array(all_states), np.array(all_actions)

# Load data - USE ALL GAMES!
print("\n" + "=" * 60)
print("📦 LOADING TRAINING DATA (ALL GAMES + AUGMENTATION)")
print("=" * 60)

states, actions = load_training_data(
    df, 
    max_games=None,  # Use ALL games!
    min_rating=1100,  # Lower threshold for more data
    winner_filter='white'  # Train to play like winning WHITE
)

print(f"\n📊 Dataset ready:")
print(f"   States shape: {states.shape}")
print(f"   Actions shape: {actions.shape}")

# ==============================================================================
# Cell 8: Neural Network
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//8), nn.ReLU(), 
            nn.Linear(ch//8, ch), nn.Sigmoid()
        )
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

class ChessNet(nn.Module):
    def __init__(self, in_ch=8, filters=128, blocks=6, actions=NUM_ACTIONS):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, actions)
        )
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        x = self.res_blocks(self.input_conv(x))
        p = self.policy_head(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e4)
        return p
    
    def predict(self, state, mask):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            p = self(x, m)
            return F.softmax(p, dim=-1).squeeze(0).cpu().numpy()

network = ChessNet(in_ch=INPUT_CHANNELS, filters=128, blocks=6).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 9: Training with Early Stopping & Best Model
# ==============================================================================

def train(net, states, actions, epochs=25, batch_size=256, lr=1e-3, patience=5):
    """Supervised training with early stopping."""
    print(f"\n📚 Training ({epochs} epochs, patience={patience})...")
    
    n = len(states)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * (n // batch_size))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    history = {'loss': [], 'acc': [], 'win_rate': []}
    best_wr = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        net.train()
        total_loss, correct, total = 0, 0, 0
        indices = np.random.permutation(n)
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            batch_states = torch.FloatTensor(states[batch_idx]).to(device)
            batch_actions = torch.LongTensor(actions[batch_idx]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                logits = net(batch_states)
                loss = F.cross_entropy(logits, batch_actions, label_smoothing=0.1)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item() * len(batch_idx)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_actions).sum().item()
            total += len(batch_idx)
        
        acc = correct / total
        avg_loss = total_loss / n
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        
        # Evaluate every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            wr = evaluate(net, 50)
            history['win_rate'].append(wr)
            print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.1%}, WR={wr:.0%}")
            
            # Save best model
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
                print(f"   🏆 New best: {wr:.0%}")
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state:
        net.load_state_dict(best_state)
        print(f"\n   ✅ Restored best model (WR={best_wr:.0%})")
    
    return history, best_wr

# ==============================================================================
# Cell 10: Evaluation
# ==============================================================================

def evaluate(net, n_games):
    """Evaluate against random opponent."""
    net.eval()
    wins, draws, losses = 0, 0, 0
    
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(150):
            if board.is_game_over():
                break
            
            if board.turn == chess.WHITE:
                state = encode_board(board)
                mask = get_legal_mask(board)
                probs = net.predict(state, mask)
                action = int(np.argmax(probs))
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
    
    return wins / n_games

# ==============================================================================
# Cell 11: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING (with Early Stopping)")
print("=" * 60)

start_time = time.time()
history, best_wr = train(network, states, actions, epochs=25, batch_size=256, lr=1e-3, patience=5)

# Final evaluations
print(f"\n📊 Final Evaluation vs Random (100 games)...")
final_wr = evaluate(network, 100)
print(f"   Win Rate: {final_wr:.0%}")

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")

# Save
torch.save(network.state_dict(), '/kaggle/working/chess_lichess_v7.pt')
print("💾 Model saved!")

# ==============================================================================
# Cell 12: Plot
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['loss'])
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['acc'], 'g-')
axes[1].set_title('Accuracy') 
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

if history['win_rate']:
    x = list(range(1, len(history['win_rate']) * 3 + 1, 3))
    axes[2].plot(x, history['win_rate'], 'b-o', markersize=6)
    axes[2].axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[2].set_title('Win Rate vs Random')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_v7.png', dpi=150)
plt.show()

print(f"\n🎉 DONE! Best: {best_wr:.0%} | Final: {final_wr:.0%}")

