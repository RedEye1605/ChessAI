"""
================================================================================
⚡ CHESS AI v13 - COMPLETE REWRITE
================================================================================
Implementasi yang benar dan lengkap:

1. STATE ENCODING (12 channels):
   - 6 channels: Pieces (pawn, knight, bishop, rook, queen, king)
   - 1 channel: Side to move
   - 1 channel: Move number
   - 4 channels: Castling rights (WK, WQ, BK, BQ)

2. VALUE TARGET (konsisten):
   - Semua dari WHITE's perspective
   - Supervised: Material evaluation
   - RL: Stockfish evaluation

3. ECO-BASED OPENING BOOK:
   - Parse opening moves dari dataset berdasarkan opening_eco
   - Automatically build opening book dari games

4. PROPER RL:
   - Model (White) vs Random (Black)
   - Reward = Stockfish eval change
   - Value = Stockfish eval
   - advantage = reward - baseline

Training time: ~4-6 hours on P100
================================================================================
"""

# ==============================================================================
# Cell 1: Dependencies
# ==============================================================================

try:
    get_ipython().system('pip install -q python-chess tqdm matplotlib pandas kagglehub')
except:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'python-chess', 'tqdm', 'matplotlib', 'pandas', 'kagglehub'], capture_output=True)

import subprocess
import os

# Install Stockfish
if os.path.exists('/kaggle'):
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
elif os.path.exists('/content'):
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
else:
    STOCKFISH_PATH = 'stockfish'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import chess
import chess.engine
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

print("=" * 60)
print("⚡ CHESS AI v13 - COMPLETE REWRITE")
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
    input_channels: int = 12  # 6 pieces + turn + move# + 4 castling
    filters: int = 128
    blocks: int = 6
    
    # Training
    supervised_epochs: int = 15
    rl_iterations: int = 50
    games_per_iter: int = 30
    
    batch_size: int = 256
    lr_supervised: float = 1e-3
    lr_rl: float = 5e-5  # Lower for RL to preserve supervised knowledge
    
    # RL
    stockfish_depth: int = 8
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    gamma: float = 0.99  # Discount factor
    
    # Buffer
    buffer_size: int = 50000
    
    # Opening
    opening_book_depth: int = 8  # Build book from first N moves

config = Config()
print(f"✅ Config: {config.input_channels} input channels, {config.blocks} blocks")

# ==============================================================================
# Cell 3: Stockfish Engine
# ==============================================================================

stockfish = None

def init_stockfish():
    global stockfish
    try:
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish.configure({"Threads": 1, "Hash": 64})
        print(f"✅ Stockfish loaded")
        return True
    except Exception as e:
        print(f"⚠️ Stockfish not available: {e}")
        return False

def stockfish_eval(board: chess.Board, depth: int = 8) -> float:
    """
    Get Stockfish evaluation ALWAYS from WHITE's perspective.
    Returns value in [-1, 1].
    """
    if stockfish is None:
        return material_eval(board)
    
    try:
        result = stockfish.analyse(board, chess.engine.Limit(depth=depth))
        # Use .white() to always get from white's perspective
        score = result['score'].white()
        
        if score.is_mate():
            mate_in = score.mate()
            return 1.0 if mate_in > 0 else -1.0
        else:
            cp = score.score()
            # Normalize: 400cp = ~0.76, 800cp = ~0.96
            return float(np.tanh(cp / 400))
    except:
        return material_eval(board)

def material_eval(board: chess.Board) -> float:
    """
    Material evaluation from WHITE's perspective.
    Returns value in [-1, 1].
    """
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            v = values[piece.piece_type]
            score += v if piece.color == chess.WHITE else -v
    
    # Normalize
    return float(np.tanh(score / 15))

has_stockfish = init_stockfish()

# ==============================================================================
# Cell 4: State Encoding (12 channels)
# ==============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    """
    12-channel board encoding:
    - Channels 0-5: Pieces (pawn, knight, bishop, rook, queen, king)
                    +1 for white, -1 for black
    - Channel 6: Side to move (+1 white, -1 black)
    - Channel 7: Move number (normalized)
    - Channel 8-11: Castling rights (WK, WQ, BK, BQ)
    """
    state = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    
    # Pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_map[piece.piece_type]
            rank, file = sq // 8, sq % 8
            state[ch, rank, file] = 1.0 if piece.color == chess.WHITE else -1.0
    
    # Side to move
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    
    # Move number (normalized to 0-1)
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    
    # Castling rights
    state[8, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    state[9, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    state[10, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    state[11, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    return state

# Action space: 64 * 64 = 4096 (from_square * 64 + to_square)
NUM_ACTIONS = 4096

def encode_move(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    from_sq, to_sq = action // 64, action % 64
    for m in board.legal_moves:
        if m.from_square == from_sq and m.to_square == to_sq:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[encode_move(m)] = True
    return mask

print(f"✅ State encoding: 12 channels (with castling rights)")

# ==============================================================================
# Cell 5: Build Opening Book from Dataset
# ==============================================================================

def build_opening_book(df: pd.DataFrame, max_depth: int = 8) -> Dict[str, List[str]]:
    """
    Build opening book from Lichess games.
    Key = FEN (position), Value = list of moves played from that position
    """
    book = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building opening book"):
        try:
            board = chess.Board()
            moves_str = str(row['moves']).split()
            
            for i, token in enumerate(moves_str[:max_depth]):
                try:
                    # Key: position without move counters
                    fen_key = ' '.join(board.fen().split()[:4])
                    
                    move = board.parse_san(token)
                    move_uci = move.uci()
                    
                    # Add to book (avoid duplicates)
                    if move_uci not in book[fen_key]:
                        book[fen_key].append(move_uci)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    # Filter: keep only positions with multiple games
    filtered = {k: v for k, v in book.items() if len(v) >= 2}
    print(f"   Opening book: {len(filtered)} positions")
    return filtered

def get_book_move(board: chess.Board, opening_book: Dict) -> Optional[str]:
    """Get random move from opening book."""
    fen_key = ' '.join(board.fen().split()[:4])
    if fen_key in opening_book:
        moves = opening_book[fen_key]
        # Filter legal
        legal = [m for m in moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal:
            return random.choice(legal)
    return None

# ==============================================================================
# Cell 6: Neural Network
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.fc(x).view(-1, x.size(1), 1, 1)
        return x * scale

class ResBlock(nn.Module):
    """Residual block with SE."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)

class ChessNet(nn.Module):
    """
    Policy-Value Network for Chess.
    Input: 12 x 8 x 8
    Output: policy (4096 logits), value (scalar)
    """
    def __init__(self, in_channels=12, filters=128, blocks=6):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, NUM_ACTIONS)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        x = self.tower(self.stem(x))
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        if mask is not None:
            policy = policy.masked_fill(~mask, -1e9)
        
        return policy, value
    
    def predict(self, state: np.ndarray, mask: np.ndarray, temperature: float = 0.5):
        """Predict action and value for single state."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            
            if temperature < 0.1:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            return action, value.item()

network = ChessNet(in_channels=config.input_channels, 
                   filters=config.filters, 
                   blocks=config.blocks).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 7: Experience Buffer
# ==============================================================================

class ReplayBuffer:
    """Simple replay buffer for RL."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, state, action, reward, value_target):
        self.buffer.append((state, action, reward, value_target))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, values = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(values, dtype=np.float32))
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(config.buffer_size)

# ==============================================================================
# Cell 8: Load Dataset
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading Lichess dataset...")
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "datasnaek/chess", "games.csv")
print(f"✅ Loaded {len(df):,} games")

# Display dataset info
print(f"   Columns: {list(df.columns)}")
print(f"   Winners: {df['winner'].value_counts().to_dict()}")

# Build opening book
print("\n📖 Building opening book from dataset...")
opening_book = build_opening_book(df, max_depth=config.opening_book_depth)

# ==============================================================================
# Cell 9: Supervised Learning Phase
# ==============================================================================

def supervised_phase(network, df, opening_book, epochs: int = 15):
    """
    Phase 1: Supervised learning from Lichess games.
    
    Policy target: Human move
    Value target: Material evaluation (from WHITE's perspective)
    """
    print("\n📚 PHASE 1: Supervised Learning")
    print("=" * 50)
    
    # Process games
    states, actions, values = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing games"):
        try:
            board = chess.Board()
            
            for token in str(row['moves']).split():
                try:
                    move = board.parse_san(token)
                    
                    # State
                    state = encode_board(board)
                    
                    # Action
                    action = encode_move(move)
                    
                    # Value: Material eval from WHITE's perspective (consistent!)
                    value = material_eval(board)
                    
                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    states = np.array(states)
    actions = np.array(actions)
    values = np.array(values)
    
    print(f"   Positions: {len(states):,}")
    print(f"   Value range: [{values.min():.2f}, {values.max():.2f}], mean={values.mean():.2f}")
    
    # Data augmentation: horizontal flip
    print("   Augmenting data (horizontal flip)...")
    aug_states = np.flip(states, axis=3).copy()
    
    aug_actions = []
    for a in actions:
        from_sq, to_sq = a // 64, a % 64
        from_r, from_f = from_sq // 8, from_sq % 8
        to_r, to_f = to_sq // 8, to_sq % 8
        new_from = from_r * 8 + (7 - from_f)
        new_to = to_r * 8 + (7 - to_f)
        aug_actions.append(new_from * 64 + new_to)
    aug_actions = np.array(aug_actions)
    
    # Combine
    states = np.concatenate([states, aug_states])
    actions = np.concatenate([actions, aug_actions])
    values = np.concatenate([values, values])  # Values unchanged
    
    print(f"   After augmentation: {len(states):,} positions")
    
    # Training
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_supervised, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    best_wr, best_state = 0, None
    history = {'ploss': [], 'vloss': [], 'wr': []}
    
    n = len(states)
    
    for epoch in range(epochs):
        network.train()
        idx = np.random.permutation(n)
        
        ploss_sum, vloss_sum, batches = 0, 0, 0
        
        for i in range(0, n, config.batch_size):
            batch_idx = idx[i:i+config.batch_size]
            
            s = torch.FloatTensor(states[batch_idx]).to(device)
            a = torch.LongTensor(actions[batch_idx]).to(device)
            v = torch.FloatTensor(values[batch_idx]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, pred_v = network(s)
                
                ploss = F.cross_entropy(logits, a, label_smoothing=0.1)
                vloss = F.mse_loss(pred_v.squeeze(-1), v)
                loss = ploss + vloss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ploss_sum += ploss.item()
            vloss_sum += vloss.item()
            batches += 1
        
        scheduler.step()
        
        history['ploss'].append(ploss_sum / batches)
        history['vloss'].append(vloss_sum / batches)
        
        # Evaluate every 3 epochs
        if (epoch + 1) % 3 == 0:
            wr = evaluate(network, opening_book, 50)
            history['wr'].append(wr)
            
            print(f"   Epoch {epoch+1}/{epochs}: PLoss={ploss_sum/batches:.4f}, "
                  f"VLoss={vloss_sum/batches:.4f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
    
    # Restore best
    if best_state:
        network.load_state_dict(best_state)
        print(f"   ✅ Restored best model (WR={best_wr:.0%})")
    
    torch.save(network.state_dict(), '/kaggle/working/chess_v13_supervised.pt')
    
    return best_wr, history

# ==============================================================================
# Cell 10: RL Phase
# ==============================================================================

def play_rl_game(network, opening_book, epsilon: float):
    """
    Play one game: Model (White) vs Random (Black).
    Collect experiences for RL training.
    
    Value target: Stockfish eval (from WHITE's perspective)
    Reward: Change in Stockfish eval after move
    """
    board = chess.Board()
    experiences = []
    prev_eval = stockfish_eval(board)  # Start eval
    move_count = 0
    
    while not board.is_game_over() and move_count < 150:
        
        if board.turn == chess.WHITE:
            # === Our Model ===
            
            # Opening book first
            if move_count < config.opening_book_depth:
                book_move = get_book_move(board, opening_book)
                if book_move:
                    board.push(chess.Move.from_uci(book_move))
                    prev_eval = stockfish_eval(board)
                    move_count += 1
                    continue
            
            state = encode_board(board)
            mask = get_legal_mask(board)
            
            # Current eval (before our move)
            current_eval = stockfish_eval(board)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                move = random.choice(list(board.legal_moves))
                action = encode_move(move)
            else:
                action, _ = network.predict(state, mask, temperature=0.3)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
                    action = encode_move(move)
            
            board.push(move)
            
            # Eval after our move (Black to play, but we want White's perspective)
            new_eval = stockfish_eval(board)
            
            # Reward = improvement in eval
            reward = new_eval - prev_eval
            
            # Store experience
            experiences.append({
                'state': state,
                'action': action,
                'reward': reward,
                'value_target': current_eval  # Stockfish eval as value target
            })
            
            prev_eval = new_eval
        
        else:
            # === Random Opponent ===
            move = random.choice(list(board.legal_moves))
            board.push(move)
            prev_eval = stockfish_eval(board)
        
        move_count += 1
    
    # Game result bonus
    result = board.result()
    if result == '1-0':
        game_bonus = 1.0
    elif result == '0-1':
        game_bonus = -1.0
    else:
        game_bonus = 0.0
    
    # Add to buffer
    n = len(experiences)
    for i, exp in enumerate(experiences):
        # Discounted game bonus
        discount = config.gamma ** (n - i - 1)
        total_reward = exp['reward'] + game_bonus * discount * 0.5
        
        buffer.add(exp['state'], exp['action'], total_reward, exp['value_target'])
    
    return result

def rl_phase(network, opening_book, iterations: int = 50):
    """
    Phase 2: RL training with Stockfish rewards.
    """
    print("\n🤖 PHASE 2: Stockfish RL")
    print("=" * 50)
    
    buffer.clear()
    print("   Buffer cleared")
    
    # Lower LR to preserve supervised knowledge
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_rl, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Track performance
    initial_wr = evaluate(network, opening_book, 50)
    best_wr = initial_wr
    print(f"   Starting WR: {initial_wr:.0%}")
    
    history = [initial_wr]
    no_improve = 0
    
    for iteration in range(iterations):
        # Decay epsilon
        progress = iteration / iterations
        epsilon = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress
        
        # Play games
        wins, draws, losses = 0, 0, 0
        for _ in range(config.games_per_iter):
            result = play_rl_game(network, opening_book, epsilon)
            if result == '1-0':
                wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        
        # Train on buffer
        if len(buffer) >= config.batch_size:
            network.train()
            
            for _ in range(10):  # 10 updates per iteration
                states, actions, rewards, value_targets = buffer.sample(config.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                value_targets = torch.FloatTensor(value_targets).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits, pred_v = network(states)
                    
                    # Policy gradient
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    # Advantage: reward is already the advantage (eval change)
                    # Normalize for stability
                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    
                    policy_loss = -(selected_log_probs * advantages.detach()).mean()
                    
                    # Value loss: predict Stockfish eval
                    value_loss = F.mse_loss(pred_v.squeeze(-1), value_targets)
                    
                    # Entropy for exploration
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
        
        # Evaluate every 10 iterations
        if (iteration + 1) % 10 == 0:
            wr = evaluate(network, opening_book, 50)
            history.append(wr)
            
            print(f"   Iter {iteration+1}/{iterations}: W={wins} D={draws} L={losses}, "
                  f"ε={epsilon:.2f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                no_improve = 0
                torch.save(network.state_dict(), '/kaggle/working/chess_v13_best.pt')
            else:
                no_improve += 1
            
            # Early stopping
            if wr < initial_wr * 0.6:
                print(f"   ⚠️ WR dropped too much, stopping")
                break
            
            if no_improve >= 3:
                print(f"   ⏹️ No improvement for 3 evals, stopping")
                break
    
    # Restore best
    if os.path.exists('/kaggle/working/chess_v13_best.pt'):
        network.load_state_dict(torch.load('/kaggle/working/chess_v13_best.pt'))
        print(f"   ✅ Restored best model (WR={best_wr:.0%})")
    
    return best_wr, history

# ==============================================================================
# Cell 11: Evaluation
# ==============================================================================

def evaluate(network, opening_book, n_games: int = 50):
    """Evaluate model against random opponent."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 150:
            if board.turn == chess.WHITE:
                # Opening book
                if move_count < config.opening_book_depth:
                    book_move = get_book_move(board, opening_book)
                    if book_move:
                        board.push(chess.Move.from_uci(book_move))
                        move_count += 1
                        continue
                
                state = encode_board(board)
                mask = get_legal_mask(board)
                action, _ = network.predict(state, mask, temperature=0.1)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1
        
        if board.result() == '1-0':
            wins += 1
    
    return wins / n_games

# ==============================================================================
# Cell 12: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

start_time = time.time()

# Phase 1: Supervised
supervised_wr, sup_history = supervised_phase(network, df, opening_book, 
                                               epochs=config.supervised_epochs)

# Phase 2: RL
rl_best_wr, rl_history = rl_phase(network, opening_book, 
                                   iterations=config.rl_iterations)

# Final evaluation
print("\n📊 Final Evaluation (100 games)...")
final_wr = evaluate(network, opening_book, 100)
print(f"   Win Rate: {final_wr:.0%}")

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/3600:.1f} hours")

# Save final model
torch.save(network.state_dict(), '/kaggle/working/chess_v13_final.pt')
print("💾 Models saved!")

# Cleanup
if stockfish:
    stockfish.quit()

# ==============================================================================
# Cell 13: Plot Results
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Policy loss
axes[0].plot(sup_history['ploss'], 'b-')
axes[0].set_title('Policy Loss (Supervised)')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

# Value loss
axes[1].plot(sup_history['vloss'], 'g-')
axes[1].set_title('Value Loss (Supervised)')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

# Win rate
x_sup = list(range(3, len(sup_history['wr'])*3+1, 3))
x_rl = list(range(0, len(rl_history)*10, 10))

axes[2].plot(x_sup, sup_history['wr'], 'b-o', label='Supervised', markersize=4)
axes[2].axhline(supervised_wr, color='blue', linestyle='--', alpha=0.5)

if len(rl_history) > 1:
    rl_x = [0] + list(range(10, (len(rl_history))*10, 10))
    axes[2].plot([max(x_sup)] + [max(x_sup) + x for x in rl_x[1:]], 
                 rl_history, 'r-o', label='RL', markersize=4)

axes[2].axhline(0.8, color='gray', linestyle=':', alpha=0.5)
axes[2].set_title(f'Win Rate (Final: {final_wr:.0%})')
axes[2].set_xlabel('Training Progress')
axes[2].set_ylim(0, 1)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_v13.png', dpi=150)
plt.show()

print(f"\n🎉 TRAINING COMPLETE!")
print(f"   Supervised best: {supervised_wr:.0%}")
print(f"   RL best: {rl_best_wr:.0%}")
print(f"   Final: {final_wr:.0%}")
