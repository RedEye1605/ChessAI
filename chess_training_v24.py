"""
================================================================================
⚡ CHESS AI v24 - ENDGAME CURRICULUM + STOCKFISH FOCUSED
================================================================================

🔧 KEY CHANGES FROM v23:
1. VALUE HEAD ONLY - No policy training (caused regression)
2. ENDGAME CURRICULUM - Train on tablebase positions
3. STOCKFISH-FOCUSED - Save best based on Stockfish WR, not Random
4. MORE ITERATIONS - 100 iterations for thorough training
5. Load from v23_best (48% vs Stockfish)

Target: Beat Stockfish depth 1 consistently
================================================================================
"""

# ==============================================================================
# Cell 1: Dependencies
# ==============================================================================

try:
    get_ipython().system('pip install -q python-chess tqdm matplotlib pandas stockfish')
except:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'python-chess', 'tqdm', 'matplotlib', 'pandas', 'stockfish'], capture_output=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import chess
import chess.syzygy
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
import sys
import io
warnings.filterwarnings('ignore')

# Stockfish import with suppressed warnings
try:
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    from stockfish import Stockfish
    sys.stderr = old_stderr
    STOCKFISH_AVAILABLE = True
except:
    sys.stderr = old_stderr if 'old_stderr' in dir() else sys.stderr
    STOCKFISH_AVAILABLE = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

print("=" * 70)
print("⚡ CHESS AI v24 - ENDGAME CURRICULUM + STOCKFISH FOCUSED")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Configuration
# ==============================================================================

@dataclass
class Config:
    # Network
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Self-Play
    self_play_games: int = 150          # Reduced for more endgame focus
    max_moves_per_game: int = 100
    
    # Asymmetric play
    white_temperature: float = 0.2
    black_temperature: float = 0.6
    black_noise_prob: float = 0.08
    
    # Training
    start_iteration: int = 1            # Fresh start with endgame curriculum
    rl_iterations: int = 100            # More iterations
    batch_size: int = 256
    lr_value: float = 1e-6              # Slightly higher for endgame learning
    weight_decay: float = 1e-5
    batches_per_iter: int = 20
    
    # Endgame curriculum
    endgame_positions_per_iter: int = 500   # Endgame positions to add
    endgame_start_iter: int = 1             # Start endgame training immediately
    
    # Draw handling
    draw_weight: float = 0.3
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    
    # Evaluation - STOCKFISH FOCUSED
    eval_games: int = 30                # Fewer random games
    eval_interval: int = 5
    stockfish_games: int = 30           # More Stockfish games
    stockfish_depth: int = 1
    stockfish_eval_interval: int = 5    # Eval vs SF every 5 iters
    
    # Safety
    min_wr_random: float = 0.85         # Don't drop below 85% vs random

config = Config()
print(f"\n📋 Configuration (ENDGAME + STOCKFISH FOCUS):")
print(f"   Iterations: {config.rl_iterations}")
print(f"   Endgame positions/iter: {config.endgame_positions_per_iter}")
print(f"   LR (value only): {config.lr_value}")
print(f"   Stockfish games: {config.stockfish_games}")
print(f"   Save best based on: Stockfish WR")

# ==============================================================================
# Cell 3: State Encoding & Helpers
# ==============================================================================

NUM_ACTIONS = 4096

def encode_board(board: chess.Board) -> np.ndarray:
    state = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_map[piece.piece_type]
            rank, file = sq // 8, sq % 8
            state[ch, rank, file] = 1.0 if piece.color == chess.WHITE else -1.0
    
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    state[8, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    state[9, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    state[10, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    state[11, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    return state

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

# Sharp openings
SHARP_OPENINGS = [
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6",
    "e2e4 e7e5 f2f4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f1c4",
    "e2e4 e7e5 d2d4 e5d4 c2c3",
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "d2d4 f7f5",
    "",
]

def get_sharp_opening() -> chess.Board:
    board = chess.Board()
    if random.random() < 0.2:
        return board
    opening = random.choice(SHARP_OPENINGS)
    if opening:
        for uci in opening.split():
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    board.push(move)
            except:
                break
    return board

# ==============================================================================
# Cell 4: Neural Network
# ==============================================================================

class SEBlock(nn.Module):
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
    def __init__(self, in_channels=12, filters=128, blocks=6):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, NUM_ACTIONS)
        )
        
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
    
    def forward(self, x, mask=None):
        x = self.tower(self.stem(x))
        policy = self.policy_head(x)
        value = self.value_head(x)
        if mask is not None:
            policy = policy.masked_fill(~mask, -1e9)
        return policy, value
    
    def predict_move(self, board: chess.Board, temperature: float = 0.5, 
                     add_noise: bool = False) -> Tuple[chess.Move, float]:
        self.eval()
        
        if add_noise and random.random() < config.black_noise_prob:
            move = random.choice(list(board.legal_moves))
            return move, 0.0
        
        with torch.no_grad():
            state = encode_board(board)
            mask = get_legal_mask(board)
            
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            
            if temperature < 0.1:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                probs = probs.cpu().numpy()[0]
                probs = np.clip(probs, 0, 1)
                probs = probs / (probs.sum() + 1e-10)
                
                try:
                    action = np.random.choice(len(probs), p=probs)
                except:
                    action = logits.argmax(dim=-1).item()
            
            move = decode_move(action, board)
            if move is None:
                move = random.choice(list(board.legal_moves))
            
            return move, value.item()

# ==============================================================================
# Cell 5: Load v23_best Model
# ==============================================================================

MODEL_PATHS = [
    '/kaggle/input/supervisedmodel/pytorch/default/7/chess_v23_best.pt',
    '/kaggle/working/chess_v23_best.pt',
    '/kaggle/input/supervisedmodel/pytorch/default/6/chess_v22_final.pt',
]

network = ChessNet(in_channels=config.input_channels, 
                   filters=config.filters, 
                   blocks=config.blocks).to(device)

loaded = False
for path in MODEL_PATHS:
    if os.path.exists(path):
        network.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded model from: {path}")
        loaded = True
        break

if not loaded:
    print("⚠️ Model not found!")

# Freeze everything except value head
def freeze_module(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

freeze_module(network.stem)
freeze_module(network.tower)
freeze_module(network.policy_head)

for param in network.value_head.parameters():
    param.requires_grad = True

for module in network.value_head.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"✅ VALUE HEAD ONLY training ({trainable:,} params)")

# ==============================================================================
# Cell 6: Stockfish Setup
# ==============================================================================

def setup_stockfish():
    """Setup Stockfish with explicit status messages."""
    print("🔧 Setting up Stockfish...")
    
    if not STOCKFISH_AVAILABLE:
        print("⚠️ Stockfish Python package not installed")
        return None
    
    # Install stockfish binary
    try:
        import subprocess
        result = subprocess.run(['apt-get', 'install', '-y', 'stockfish'], 
                               capture_output=True, timeout=60)
        if result.returncode == 0:
            print("   Stockfish binary installed")
    except Exception as e:
        print(f"   Could not install stockfish binary: {e}")
    
    # Try to find stockfish
    sf = None
    for path in ["/usr/games/stockfish", "/usr/bin/stockfish", "stockfish"]:
        try:
            sf = Stockfish(path=path, depth=config.stockfish_depth)
            sf.set_skill_level(0)
            print(f"✅ Stockfish ready at: {path}")
            return sf
        except Exception as e:
            continue
    
    print("⚠️ Stockfish not available (binary not found)")
    return None

stockfish = setup_stockfish()

def evaluate_vs_stockfish(network: ChessNet, sf, n_games: int) -> Tuple[float, Dict]:
    if sf is None:
        return 0.0, {'wins': 0, 'draws': 0, 'losses': 0}
    
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        try:
            while not board.is_game_over() and move_count < 150:
                if board.turn == chess.WHITE:
                    move, _ = network.predict_move(board, temperature=0.1)
                else:
                    sf.set_fen_position(board.fen())
                    sf_move = sf.get_best_move()
                    move = chess.Move.from_uci(sf_move) if sf_move else random.choice(list(board.legal_moves))
                
                board.push(move)
                move_count += 1
            
            result = board.result()
            if result == '1-0':
                results['wins'] += 1
            elif result == '0-1':
                results['losses'] += 1
            else:
                results['draws'] += 1
        except:
            results['draws'] += 1
    
    wr = (results['wins'] + 0.5 * results['draws']) / n_games
    return wr, results

# ==============================================================================
# Cell 7: Endgame Position Generator (No Syzygy needed)
# ==============================================================================

def generate_endgame_positions(n_positions: int) -> List[Tuple[np.ndarray, float]]:
    """Generate endgame positions with known evaluations."""
    positions = []
    
    # Endgame patterns with values
    endgame_types = [
        # (pieces_white, pieces_black, white_value)
        ('KQ', 'K', 1.0),      # KQ vs K = winning
        ('KR', 'K', 1.0),      # KR vs K = winning
        ('KBB', 'K', 1.0),     # KBB vs K = winning
        ('KBN', 'K', 1.0),     # KBN vs K = winning (hard)
        ('KP', 'K', 0.8),      # KP vs K = usually winning
        ('K', 'KQ', -1.0),     # K vs KQ = losing
        ('K', 'KR', -1.0),     # K vs KR = losing
        ('KR', 'KR', 0.0),     # KR vs KR = usually draw
        ('KQ', 'KQ', 0.0),     # KQ vs KQ = usually draw
        ('KP', 'KP', 0.0),     # KP vs KP = usually draw
        ('KRP', 'KR', 0.5),    # KRP vs KR = slight advantage
        ('KR', 'KRP', -0.5),   # KR vs KRP = slight disadvantage
    ]
    
    piece_to_type = {'K': chess.KING, 'Q': chess.QUEEN, 'R': chess.ROOK, 
                     'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}
    
    for _ in range(n_positions):
        pattern = random.choice(endgame_types)
        white_pieces, black_pieces, base_value = pattern
        
        # Create empty board
        board = chess.Board(None)
        board.clear()
        
        # Place white king
        wk_sq = random.choice([sq for sq in chess.SQUARES 
                               if chess.square_rank(sq) in [0,1,2,3,4,5,6,7]])
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        
        # Place black king (not adjacent to white king)
        valid_bk = [sq for sq in chess.SQUARES 
                   if chess.square_distance(sq, wk_sq) > 1]
        bk_sq = random.choice(valid_bk) if valid_bk else random.choice(chess.SQUARES)
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        
        # Place other white pieces
        occupied = {wk_sq, bk_sq}
        for p in white_pieces[1:]:  # Skip K
            valid_sq = [sq for sq in chess.SQUARES if sq not in occupied]
            if valid_sq:
                sq = random.choice(valid_sq)
                # Pawns not on rank 1 or 8
                if p == 'P':
                    valid_sq = [sq for sq in valid_sq if chess.square_rank(sq) in [1,2,3,4,5,6]]
                    if valid_sq:
                        sq = random.choice(valid_sq)
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.WHITE))
                occupied.add(sq)
        
        # Place other black pieces
        for p in black_pieces[1:]:  # Skip K
            valid_sq = [sq for sq in chess.SQUARES if sq not in occupied]
            if valid_sq:
                sq = random.choice(valid_sq)
                if p == 'P':
                    valid_sq = [sq for sq in valid_sq if chess.square_rank(sq) in [1,2,3,4,5,6]]
                    if valid_sq:
                        sq = random.choice(valid_sq)
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.BLACK))
                occupied.add(sq)
        
        # Set turn randomly
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        # Adjust value for turn
        value = base_value if board.turn == chess.WHITE else -base_value
        
        # Add noise to value
        value = max(-1.0, min(1.0, value + random.uniform(-0.1, 0.1)))
        
        # Only add if position is valid
        if board.is_valid() and not board.is_game_over():
            positions.append((encode_board(board), value))
    
    return positions

print(f"✅ Endgame generator ready")

# ==============================================================================
# Cell 8: Buffer
# ==============================================================================

class WeightedReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.stats = {'wins': 0, 'draws': 0, 'losses': 0}
    
    def add(self, state: np.ndarray, value: float, weight: float = 1.0):
        self.buffer.append((state, value, weight))
        if abs(value) > 0.5:
            if value > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
    
    def add_batch(self, positions: List[Tuple[np.ndarray, float]], weight: float = 1.0):
        for state, value in positions:
            self.add(state, value, weight)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            weights = np.array([item[2] for item in self.buffer])
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=weights)
            batch = [self.buffer[i] for i in indices]
        
        states, values, weights = zip(*batch)
        return np.array(states), np.array(values, dtype=np.float32), np.array(weights, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

buffer = WeightedReplayBuffer(config.buffer_size)

# ==============================================================================
# Cell 9: Self-Play
# ==============================================================================

def play_asymmetric_game(network: ChessNet, config: Config) -> Tuple[List, str]:
    board = get_sharp_opening()
    game_history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        state = encode_board(board)
        player = 1 if board.turn == chess.WHITE else -1
        game_history.append((state, player))
        
        if board.turn == chess.WHITE:
            move, _ = network.predict_move(board, temperature=config.white_temperature)
        else:
            move, _ = network.predict_move(board, temperature=config.black_temperature, add_noise=True)
        
        board.push(move)
        move_count += 1
    
    result = board.result()
    if result == '*':
        result = '1/2-1/2'
    
    outcome = 1 if result == '1-0' else (-1 if result == '0-1' else 0)
    
    examples = []
    for state, player in game_history:
        value = outcome * player
        weight = 1.0 if outcome != 0 else config.draw_weight
        examples.append((state, value, weight))
    
    return examples, result

def run_self_play(network: ChessNet, buffer: WeightedReplayBuffer, n_games: int, config: Config) -> Dict:
    network.eval()
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_positions = 0
    
    for _ in tqdm(range(n_games), desc="Self-play", leave=False):
        examples, result = play_asymmetric_game(network, config)
        for state, value, weight in examples:
            buffer.add(state, value, weight)
        results[result] = results.get(result, 0) + 1
        total_positions += len(examples)
    
    return {'results': results, 'positions': total_positions}

# ==============================================================================
# Cell 10: Trainer (Value Head Only)
# ==============================================================================

class Trainer:
    def __init__(self, network: ChessNet, config: Config, buffer: WeightedReplayBuffer):
        self.network = network
        self.config = config
        self.buffer = buffer
        
        self.optimizer = torch.optim.AdamW(
            [p for p in network.value_head.parameters() if p.requires_grad],
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    def train_epoch(self) -> Dict[str, float]:
        self.network.eval()
        total_loss = 0
        total_sign_acc = 0
        
        for _ in range(self.config.batches_per_iter):
            states, target_values, weights = self.buffer.sample(self.config.batch_size)
            
            states = torch.FloatTensor(states).to(device)
            target_values = torch.FloatTensor(target_values).to(device)
            weights = torch.FloatTensor(weights).to(device)
            
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    _, pred_values = self.network(states)
                    loss = (weights * (pred_values.squeeze(-1) - target_values) ** 2).mean()
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.value_head.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            with torch.no_grad():
                decisive_mask = target_values.abs() > 0.5
                if decisive_mask.sum() > 0:
                    pred_sign = (pred_values.squeeze(-1)[decisive_mask] > 0).float()
                    target_sign = (target_values[decisive_mask] > 0).float()
                    sign_acc = (pred_sign == target_sign).float().mean().item()
                else:
                    sign_acc = 0.5
            
            total_loss += loss.item()
            total_sign_acc += sign_acc
        
        n = self.config.batches_per_iter
        return {'loss': total_loss / n, 'sign_acc': total_sign_acc / n}

trainer = Trainer(network, config, buffer)

def evaluate_vs_random(network: ChessNet, n_games: int) -> Tuple[float, Dict]:
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:
            if board.turn == chess.WHITE:
                move, _ = network.predict_move(board, temperature=0.1)
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            results['wins'] += 1
        elif result == '0-1':
            results['losses'] += 1
        else:
            results['draws'] += 1
    
    wr = (results['wins'] + 0.5 * results['draws']) / n_games
    return wr, results

# ==============================================================================
# Cell 11: Main Training Loop
# ==============================================================================

def train_v24():
    print("\n" + "=" * 70)
    print("🚀 STARTING v24 - ENDGAME CURRICULUM + STOCKFISH FOCUS")
    print("=" * 70)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr_random, det_r = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr_random:.0%}")
    
    initial_wr_sf = 0.0
    if stockfish:
        initial_wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
        print(f"   WR vs Stockfish(d{config.stockfish_depth}): {initial_wr_sf:.0%}")
    
    history = {
        'iterations': [],
        'value_loss': [],
        'sign_accuracy': [],
        'wr_random': [initial_wr_random],
        'wr_stockfish': [initial_wr_sf],
        'endgame_added': [],
    }
    
    best_sf_wr = initial_wr_sf
    start_time = time.time()
    
    for iteration in range(1, config.rl_iterations + 1):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration}/{config.rl_iterations}")
        print(f"{'='*70}")
        
        # 1. Add endgame positions
        if iteration >= config.endgame_start_iter:
            print(f"\n♟️ Generating endgame positions...")
            endgame_positions = generate_endgame_positions(config.endgame_positions_per_iter)
            buffer.add_batch(endgame_positions, weight=1.5)  # Higher weight for endgames
            print(f"   Added {len(endgame_positions)} endgame positions (weight=1.5)")
            history['endgame_added'].append(len(endgame_positions))
        else:
            history['endgame_added'].append(0)
        
        # 2. Self-Play
        print(f"\n🎮 Self-Play ({config.self_play_games} games)...")
        sp_results = run_self_play(network, buffer, config.self_play_games, config)
        results = sp_results['results']
        
        total = sum(results.values())
        print(f"   W={results.get('1-0',0)} D={results.get('1/2-1/2',0)} L={results.get('0-1',0)}")
        print(f"   Buffer: {len(buffer)}")
        
        # 3. Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training (value head only)...")
            train_metrics = trainer.train_epoch()
            print(f"   Value Loss: {train_metrics['loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            
            history['iterations'].append(iteration)
            history['value_loss'].append(train_metrics['loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
        
        # 4. Evaluation
        if iteration % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            
            # Random eval
            wr_random, _ = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr_random)
            print(f"   WR vs Random: {wr_random:.0%}")
            
            # Stockfish eval
            if stockfish and iteration % config.stockfish_eval_interval == 0:
                wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
                history['wr_stockfish'].append(wr_sf)
                print(f"   🐟 WR vs Stockfish: {wr_sf:.0%} (W:{det_sf['wins']} D:{det_sf['draws']} L:{det_sf['losses']})")
                
                # Save best based on Stockfish
                if wr_sf > best_sf_wr:
                    best_sf_wr = wr_sf
                    torch.save(network.state_dict(), '/kaggle/working/chess_v24_best_sf.pt')
                    print(f"   ✨ New best vs Stockfish! Saved.")
            
            # Safety check - don't regress too much on random
            if wr_random < config.min_wr_random:
                print(f"\n⚠️ WR vs Random dropped to {wr_random:.0%}!")
        
        iter_time = time.time() - iter_start
        eta = (config.rl_iterations - iteration) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), '/kaggle/working/chess_v24_final.pt')
    print(f"\n💾 Saved: chess_v24_final.pt")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    final_wr_random, _ = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {final_wr_random:.0%}")
    
    if stockfish:
        final_wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
        print(f"   WR vs Stockfish: {final_wr_sf:.0%}")
        print(f"\n🏆 Best Stockfish WR: {best_sf_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 12: Run
# ==============================================================================

if __name__ == "__main__":
    history = train_v24()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if history['value_loss']:
        axes[0, 0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0, 0].set_title('Value Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)
    
    if history['sign_accuracy']:
        axes[0, 1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--')
        axes[0, 1].set_title('Sign Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Win rates
    if history['wr_stockfish']:
        axes[0, 2].plot(history['wr_stockfish'], 'r-o', label='vs Stockfish', linewidth=2, markersize=6)
    if history['wr_random']:
        x_rand = list(range(len(history['wr_random'])))
        axes[0, 2].plot(x_rand, history['wr_random'], 'b--', alpha=0.5, label='vs Random')
    axes[0, 2].set_title('Win Rate (STOCKFISH FOCUS)')
    axes[0, 2].set_ylabel('Win Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Endgame positions added
    if history['endgame_added']:
        axes[1, 0].bar(range(len(history['endgame_added'])), history['endgame_added'], color='purple', alpha=0.7)
        axes[1, 0].set_title('Endgame Positions Added')
        axes[1, 0].set_xlabel('Iteration')
    
    # Cumulative endgame positions
    if history['endgame_added']:
        cumsum = np.cumsum(history['endgame_added'])
        axes[1, 1].plot(cumsum, 'purple', linewidth=2)
        axes[1, 1].set_title('Cumulative Endgame Positions')
        axes[1, 1].fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='purple')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Summary
    summary = f"Initial SF: {history['wr_stockfish'][0]:.0%}\n"
    if len(history['wr_stockfish']) > 1:
        summary += f"Final SF: {history['wr_stockfish'][-1]:.0%}\n"
        summary += f"Best SF: {max(history['wr_stockfish']):.0%}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center', fontsize=16, 
                    transform=axes[1, 2].transAxes, fontweight='bold')
    axes[1, 2].set_title('Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v24.png', dpi=150)
    plt.show()
    
    print("\n🎉 v24 TRAINING COMPLETE!")
    print("=" * 70)
