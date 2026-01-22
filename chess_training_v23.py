"""
================================================================================
⚡ CHESS AI v23 - GENTLE POLICY TRAINING + STOCKFISH EVAL
================================================================================

🔧 IMPROVEMENTS FROM v22:
1. Gentle policy head unfreezing with ultra-low LR (1e-8)
2. Policy warmup period (first 20 iters = value only)
3. Stockfish evaluation benchmark
4. Fixed logging bug (games not adding to total)
5. More games per iteration (200)
6. Safety: auto-freeze policy if WR drops

Training time: ~2-3 hours on P100
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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import stockfish
try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except:
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
print("⚡ CHESS AI v23 - GENTLE POLICY + STOCKFISH")
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
    
    # Self-Play - More games for better signal
    self_play_games: int = 200          # Increased from 100
    max_moves_per_game: int = 100
    
    # Asymmetric play
    white_temperature: float = 0.2      # Stronger white (was 0.3)
    black_temperature: float = 0.6      # More realistic black (was 0.8)
    black_noise_prob: float = 0.08      # Slightly less noise
    
    # Training - CONTINUE FROM v23 iteration 170
    start_iteration: int = 171          # Continue from v23_best (iter 170)
    rl_iterations: int = 30             # 30 more iterations (171-200)
    batch_size: int = 256
    
    # Two-phase LR
    lr_value: float = 5e-7              # Same as v22
    lr_policy: float = 1e-8             # ULTRA LOW for policy
    weight_decay: float = 1e-5
    batches_per_iter: int = 20
    
    # Policy warmup (0 since we're already past warmup)
    policy_warmup: int = 0              # Already past warmup, start policy immediately
    policy_update_interval: int = 3     # Update policy every 3 iters after warmup
    
    # Safety thresholds
    min_wr_threshold: float = 0.85      # Stop if WR drops below 85%
    wr_drop_threshold: float = 0.05     # Freeze policy if WR drops 5%+
    
    # Draw handling
    draw_weight: float = 0.3
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    
    # Evaluation
    eval_games: int = 50
    eval_interval: int = 5
    stockfish_games: int = 20           # Games vs Stockfish
    stockfish_depth: int = 1            # Stockfish search depth

config = Config()
print(f"\n📋 Configuration (GENTLE POLICY + STOCKFISH):")
print(f"   Starting from iteration: {config.start_iteration}")
print(f"   Games per iteration: {config.self_play_games}")
print(f"   Value LR: {config.lr_value}")
print(f"   Policy LR: {config.lr_policy} (100x lower!)")
print(f"   Policy warmup: {config.policy_warmup} iterations")
print(f"   Stockfish eval: {'✅ Enabled' if STOCKFISH_AVAILABLE else '❌ Not available'}")

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
    
    def get_policy_entropy(self, board: chess.Board) -> float:
        """Calculate policy entropy for monitoring."""
        self.eval()
        with torch.no_grad():
            state = encode_board(board)
            mask = get_legal_mask(board)
            
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, _ = self(x, m)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            
        return entropy

# ==============================================================================
# Cell 5: Load v22 Model
# ==============================================================================

# Load from v23_best (saved at iteration 170)
MODEL_PATHS = [
    '/kaggle/input/supervisedmodel/pytorch/default/7/chess_v23_best.pt',  # Primary - v23 best from iter 170
    '/kaggle/input/supervisedmodel/pytorch/default/6/chess_v22_final.pt',
    '/kaggle/working/chess_v23_best.pt',
    '/kaggle/working/chess_v22_final.pt',
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
    print("⚠️ Model not found! Searching...")
    for search_dir in ['/kaggle/input', '/kaggle/working']:
        if os.path.exists(search_dir):
            print(f"   Files in {search_dir}:")
            for f in os.listdir(search_dir):
                print(f"      {f}")

# ==============================================================================
# Cell 6: Setup Training (Two-Phase)
# ==============================================================================

def freeze_module_completely(module: nn.Module):
    """Freeze weights AND BatchNorm running stats."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

def setup_value_only_training(network: ChessNet):
    """Phase 1: Only train value head."""
    freeze_module_completely(network.stem)
    freeze_module_completely(network.tower)
    freeze_module_completely(network.policy_head)
    
    for param in network.value_head.parameters():
        param.requires_grad = True
    
    for module in network.value_head.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

def setup_gentle_both_training(network: ChessNet):
    """Phase 2: Train both heads, policy very gently."""
    freeze_module_completely(network.stem)
    freeze_module_completely(network.tower)
    
    # Value head - trainable
    for param in network.value_head.parameters():
        param.requires_grad = True
    
    # Policy head - only Linear layers, NOT BatchNorm
    for name, module in network.policy_head.named_modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True
        elif isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    # Keep all BatchNorm in eval
    for module in network.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

# Start with value-only
trainable = setup_value_only_training(network)
print(f"✅ Phase 1: Value-only training ({trainable:,} params)")
print(f"   Policy training will start after iteration {config.start_iteration + config.policy_warmup - 1}")

# ==============================================================================
# Cell 7: Stockfish Evaluation
# ==============================================================================

def setup_stockfish():
    """Try to setup Stockfish engine with suppressed errors."""
    if not STOCKFISH_AVAILABLE:
        return None
    
    import sys
    import io
    
    # Suppress stderr during stockfish init attempts
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    sf = None
    
    # Try to install stockfish first (most reliable on Kaggle)
    try:
        import subprocess
        subprocess.run(['apt-get', 'install', '-y', 'stockfish'], 
                      capture_output=True, check=True)
    except:
        pass
    
    # Now try to find it
    stockfish_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/kaggle/working/stockfish",
        "stockfish"
    ]
    
    for path in stockfish_paths:
        try:
            sf = Stockfish(path=path, depth=config.stockfish_depth)
            sf.set_skill_level(0)
            break
        except:
            sf = None
            continue
    
    # Restore stderr
    sys.stderr = old_stderr
    
    if sf:
        print("✅ Stockfish ready")
        return sf
    else:
        print("⚠️ Stockfish not available")
        return None

stockfish = setup_stockfish()

def evaluate_vs_stockfish(network: ChessNet, stockfish, n_games: int) -> Tuple[float, Dict]:
    """Evaluate network against Stockfish."""
    if stockfish is None:
        return 0.0, {'wins': 0, 'draws': 0, 'losses': 0, 'error': 'no stockfish'}
    
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        try:
            while not board.is_game_over() and move_count < 150:
                if board.turn == chess.WHITE:
                    # Network plays white
                    move, _ = network.predict_move(board, temperature=0.1)
                else:
                    # Stockfish plays black
                    stockfish.set_fen_position(board.fen())
                    sf_move = stockfish.get_best_move()
                    if sf_move:
                        move = chess.Move.from_uci(sf_move)
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
        except Exception as e:
            results['draws'] += 1  # Count errors as draws
    
    wr = (results['wins'] + 0.5 * results['draws']) / n_games
    return wr, results

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
    
    def get_win_ratio(self) -> float:
        total = self.stats['wins'] + self.stats['losses']
        return self.stats['wins'] / total if total > 0 else 0
    
    def __len__(self):
        return len(self.buffer)

buffer = WeightedReplayBuffer(config.buffer_size)

# ==============================================================================
# Cell 9: Self-Play (with fixed logging)
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
    
    # FIX: Handle unfinished games (result = '*')
    if result == '*':
        result = '1/2-1/2'  # Count as draw
    
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
    
    return {'results': results, 'positions': total_positions, 'win_ratio': buffer.get_win_ratio()}

# ==============================================================================
# Cell 10: Trainer (Two optimizers)
# ==============================================================================

class Trainer:
    """Trainer with single optimizer using parameter groups (fixes GradScaler issue)."""
    
    def __init__(self, network: ChessNet, config: Config, buffer: WeightedReplayBuffer):
        self.network = network
        self.config = config
        self.buffer = buffer
        self.policy_trainable = False
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        
        # Start with value-only optimizer
        self._create_value_only_optimizer()
    
    def _create_value_only_optimizer(self):
        """Create optimizer for value head only."""
        value_params = [p for p in self.network.value_head.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            value_params,
            lr=self.config.lr_value,
            weight_decay=self.config.weight_decay
        )
        self.policy_trainable = False
    
    def _create_combined_optimizer(self):
        """Create single optimizer with parameter groups for both heads."""
        # Value head params
        value_params = [p for p in self.network.value_head.parameters() if p.requires_grad]
        
        # Policy head Linear params only
        policy_params = []
        for name, module in self.network.policy_head.named_modules():
            if isinstance(module, nn.Linear):
                policy_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # Create optimizer with parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': value_params, 'lr': self.config.lr_value},
            {'params': policy_params, 'lr': self.config.lr_policy},
        ], weight_decay=self.config.weight_decay)
        
        self.policy_trainable = True
    
    def enable_policy_training(self):
        """Enable policy training (called after warmup)."""
        if self.policy_trainable:
            return
        
        trainable = setup_gentle_both_training(self.network)
        self._create_combined_optimizer()
        print(f"✅ Policy training enabled ({trainable:,} trainable params)")
    
    def disable_policy_training(self):
        """Disable policy training (safety fallback)."""
        if not self.policy_trainable:
            return
        
        # Freeze policy head
        for param in self.network.policy_head.parameters():
            param.requires_grad = False
        
        # Recreate value-only optimizer
        self._create_value_only_optimizer()
        print("⚠️ Policy training disabled (safety)")
    
    def train_epoch(self, train_policy: bool = False) -> Dict[str, float]:
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
            
            # Clip gradients for all trainable params
            trainable_params = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
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

def train_v23():
    print("\n" + "=" * 70)
    print(f"🚀 STARTING v23 - GENTLE POLICY TRAINING")
    print("=" * 70)
    
    print("\n📊 Initial Evaluation:")
    initial_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    if stockfish:
        sf_wr, sf_details = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
        print(f"   WR vs Stockfish(d{config.stockfish_depth}): {sf_wr:.0%} (W:{sf_details['wins']} D:{sf_details['draws']} L:{sf_details['losses']})")
    
    history = {
        'iterations': [],
        'value_loss': [],
        'sign_accuracy': [],
        'wr_random': [initial_wr],
        'wr_stockfish': [],
        'white_wins': [],
        'black_wins': [],
        'draws': [],
        'policy_trained': [],
    }
    
    best_wr = initial_wr
    last_wr = initial_wr
    policy_frozen_due_to_drop = False
    start_time = time.time()
    
    for i in range(config.rl_iterations):
        iteration = config.start_iteration + i
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration}/{config.start_iteration + config.rl_iterations - 1}")
        print(f"{'='*70}")
        
        # Check if should enable policy training
        policy_phase = i >= config.policy_warmup and not policy_frozen_due_to_drop
        should_train_policy = policy_phase and (i - config.policy_warmup) % config.policy_update_interval == 0
        
        if policy_phase and not trainer.policy_trainable:
            trainer.enable_policy_training()
        
        # Self-Play
        print(f"\n🎮 Self-Play ({config.self_play_games} games)...")
        sp_results = run_self_play(network, buffer, config.self_play_games, config)
        results = sp_results['results']
        
        total = sum(results.values())
        win_pct = results.get('1-0', 0) / max(1, total)
        draw_pct = results.get('1/2-1/2', 0) / max(1, total)
        loss_pct = results.get('0-1', 0) / max(1, total)
        
        # FIXED: Now total should equal self_play_games
        print(f"   W={results.get('1-0',0)} ({win_pct:.0%}) D={results.get('1/2-1/2',0)} ({draw_pct:.0%}) L={results.get('0-1',0)} ({loss_pct:.0%})")
        print(f"   Total: {total}/{config.self_play_games} | Buffer: {len(buffer)}")
        
        history['white_wins'].append(results.get('1-0', 0))
        history['black_wins'].append(results.get('0-1', 0))
        history['draws'].append(results.get('1/2-1/2', 0))
        
        # Training
        if len(buffer) >= config.min_buffer_size:
            mode = "policy+value" if should_train_policy else "value only"
            print(f"\n📚 Training ({mode})...")
            train_metrics = trainer.train_epoch(train_policy=should_train_policy)
            print(f"   Value Loss: {train_metrics['loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            
            history['iterations'].append(iteration)
            history['value_loss'].append(train_metrics['loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
            history['policy_trained'].append(1 if should_train_policy else 0)
        
        # Evaluation
        if (i + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr, details = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr)
            
            delta = (wr - initial_wr) * 100
            status = "✅" if wr >= initial_wr - 0.02 else "⚠️"
            print(f"   {status} WR vs Random: {wr:.0%} ({delta:+.1f}%)")
            
            # Stockfish eval
            if stockfish and (i + 1) % (config.eval_interval * 2) == 0:
                sf_wr, sf_details = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
                history['wr_stockfish'].append(sf_wr)
                print(f"   🐟 WR vs Stockfish: {sf_wr:.0%}")
            
            # Check for WR drop -> freeze policy
            if policy_phase and (last_wr - wr) >= config.wr_drop_threshold:
                print(f"\n⚠️ WR dropped {(last_wr - wr)*100:.1f}%! Freezing policy.")
                trainer.disable_policy_training()
                policy_frozen_due_to_drop = True
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v23_best.pt')
                print(f"   ✨ New best! Saved.")
            
            if wr < config.min_wr_threshold:
                print(f"\n🛑 WR dropped below {config.min_wr_threshold:.0%}! Stopping.")
                break
            
            last_wr = wr
        
        iter_time = time.time() - iter_start
        eta = (config.rl_iterations - i - 1) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | ETA: {eta/60:.1f}min")
    
    torch.save(network.state_dict(), '/kaggle/working/chess_v23_final.pt')
    print(f"\n💾 Saved: chess_v23_final.pt")
    
    final_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"\n📊 Final: WR {final_wr:.0%}")
    
    if stockfish:
        sf_wr, sf_details = evaluate_vs_stockfish(network, stockfish, config.stockfish_games)
        print(f"📊 Final vs Stockfish: {sf_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 12: Run
# ==============================================================================

if __name__ == "__main__":
    history = train_v23()
    
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
    
    if history['wr_random']:
        axes[0, 2].plot(history['wr_random'], 'b-o', label='vs Random')
        if history['wr_stockfish']:
            x_sf = list(range(0, len(history['wr_stockfish']) * 2, 2))
            axes[0, 2].plot(x_sf, history['wr_stockfish'], 'r-s', label='vs Stockfish')
        axes[0, 2].set_title('Win Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    if history['white_wins']:
        x = range(len(history['white_wins']))
        axes[1, 0].stackplot(x, history['white_wins'], history['draws'], history['black_wins'],
                            labels=['White', 'Draw', 'Black'], colors=['green', 'gray', 'red'], alpha=0.8)
        axes[1, 0].set_title('Self-Play Results')
        axes[1, 0].legend(loc='upper right')
    
    if history['policy_trained']:
        axes[1, 1].bar(range(len(history['policy_trained'])), history['policy_trained'], 
                       color=['blue' if p else 'gray' for p in history['policy_trained']])
        axes[1, 1].set_title('Policy Training (blue=trained)')
        axes[1, 1].set_xlabel('Iteration')
    
    # Summary
    summary = f"Initial: {history['wr_random'][0]:.0%}\nFinal: {history['wr_random'][-1]:.0%}"
    if history['wr_stockfish']:
        summary += f"\nSF Final: {history['wr_stockfish'][-1]:.0%}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center', fontsize=14, 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v23.png', dpi=150)
    plt.show()
    
    print("\n🎉 v23 TRAINING COMPLETE!")
