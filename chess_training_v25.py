"""
================================================================================
⚡ CHESS AI v25 - SAFE POLICY TRAINING + VS STOCKFISH
================================================================================

🔧 KEY FEATURES:
1. SELF-PLAY VS STOCKFISH - High quality training data
2. SAFE POLICY TRAINING - Only after warmup, ultra-low LR
3. ULTRA LOW POLICY LR - 1e-9 (basically frozen but learning)
4. AUTO-FREEZE - Stop policy training if WR drops
5. GRADUAL - Policy update every 5 iterations only

Safety measures:
- Policy only learns from games vs Stockfish (high quality)
- LR 1e-9 for policy (minimal drift)
- Monitor WR every 5 iterations
- Auto-freeze policy if WR drops 5%+

Target: Stable improvement without collapse
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
import sys
import io
warnings.filterwarnings('ignore')

# Stockfish import
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

# Save directory - works on both Kaggle and local
SAVE_DIR = '/kaggle/working' if os.path.exists('/kaggle') else '.'

print("=" * 70)
print("⚡ CHESS AI v25 - SELF-PLAY VS STOCKFISH")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")
print(f"✅ Save Dir: {SAVE_DIR}")


# ==============================================================================
# Cell 2: Configuration
# ==============================================================================

@dataclass
class Config:
    # Network
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Games vs Stockfish
    games_vs_stockfish: int = 100
    max_moves_per_game: int = 150
    
    # Model temperature
    model_temperature: float = 0.3
    
    # Training
    rl_iterations: int = 100
    batch_size: int = 256
    
    # Learning rates - ULTRA CONSERVATIVE
    lr_value: float = 5e-7              # Value head LR
    lr_policy: float = 1e-9             # Policy head LR (basically frozen)
    weight_decay: float = 1e-5
    batches_per_iter: int = 25
    
    # Policy training safety
    policy_update_interval: int = 5     # Update policy every 5 iterations
    policy_warmup: int = 10             # No policy training for first 10 iters
    wr_drop_threshold: float = 0.05     # Freeze policy if WR drops 5%
    min_win_games_for_policy: int = 5   # Need at least 5 wins to train policy
    
    # Stockfish settings
    stockfish_skill: int = 0
    stockfish_depth: int = 1
    skill_up_threshold: float = 0.65
    
    # Endgame curriculum
    endgame_positions_per_iter: int = 300
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    
    # Separate buffer for policy (only winning games)
    policy_buffer_size: int = 20000
    
    # Evaluation
    eval_games: int = 30
    eval_interval: int = 5
    stockfish_eval_games: int = 50

config = Config()
print(f"\n📋 Configuration (SAFE POLICY TRAINING):")
print(f"   Value LR: {config.lr_value}")
print(f"   Policy LR: {config.lr_policy} (ultra low!)")
print(f"   Policy update: every {config.policy_update_interval} iters")
print(f"   Policy warmup: {config.policy_warmup} iters")
print(f"   Auto-freeze threshold: {config.wr_drop_threshold*100:.0f}% WR drop")

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
            # -1e4 is safe for float16 and small enough for softmax to be effectively 0
            policy = policy.masked_fill(~mask, -1e4)
        return policy, value
    
    def predict_move(self, board: chess.Board, temperature: float = 0.5) -> Tuple[chess.Move, float]:
        self.eval()
        
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
# Cell 5: Load v24 Model
# ==============================================================================

MODEL_PATHS = [
    '/kaggle/input/supervisedmodel/pytorch/default/8/chess_v24_final.pt',  # Primary
    '/kaggle/input/supervisedmodel/pytorch/default/8/chess_v24_best_sf.pt',
    '/kaggle/working/chess_v24_final.pt',
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
    for d in ['/kaggle/input', '/kaggle/working']:
        if os.path.exists(d):
            for f in os.listdir(d):
                print(f"   {d}/{f}")

# Freeze helper function
def freeze_module(module: nn.Module):
    """Freeze module weights and BatchNorm."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

# Setup for safe policy training
def setup_safe_training(network: ChessNet):
    """Setup network for safe policy + value training."""
    # Freeze stem and tower
    freeze_module(network.stem)
    freeze_module(network.tower)
    
    # Value head - trainable
    for param in network.value_head.parameters():
        param.requires_grad = True
    for module in network.value_head.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    
    # Policy head - only Linear layers trainable, BatchNorm frozen
    for name, module in network.policy_head.named_modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True
        elif isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return trainable

# Initially freeze everything, enable policy later
freeze_module(network.stem)
freeze_module(network.tower)
freeze_module(network.policy_head)

for param in network.value_head.parameters():
    param.requires_grad = True
for module in network.value_head.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"✅ Initial: VALUE HEAD ONLY ({trainable:,} params)")
print(f"   Policy training will be enabled after warmup ({config.policy_warmup} iters)")

# ==============================================================================
# Cell 6: Stockfish Setup
# ==============================================================================

def setup_stockfish(skill: int = 0, depth: int = 1):
    """Setup Stockfish with explicit status messages."""
    print(f"🔧 Setting up Stockfish (skill={skill}, depth={depth})...")
    
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
            sf = Stockfish(path=path, depth=depth)
            sf.set_skill_level(skill)
            print(f"✅ Stockfish ready at: {path}")
            return sf
        except Exception as e:
            continue
    
    print("⚠️ Stockfish not available (binary not found)")
    return None

stockfish = setup_stockfish(config.stockfish_skill, config.stockfish_depth)

# ==============================================================================
# Cell 7: Play vs Stockfish (Training Data Generator)
# ==============================================================================

def play_game_vs_stockfish(network: ChessNet, sf: Stockfish, config: Config, 
                           model_plays_white: bool = True) -> Tuple[List, List, str, int]:
    """Play one game against Stockfish and collect training data.
    
    Returns:
        value_examples: List of (state, value, weight) for value training
        policy_examples: List of (state, action, mask) for policy training (only model moves)
        result: Game result string
        outcome: +1 win, 0 draw, -1 loss from model's perspective
    """
    board = chess.Board()
    value_history = []    # (state, player_sign)
    policy_history = []   # (state, action, mask) - only model's moves
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        state = encode_board(board)
        is_model_turn = (board.turn == chess.WHITE) == model_plays_white
        
        if is_model_turn:
            # Model's turn - record position for value training
            value_history.append((state.copy(), 1 if board.turn == chess.WHITE else -1))
            
            # Get move and store for policy training
            mask = get_legal_mask(board)
            move, _ = network.predict_move(board, temperature=config.model_temperature)
            action = encode_move(move)
            policy_history.append((state.copy(), action, mask.copy()))
        else:
            # Stockfish's turn
            try:
                sf.set_fen_position(board.fen())
                sf_move = sf.get_best_move()
                if sf_move:
                    move = chess.Move.from_uci(sf_move)
                else:
                    move = random.choice(list(board.legal_moves))
            except:
                move = random.choice(list(board.legal_moves))
        
        board.push(move)
        move_count += 1
    
    # Determine result
    result = board.result()
    if result == '*':
        result = '1/2-1/2'
    
    # Calculate outcome from model's perspective
    if result == '1-0':
        outcome = 1 if model_plays_white else -1
    elif result == '0-1':
        outcome = -1 if model_plays_white else 1
    else:
        outcome = 0
    
    # Create value training examples
    value_examples = []
    for state, player_sign in value_history:
        value = outcome  # outcome is already from model's perspective
        weight = 1.0 if outcome != 0 else 0.3  # Lower weight for draws
        value_examples.append((state, value, weight))
    
    return value_examples, policy_history, result, outcome

def run_games_vs_stockfish(network: ChessNet, sf: Stockfish, buffer, policy_buffer,
                           n_games: int, config: Config) -> Dict:
    """Play multiple games against Stockfish.
    
    Args:
        network: Chess network
        sf: Stockfish instance
        buffer: Value training buffer (all games)
        policy_buffer: Policy training buffer (only winning games)
        n_games: Number of games to play
        config: Configuration
        
    Returns:
        Dict with results, positions counts, and win count
    """
    if sf is None:
        print("⚠️ Stockfish not available, skipping vs Stockfish games")
        return {'results': {'wins': 0, 'draws': 0, 'losses': 0}, 'positions': 0, 'policy_positions': 0}
    
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    total_value_positions = 0
    total_policy_positions = 0
    
    for i in tqdm(range(n_games), desc="vs Stockfish", leave=False):
        # Alternate colors
        model_plays_white = (i % 2 == 0)
        
        value_examples, policy_examples, result, outcome = play_game_vs_stockfish(
            network, sf, config, model_plays_white
        )
        
        # Add value examples to buffer (all games)
        for state, value, weight in value_examples:
            buffer.add(state, value, weight)
        total_value_positions += len(value_examples)
        
        # Add policy examples ONLY for winning games
        if outcome == 1:  # Model won
            for state, action, mask in policy_examples:
                policy_buffer.add(state, action, mask)
            total_policy_positions += len(policy_examples)
        
        # Track results
        if outcome == 1:
            results['wins'] += 1
        elif outcome == -1:
            results['losses'] += 1
        else:
            results['draws'] += 1
    
    return {
        'results': results, 
        'positions': total_value_positions,
        'policy_positions': total_policy_positions
    }

# ==============================================================================
# Cell 8: Endgame Position Generator
# ==============================================================================

def generate_endgame_positions(n_positions: int) -> List[Tuple[np.ndarray, float]]:
    """Generate endgame positions with known evaluations."""
    positions = []
    
    endgame_types = [
        ('KQ', 'K', 1.0),
        ('KR', 'K', 1.0),
        ('KBB', 'K', 1.0),
        ('KBN', 'K', 0.9),
        ('KP', 'K', 0.8),
        ('K', 'KQ', -1.0),
        ('K', 'KR', -1.0),
        ('KR', 'KR', 0.0),
        ('KQ', 'KQ', 0.0),
        ('KP', 'KP', 0.0),
        ('KRP', 'KR', 0.5),
        ('KR', 'KRP', -0.5),
        ('KQP', 'KQ', 0.4),
        ('KRR', 'K', 1.0),
        ('K', 'KRR', -1.0),
    ]
    
    piece_to_type = {'K': chess.KING, 'Q': chess.QUEEN, 'R': chess.ROOK, 
                     'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}
    
    for _ in range(n_positions):
        pattern = random.choice(endgame_types)
        white_pieces, black_pieces, base_value = pattern
        
        board = chess.Board(None)
        board.clear()
        
        wk_sq = random.choice(list(chess.SQUARES))
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        
        valid_bk = [sq for sq in chess.SQUARES if chess.square_distance(sq, wk_sq) > 1]
        bk_sq = random.choice(valid_bk) if valid_bk else random.choice(chess.SQUARES)
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        
        occupied = {wk_sq, bk_sq}
        for p in white_pieces[1:]:
            valid_sq = [sq for sq in chess.SQUARES if sq not in occupied]
            if valid_sq:
                sq = random.choice(valid_sq)
                if p == 'P':
                    valid_sq = [sq for sq in valid_sq if 1 <= chess.square_rank(sq) <= 6]
                    sq = random.choice(valid_sq) if valid_sq else sq
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.WHITE))
                occupied.add(sq)
        
        for p in black_pieces[1:]:
            valid_sq = [sq for sq in chess.SQUARES if sq not in occupied]
            if valid_sq:
                sq = random.choice(valid_sq)
                if p == 'P':
                    valid_sq = [sq for sq in valid_sq if 1 <= chess.square_rank(sq) <= 6]
                    sq = random.choice(valid_sq) if valid_sq else sq
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.BLACK))
                occupied.add(sq)
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        value = base_value if board.turn == chess.WHITE else -base_value
        value = max(-1.0, min(1.0, value + random.uniform(-0.1, 0.1)))
        
        if board.is_valid() and not board.is_game_over():
            positions.append((encode_board(board), value))
    
    return positions

print(f"✅ Endgame generator ready")

# ==============================================================================
# Cell 9: Buffer
# ==============================================================================

class WeightedReplayBuffer:
    """Buffer for value training - stores (state, value, weight)."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, value: float, weight: float = 1.0):
        self.buffer.append((state, value, weight))
    
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


class PolicyBuffer:
    """Buffer for policy training - stores (state, action, mask) from WINNING games only."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: int, mask: np.ndarray):
        self.buffer.append((state, action, mask))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        
        if len(batch) == 0:
            return None, None, None
            
        states, actions, masks = zip(*batch)
        return np.array(states), np.array(actions, dtype=np.int64), np.array(masks)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


buffer = WeightedReplayBuffer(config.buffer_size)
policy_buffer = PolicyBuffer(config.policy_buffer_size)

# ==============================================================================
# Cell 10: Trainer
# ==============================================================================

class Trainer:
    """Trainer with safe policy training support.
    
    Policy training uses REINFORCE from wins:
    - Only trains on moves from WINNING games (stored in policy_buffer)
    - Uses cross-entropy loss to reinforce winning moves
    - Ultra-low learning rate to prevent collapse
    """
    
    def __init__(self, network: ChessNet, config: Config, 
                 buffer: WeightedReplayBuffer, policy_buffer: PolicyBuffer):
        self.network = network
        self.config = config
        self.buffer = buffer  # Value training buffer
        self.policy_buffer = policy_buffer  # Policy training buffer (wins only)
        self.policy_training_enabled = False
        self.policy_frozen_permanently = False
        
        # Value optimizer (always active)
        self.optimizer_value = torch.optim.AdamW(
            [p for p in network.value_head.parameters() if p.requires_grad],
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        
        # Policy optimizer (created when enabled)
        self.optimizer_policy = None
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    def enable_policy_training(self):
        """Enable policy training after warmup."""
        if self.policy_training_enabled or self.policy_frozen_permanently:
            return
        
        trainable = setup_safe_training(self.network)
        
        # Create policy optimizer for Linear layers only
        policy_params = []
        for name, module in self.network.policy_head.named_modules():
            if isinstance(module, nn.Linear):
                policy_params.extend([p for p in module.parameters() if p.requires_grad])
        
        if policy_params:
            self.optimizer_policy = torch.optim.AdamW(
                policy_params,
                lr=self.config.lr_policy,
                weight_decay=self.config.weight_decay
            )
        
        self.policy_training_enabled = True
        print(f"✅ Policy training enabled ({trainable:,} trainable params)")
        print(f"   Policy LR: {self.config.lr_policy} (ultra-low)")
    
    def disable_policy_training(self, permanent: bool = False):
        """Disable policy training (safety fallback)."""
        if not self.policy_training_enabled:
            return
        
        # Freeze policy head
        for param in self.network.policy_head.parameters():
            param.requires_grad = False
        
        self.optimizer_policy = None
        self.policy_training_enabled = False
        
        if permanent:
            self.policy_frozen_permanently = True
            print("⚠️ Policy training PERMANENTLY disabled (WR drop detected)")
        else:
            print("⚠️ Policy training disabled")
    
    def train_value_epoch(self) -> Dict[str, float]:
        """Train value head only."""
        self.network.eval()
        total_loss = 0
        total_sign_acc = 0
        
        for _ in range(self.config.batches_per_iter):
            states, target_values, weights = self.buffer.sample(self.config.batch_size)
            
            states = torch.FloatTensor(states).to(device)
            target_values = torch.FloatTensor(target_values).to(device)
            weights = torch.FloatTensor(weights).to(device)
            
            self.optimizer_value.zero_grad()
            
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    _, pred_values = self.network(states)
                    
                    # Value loss (weighted MSE)
                    value_loss = (weights * (pred_values.squeeze(-1) - target_values) ** 2).mean()
            
            self.scaler.scale(value_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.value_head.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optimizer_value)
            self.scaler.update()
            
            # Compute sign accuracy
            with torch.no_grad():
                decisive_mask = target_values.abs() > 0.5
                if decisive_mask.sum() > 0:
                    pred_sign = (pred_values.squeeze(-1)[decisive_mask] > 0).float()
                    target_sign = (target_values[decisive_mask] > 0).float()
                    sign_acc = (pred_sign == target_sign).float().mean().item()
                else:
                    sign_acc = 0.5
            
            total_loss += value_loss.item()
            total_sign_acc += sign_acc
        
        n = self.config.batches_per_iter
        return {'value_loss': total_loss / n, 'sign_acc': total_sign_acc / n}
    
    def train_policy_epoch(self) -> Dict[str, float]:
        """Train policy head using REINFORCE from wins.
        
        Uses cross-entropy loss on actions taken in winning games.
        This reinforces the moves that led to wins.
        """
        if not self.policy_training_enabled or self.optimizer_policy is None:
            return {'policy_loss': 0.0, 'policy_acc': 0.0}
        
        if len(self.policy_buffer) < self.config.min_win_games_for_policy * 10:
            print(f"   ⚠️ Not enough winning moves for policy training ({len(self.policy_buffer)})")
            return {'policy_loss': 0.0, 'policy_acc': 0.0}
        
        self.network.eval()
        total_policy_loss = 0
        total_policy_acc = 0
        n_batches = 0
        
        # Train on fewer batches for policy (more conservative)
        policy_batches = max(5, self.config.batches_per_iter // 2)
        
        for _ in range(policy_batches):
            states, actions, masks = self.policy_buffer.sample(self.config.batch_size)
            
            if states is None or len(states) == 0:
                continue
            
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            masks = torch.BoolTensor(masks).to(device)
            
            self.optimizer_policy.zero_grad()
            
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    policy_logits, _ = self.network(states, masks)
                    
                    # Cross-entropy loss on actual moves taken (REINFORCE from wins)
                    # This reinforces the moves that led to victories
                    policy_loss = F.cross_entropy(policy_logits, actions)
            
            self.scaler.scale(policy_loss).backward()
            
            # Very aggressive gradient clipping for policy (prevent collapse)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.policy_head.parameters() if p.requires_grad], 0.5
            )
            
            self.scaler.step(self.optimizer_policy)
            self.scaler.update()
            
            # Compute policy accuracy
            with torch.no_grad():
                pred_actions = policy_logits.argmax(dim=-1)
                policy_acc = (pred_actions == actions).float().mean().item()
            
            total_policy_loss += policy_loss.item()
            total_policy_acc += policy_acc
            n_batches += 1
        
        if n_batches == 0:
            return {'policy_loss': 0.0, 'policy_acc': 0.0}
        
        return {
            'policy_loss': total_policy_loss / n_batches, 
            'policy_acc': total_policy_acc / n_batches
        }
    
    def train_epoch(self, train_policy: bool = False) -> Dict[str, float]:
        """Train value head (always) and optionally policy head."""
        # Always train value
        metrics = self.train_value_epoch()
        
        # Optionally train policy
        if train_policy and self.policy_training_enabled:
            policy_metrics = self.train_policy_epoch()
            metrics.update(policy_metrics)
        else:
            metrics['policy_loss'] = 0.0
            metrics['policy_acc'] = 0.0
        
        return metrics

trainer = Trainer(network, config, buffer, policy_buffer)

# ==============================================================================
# Cell 11: Evaluation
# ==============================================================================

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

def evaluate_vs_stockfish(network: ChessNet, sf, n_games: int) -> Tuple[float, Dict]:
    if sf is None:
        return 0.0, {'wins': 0, 'draws': 0, 'losses': 0}
    
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    
    for i in range(n_games):
        board = chess.Board()
        move_count = 0
        model_is_white = (i % 2 == 0)
        
        try:
            while not board.is_game_over() and move_count < 150:
                is_model_turn = (board.turn == chess.WHITE) == model_is_white
                
                if is_model_turn:
                    move, _ = network.predict_move(board, temperature=0.1)
                else:
                    sf.set_fen_position(board.fen())
                    sf_move = sf.get_best_move()
                    move = chess.Move.from_uci(sf_move) if sf_move else random.choice(list(board.legal_moves))
                
                board.push(move)
                move_count += 1
            
            result = board.result()
            if result == '1-0':
                if model_is_white:
                    results['wins'] += 1
                else:
                    results['losses'] += 1
            elif result == '0-1':
                if model_is_white:
                    results['losses'] += 1
                else:
                    results['wins'] += 1
            else:
                results['draws'] += 1
        except:
            results['draws'] += 1
    
    wr = (results['wins'] + 0.5 * results['draws']) / n_games
    return wr, results

# ==============================================================================
# Cell 12: Main Training Loop
# ==============================================================================

def train_v25():
    global stockfish
    current_skill = config.stockfish_skill
    last_wr_sf = 0.0
    policy_frozen = False
    
    print("\n" + "=" * 70)
    print("🚀 STARTING v25 - SAFE POLICY TRAINING + VS STOCKFISH")
    print("=" * 70)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr_random, _ = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr_random:.0%}")
    
    initial_wr_sf = 0.0
    if stockfish:
        initial_wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_eval_games)
        print(f"   WR vs Stockfish(skill={current_skill}): {initial_wr_sf:.0%}")
        last_wr_sf = initial_wr_sf
    
    history = {
        'iterations': [],
        'value_loss': [],
        'sign_accuracy': [],
        'wr_random': [initial_wr_random],
        'wr_stockfish': [initial_wr_sf],
        'training_wr': [],
        'stockfish_skill': [current_skill],
        'policy_trained': [],
    }
    
    best_sf_wr = initial_wr_sf
    
    for iteration in range(1, config.rl_iterations + 1):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration}/{config.rl_iterations} (Stockfish skill={current_skill})")
        print(f"{'='*70}")
        
        # Check if should enable policy training (after warmup)
        if iteration == config.policy_warmup + 1 and not policy_frozen:
            trainer.enable_policy_training()
        
        # Determine if should train policy this iteration
        # Policy starts at warmup+1, then every policy_update_interval iterations
        policy_phase = iteration > config.policy_warmup and not policy_frozen
        iters_since_warmup = iteration - config.policy_warmup
        should_train_policy = policy_phase and (iters_since_warmup == 1 or iters_since_warmup % config.policy_update_interval == 0)
        
        # 1. Generate endgame positions
        print(f"\n♟️ Generating endgame positions...")
        endgame_positions = generate_endgame_positions(config.endgame_positions_per_iter)
        buffer.add_batch(endgame_positions, weight=1.2)
        print(f"   Added {len(endgame_positions)} endgame positions")
        
        # 2. Play vs Stockfish
        if stockfish:
            print(f"\n🎮 Playing vs Stockfish (skill={current_skill}, {config.games_vs_stockfish} games)...")
            sf_results = run_games_vs_stockfish(network, stockfish, buffer, policy_buffer,
                                                 config.games_vs_stockfish, config)
            res = sf_results['results']
            training_wr = (res['wins'] + 0.5 * res['draws']) / max(1, sum(res.values()))
            print(f"   W={res['wins']} D={res['draws']} L={res['losses']} (WR={training_wr:.0%})")
            print(f"   Value Buffer: {len(buffer)} | Policy Buffer: {len(policy_buffer)} (wins only)")
            history['training_wr'].append(training_wr)
        else:
            print("⚠️ Stockfish not available")
            history['training_wr'].append(0)
        
        # 3. Training
        if len(buffer) >= config.min_buffer_size:
            mode = "POLICY+VALUE" if should_train_policy else "value only"
            print(f"\n📚 Training ({mode})...")
            train_metrics = trainer.train_epoch(train_policy=should_train_policy)
            print(f"   Value Loss: {train_metrics['value_loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            if should_train_policy and train_metrics.get('policy_loss', 0) > 0:
                print(f"   Policy Loss: {train_metrics['policy_loss']:.4f}")
                print(f"   Policy Acc: {train_metrics['policy_acc']:.1%}")
            
            history['iterations'].append(iteration)
            history['value_loss'].append(train_metrics['value_loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
            history['policy_trained'].append(1 if should_train_policy else 0)
        
        # 4. Evaluation
        if iteration % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            
            wr_random, _ = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr_random)
            print(f"   WR vs Random: {wr_random:.0%}")
            
            if stockfish:
                wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_eval_games)
                history['wr_stockfish'].append(wr_sf)
                history['stockfish_skill'].append(current_skill)
                print(f"   🐟 WR vs Stockfish(skill={current_skill}): {wr_sf:.0%} (W:{det_sf['wins']} D:{det_sf['draws']} L:{det_sf['losses']})")
                
                # Save best
                if wr_sf > best_sf_wr:
                    best_sf_wr = wr_sf
                    torch.save(network.state_dict(), f'{SAVE_DIR}/chess_v25_best_sf.pt')
                    print(f"   ✨ New best vs Stockfish! Saved.")
                
                # SAFETY: Check for WR drop -> freeze policy permanently
                if trainer.policy_training_enabled and last_wr_sf > 0:
                    wr_drop = last_wr_sf - wr_sf
                    if wr_drop >= config.wr_drop_threshold:
                        print(f"\n⚠️ WR dropped {wr_drop*100:.1f}%! Freezing policy PERMANENTLY.")
                        trainer.disable_policy_training(permanent=True)
                        policy_frozen = True
                
                last_wr_sf = wr_sf
                
                # Curriculum: increase skill if doing well
                if wr_sf >= config.skill_up_threshold and current_skill < 5:
                    current_skill += 1
                    stockfish.set_skill_level(current_skill)
                    print(f"   🎯 Stockfish skill increased to {current_skill}!")
        
        iter_time = time.time() - iter_start
        eta = (config.rl_iterations - iteration) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), f'{SAVE_DIR}/chess_v25_final.pt')
    print(f"\n💾 Saved: chess_v25_final.pt")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    final_wr_random, _ = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {final_wr_random:.0%}")
    
    if stockfish:
        final_wr_sf, det_sf = evaluate_vs_stockfish(network, stockfish, config.stockfish_eval_games)
        print(f"   WR vs Stockfish: {final_wr_sf:.0%}")
        print(f"\n🏆 Best Stockfish WR: {best_sf_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 13: Run
# ==============================================================================

if __name__ == "__main__":
    history = train_v25()
    
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
    
    if history['wr_stockfish']:
        axes[0, 2].plot(history['wr_stockfish'], 'r-o', label='Eval vs SF', linewidth=2)
        if history['training_wr']:
            axes[0, 2].plot(history['training_wr'], 'b--', alpha=0.5, label='Training WR')
        axes[0, 2].set_title('Win Rate vs Stockfish')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    if history['wr_random']:
        axes[1, 0].plot(history['wr_random'], 'g-o')
        axes[1, 0].set_title('Win Rate vs Random')
        axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='Target')
        axes[1, 0].grid(True, alpha=0.3)
    
    if history['training_wr']:
        axes[1, 1].bar(range(len(history['training_wr'])), history['training_wr'], 
                       color='blue', alpha=0.7)
        axes[1, 1].set_title('Training WR vs Stockfish (per iter)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--')
    
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
    plt.savefig(f'{SAVE_DIR}/training_v25.png', dpi=150)
    plt.show()
    
    print("\n🎉 v25 TRAINING COMPLETE!")
    print("=" * 70)
