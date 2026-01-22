"""
================================================================================
⚡ CHESS AI v27 - STOCKFISH-GUIDED LEARNING (FINAL VERSION)
================================================================================

🔥 MAJOR IMPROVEMENTS:
1. STOCKFISH EVALUATION - Use SF depth 10 for accurate value targets
2. POLICY DISTILLATION - Learn directly from Stockfish best moves
3. MIXED OPPONENT SKILL - Rotate skill 0, 1, 2 for generalization
4. HIGHER POLICY LR - 1e-7 (100x higher than v25!)
5. SMARTER EXPLORATION - Temperature 0.4 for more varied play

Target: Continue from v26 (31% WR) -> 50%+ WR vs Stockfish skill 0
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

SAVE_DIR = '/kaggle/working' if os.path.exists('/kaggle') else '.'

print("=" * 70)
print("⚡ CHESS AI v27 - STOCKFISH-GUIDED LEARNING")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")
print(f"✅ Save Dir: {SAVE_DIR}")


# ==============================================================================
# Cell 2: Configuration (V27 - FINAL)
# ==============================================================================

@dataclass
class Config:
    # Network
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Games
    games_vs_stockfish: int = 100
    max_moves_per_game: int = 150
    
    # Temperature - slightly higher for exploration
    model_temperature: float = 0.4
    
    # Training
    rl_iterations: int = 100
    batch_size: int = 256
    
    # Learning rates - HIGHER POLICY LR
    lr_value: float = 5e-7
    lr_policy: float = 1e-7              # 100x higher than v25!
    weight_decay: float = 1e-5
    batches_per_iter: int = 25
    
    # Policy training
    policy_update_interval: int = 2      # Every 2 iters (more frequent)
    policy_warmup: int = 3               # Short warmup
    wr_drop_threshold: float = 0.10      # 10% tolerance
    min_policy_evals: int = 3
    min_win_games_for_policy: int = 5
    
    # Stockfish - DUAL DEPTH
    stockfish_depth_play: int = 1        # For playing (weak opponent)
    stockfish_depth_eval: int = 10       # For evaluation (accurate teacher)
    stockfish_skills: List[int] = None   # Will be [0, 1, 2] - rotating
    
    # Value learning from SF eval
    use_stockfish_eval: bool = True      # Use SF eval as value target
    
    # Policy distillation
    policy_distillation_weight: float = 0.3  # 30% weight on SF moves
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    policy_buffer_size: int = 30000      # Larger buffer
    buffer_refresh_interval: int = 50    # Less frequent refresh
    
    # Endgame (reduced - focus on SF games)
    endgame_positions_per_iter: int = 100
    
    # Evaluation
    eval_games: int = 30
    eval_interval: int = 5
    stockfish_eval_games: int = 100
    
    # Checkpoint
    checkpoint_interval: int = 20
    
    def __post_init__(self):
        if self.stockfish_skills is None:
            self.stockfish_skills = [0, 1, 2]

config = Config()
print(f"\n📋 Configuration (V27 - STOCKFISH GUIDED):")
print(f"   Value LR: {config.lr_value}")
print(f"   Policy LR: {config.lr_policy} (100x higher than v25!)")
print(f"   SF Depth Play: {config.stockfish_depth_play} | SF Depth Eval: {config.stockfish_depth_eval}")
print(f"   SF Skills: {config.stockfish_skills} (rotating)")
print(f"   Use SF Eval: {config.use_stockfish_eval}")
print(f"   Policy Distillation: {config.policy_distillation_weight*100:.0f}%")

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
# Cell 5: Load v26 Model
# ==============================================================================

MODEL_PATHS = [
    '/kaggle/input/supervisedmodel/pytorch/default/10/chess_v26_final.pt',
    '/kaggle/input/supervisedmodel/pytorch/default/10/chess_v26_best_sf.pt',
    '/kaggle/input/supervisedmodel/pytorch/default/9/chess_v25_best_sf.pt',
    '/kaggle/working/chess_v26_final.pt',
    '/kaggle/working/chess_v26_best_sf.pt',
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
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.endswith('.pt'):
                        print(f"   Found: {os.path.join(root, f)}")

def freeze_module(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

def setup_safe_training(network: ChessNet):
    freeze_module(network.stem)
    freeze_module(network.tower)
    
    for param in network.value_head.parameters():
        param.requires_grad = True
    for module in network.value_head.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    
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

# ==============================================================================
# Cell 6: Stockfish Setup (DUAL - Play & Eval)
# ==============================================================================

def setup_stockfish(depth: int = 1, skill: int = 0):
    if not STOCKFISH_AVAILABLE:
        return None
    
    try:
        import subprocess
        subprocess.run(['apt-get', 'install', '-y', 'stockfish'], 
                      capture_output=True, timeout=60)
    except:
        pass
    
    for path in ["/usr/games/stockfish", "/usr/bin/stockfish", "stockfish"]:
        try:
            sf = Stockfish(path=path, depth=depth)
            sf.set_skill_level(skill)
            return sf
        except:
            continue
    return None

print(f"\n🔧 Setting up Stockfish instances...")

# SF for playing (weak opponent)
sf_play = setup_stockfish(depth=config.stockfish_depth_play, skill=config.stockfish_skills[0])
if sf_play:
    print(f"✅ SF Play: depth={config.stockfish_depth_play}")

# SF for evaluation (accurate teacher)  
sf_eval = setup_stockfish(depth=config.stockfish_depth_eval, skill=20)
if sf_eval:
    print(f"✅ SF Eval: depth={config.stockfish_depth_eval} (teacher)")

# ==============================================================================
# Cell 7: Get Stockfish Evaluation (NEW)
# ==============================================================================

def get_stockfish_eval(sf: Stockfish, board: chess.Board) -> float:
    """Get Stockfish evaluation normalized to [-1, 1]."""
    if sf is None:
        return 0.0
    
    try:
        sf.set_fen_position(board.fen())
        eval_result = sf.get_evaluation()
        
        if eval_result['type'] == 'mate':
            # Mate in N moves
            mate_in = eval_result['value']
            if mate_in > 0:
                return 1.0  # Winning
            else:
                return -1.0  # Losing
        else:
            # Centipawn evaluation
            cp = eval_result['value']
            # Normalize: 100cp = 0.2, 500cp = 0.7, 1000cp+ = ~1.0
            normalized = np.tanh(cp / 400.0)
            return float(normalized)
    except:
        return 0.0

def get_stockfish_best_move(sf: Stockfish, board: chess.Board) -> Optional[chess.Move]:
    """Get Stockfish's best move for policy distillation."""
    if sf is None:
        return None
    
    try:
        sf.set_fen_position(board.fen())
        sf_move = sf.get_best_move()
        if sf_move:
            return chess.Move.from_uci(sf_move)
    except:
        pass
    return None

# ==============================================================================
# Cell 8: Play vs Stockfish (ENHANCED)
# ==============================================================================

def play_game_vs_stockfish(network: ChessNet, sf_play: Stockfish, sf_eval: Stockfish,
                           config: Config, model_plays_white: bool = True) -> Tuple[List, List, List, str, int]:
    """Play game and collect enhanced training data.
    
    Returns:
        value_examples: (state, sf_eval, weight) - using Stockfish evaluation
        policy_examples: (state, action, mask) - from winning games
        distill_examples: (state, sf_action, mask) - Stockfish's best moves
        result, outcome
    """
    board = chess.Board()
    value_history = []
    policy_history = []
    distill_history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        state = encode_board(board)
        is_model_turn = (board.turn == chess.WHITE) == model_plays_white
        
        if is_model_turn:
            # Get Stockfish evaluation for value training (NEW)
            if config.use_stockfish_eval and sf_eval:
                sf_value = get_stockfish_eval(sf_eval, board)
                # Flip sign if model is black
                if not model_plays_white:
                    sf_value = -sf_value
            else:
                sf_value = None  # Will use game outcome later
            
            value_history.append((state.copy(), sf_value, 1 if board.turn == chess.WHITE else -1))
            
            # Get model's move
            mask = get_legal_mask(board)
            move, _ = network.predict_move(board, temperature=config.model_temperature)
            action = encode_move(move)
            policy_history.append((state.copy(), action, mask.copy()))
            
            # Get Stockfish's best move for distillation (NEW)
            if sf_eval:
                sf_best = get_stockfish_best_move(sf_eval, board)
                if sf_best and sf_best in board.legal_moves:
                    sf_action = encode_move(sf_best)
                    distill_history.append((state.copy(), sf_action, mask.copy()))
        else:
            # Stockfish opponent's turn
            try:
                sf_play.set_fen_position(board.fen())
                sf_move = sf_play.get_best_move()
                if sf_move:
                    move = chess.Move.from_uci(sf_move)
                else:
                    move = random.choice(list(board.legal_moves))
            except:
                move = random.choice(list(board.legal_moves))
        
        board.push(move)
        move_count += 1
    
    result = board.result()
    if result == '*':
        result = '1/2-1/2'
    
    if result == '1-0':
        outcome = 1 if model_plays_white else -1
    elif result == '0-1':
        outcome = -1 if model_plays_white else 1
    else:
        outcome = 0
    
    # Create value training examples
    value_examples = []
    for state, sf_value, player_sign in value_history:
        if sf_value is not None:
            value = sf_value
        else:
            value = float(outcome)
        weight = 1.0 if outcome != 0 else 0.5
        value_examples.append((state, value, weight))
    
    return value_examples, policy_history, distill_history, result, outcome


def run_games_vs_stockfish(network: ChessNet, sf_play: Stockfish, sf_eval: Stockfish,
                           buffer, policy_buffer, distill_buffer,
                           n_games: int, config: Config, current_skill: int) -> Dict:
    """Play multiple games with enhanced data collection."""
    if sf_play is None:
        return {'results': {'wins': 0, 'draws': 0, 'losses': 0}, 
                'positions': 0, 'policy_positions': 0, 'distill_positions': 0}
    
    # Set current skill level
    sf_play.set_skill_level(current_skill)
    
    network.eval()
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    total_value = 0
    total_policy = 0
    total_distill = 0
    
    for i in tqdm(range(n_games), desc=f"vs SF(skill={current_skill})", leave=False):
        model_plays_white = (i % 2 == 0)
        
        value_ex, policy_ex, distill_ex, result, outcome = play_game_vs_stockfish(
            network, sf_play, sf_eval, config, model_plays_white
        )
        
        # Add value examples (all games)
        for state, value, weight in value_ex:
            buffer.add(state, value, weight)
        total_value += len(value_ex)
        
        # Add policy examples (winning games only)
        if outcome == 1:
            for state, action, mask in policy_ex:
                policy_buffer.add(state, action, mask)
            total_policy += len(policy_ex)
        
        # Add distillation examples (all games - learning from SF)
        for state, action, mask in distill_ex:
            distill_buffer.add(state, action, mask)
        total_distill += len(distill_ex)
        
        if outcome == 1:
            results['wins'] += 1
        elif outcome == -1:
            results['losses'] += 1
        else:
            results['draws'] += 1
    
    return {
        'results': results,
        'positions': total_value,
        'policy_positions': total_policy,
        'distill_positions': total_distill
    }

# ==============================================================================
# Cell 9: Endgame Generator (Simplified)
# ==============================================================================

def generate_endgame_positions(n_positions: int, sf_eval: Stockfish = None) -> List[Tuple[np.ndarray, float]]:
    """Generate endgame positions with Stockfish evaluation."""
    positions = []
    
    endgame_types = [
        ('KQ', 'K', 1.0), ('KR', 'K', 1.0), ('KP', 'K', 0.8),
        ('K', 'KQ', -1.0), ('K', 'KR', -1.0), ('KR', 'KR', 0.0),
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
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.WHITE))
                occupied.add(sq)
        
        for p in black_pieces[1:]:
            valid_sq = [sq for sq in chess.SQUARES if sq not in occupied]
            if valid_sq:
                sq = random.choice(valid_sq)
                board.set_piece_at(sq, chess.Piece(piece_to_type[p], chess.BLACK))
                occupied.add(sq)
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        if board.is_valid() and not board.is_game_over():
            # Use SF eval if available
            if sf_eval:
                value = get_stockfish_eval(sf_eval, board)
            else:
                value = base_value if board.turn == chess.WHITE else -base_value
            positions.append((encode_board(board), value))
    
    return positions

print(f"✅ Endgame generator ready")

# ==============================================================================
# Cell 10: Buffers
# ==============================================================================

class WeightedReplayBuffer:
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
distill_buffer = PolicyBuffer(config.policy_buffer_size)  # For Stockfish move distillation

# ==============================================================================
# Cell 11: Trainer (V27 - ENHANCED)
# ==============================================================================

class Trainer:
    def __init__(self, network: ChessNet, config: Config, 
                 buffer, policy_buffer, distill_buffer):
        self.network = network
        self.config = config
        self.buffer = buffer
        self.policy_buffer = policy_buffer
        self.distill_buffer = distill_buffer
        self.policy_training_enabled = False
        self.policy_frozen_permanently = False
        self.wr_history = []
        
        self.optimizer_value = torch.optim.AdamW(
            [p for p in network.value_head.parameters() if p.requires_grad],
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        
        self.optimizer_policy = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    def enable_policy_training(self):
        if self.policy_training_enabled or self.policy_frozen_permanently:
            return
        
        trainable = setup_safe_training(self.network)
        
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
        print(f"✅ Policy training enabled ({trainable:,} params)")
        print(f"   Policy LR: {self.config.lr_policy}")
    
    def disable_policy_training(self, permanent: bool = False):
        if not self.policy_training_enabled:
            return
        
        for param in self.network.policy_head.parameters():
            param.requires_grad = False
        
        self.optimizer_policy = None
        self.policy_training_enabled = False
        
        if permanent:
            self.policy_frozen_permanently = True
            print("⚠️ Policy training PERMANENTLY disabled")
    
    def check_should_freeze(self, current_wr: float) -> bool:
        self.wr_history.append(current_wr)
        
        if len(self.wr_history) < self.config.min_policy_evals:
            return False
        
        recent_avg = np.mean(self.wr_history[-2:])
        older_avg = np.mean(self.wr_history[:-2]) if len(self.wr_history) > 2 else self.wr_history[0]
        
        drop = older_avg - recent_avg
        
        if drop >= self.config.wr_drop_threshold:
            print(f"   📉 WR dropped: {older_avg:.1%} -> {recent_avg:.1%}")
            return True
        return False
    
    def train_value_epoch(self) -> Dict[str, float]:
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
                    value_loss = (weights * (pred_values.squeeze(-1) - target_values) ** 2).mean()
            
            self.scaler.scale(value_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.value_head.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optimizer_value)
            self.scaler.update()
            
            with torch.no_grad():
                decisive_mask = target_values.abs() > 0.3
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
        """Train policy with REINFORCE from wins + Distillation from Stockfish."""
        if not self.policy_training_enabled or self.optimizer_policy is None:
            return {'policy_loss': 0.0, 'policy_acc': 0.0, 'distill_loss': 0.0}
        
        self.network.eval()
        total_reinforce_loss = 0
        total_distill_loss = 0
        total_acc = 0
        n_batches = 0
        
        policy_batches = max(5, self.config.batches_per_iter // 2)
        
        for _ in range(policy_batches):
            # REINFORCE from wins
            reinforce_loss = torch.tensor(0.0, device=device)
            if len(self.policy_buffer) >= self.config.batch_size // 2:
                states, actions, masks = self.policy_buffer.sample(self.config.batch_size // 2)
                if states is not None and len(states) > 0:
                    states_r = torch.FloatTensor(states).to(device)
                    actions_r = torch.LongTensor(actions).to(device)
                    masks_r = torch.BoolTensor(masks).to(device)
                    
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logits_r, _ = self.network(states_r, masks_r)
                        reinforce_loss = F.cross_entropy(logits_r, actions_r)
            
            # Distillation from Stockfish
            distill_loss = torch.tensor(0.0, device=device)
            if len(self.distill_buffer) >= self.config.batch_size // 2:
                states, actions, masks = self.distill_buffer.sample(self.config.batch_size // 2)
                if states is not None and len(states) > 0:
                    states_d = torch.FloatTensor(states).to(device)
                    actions_d = torch.LongTensor(actions).to(device)
                    masks_d = torch.BoolTensor(masks).to(device)
                    
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logits_d, _ = self.network(states_d, masks_d)
                        distill_loss = F.cross_entropy(logits_d, actions_d)
            
            # Combined loss
            w = self.config.policy_distillation_weight
            combined_loss = (1 - w) * reinforce_loss + w * distill_loss
            
            if combined_loss.item() > 0:
                self.optimizer_policy.zero_grad()
                
                with torch.enable_grad():
                    self.scaler.scale(combined_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.network.policy_head.parameters() if p.requires_grad], 0.5
                    )
                    self.scaler.step(self.optimizer_policy)
                    self.scaler.update()
                
                total_reinforce_loss += reinforce_loss.item()
                total_distill_loss += distill_loss.item()
                n_batches += 1
        
        if n_batches == 0:
            return {'policy_loss': 0.0, 'policy_acc': 0.0, 'distill_loss': 0.0}
        
        return {
            'policy_loss': total_reinforce_loss / n_batches,
            'distill_loss': total_distill_loss / n_batches,
            'policy_acc': 0.0  # Not computed for simplicity
        }
    
    def train_epoch(self, train_policy: bool = False) -> Dict[str, float]:
        metrics = self.train_value_epoch()
        
        if train_policy and self.policy_training_enabled:
            policy_metrics = self.train_policy_epoch()
            metrics.update(policy_metrics)
        else:
            metrics['policy_loss'] = 0.0
            metrics['distill_loss'] = 0.0
        
        return metrics

trainer = Trainer(network, config, buffer, policy_buffer, distill_buffer)

# ==============================================================================
# Cell 12: Evaluation
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
                results['wins' if model_is_white else 'losses'] += 1
            elif result == '0-1':
                results['losses' if model_is_white else 'wins'] += 1
            else:
                results['draws'] += 1
        except:
            results['draws'] += 1
    
    wr = (results['wins'] + 0.5 * results['draws']) / n_games
    return wr, results

# ==============================================================================
# Cell 13: Main Training Loop (V27 - FINAL)
# ==============================================================================

def train_v27():
    global sf_play
    skill_idx = 0
    policy_frozen = False
    
    print("\n" + "=" * 70)
    print("🚀 STARTING v27 - STOCKFISH-GUIDED LEARNING")
    print("=" * 70)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr_random, _ = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr_random:.0%}")
    
    initial_wr_sf = 0.0
    if sf_play:
        sf_play.set_skill_level(0)
        initial_wr_sf, det_sf = evaluate_vs_stockfish(network, sf_play, config.stockfish_eval_games)
        print(f"   WR vs Stockfish(skill=0): {initial_wr_sf:.0%}")
    
    history = {
        'iterations': [], 'value_loss': [], 'sign_accuracy': [],
        'wr_random': [initial_wr_random], 'wr_stockfish': [initial_wr_sf],
        'training_wr': [], 'stockfish_skill': [0],
        'policy_loss': [], 'distill_loss': [],
    }
    
    best_sf_wr = initial_wr_sf
    
    for iteration in range(1, config.rl_iterations + 1):
        iter_start = time.time()
        
        # Rotate skill level
        current_skill = config.stockfish_skills[skill_idx % len(config.stockfish_skills)]
        skill_idx += 1
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration}/{config.rl_iterations} (SF skill={current_skill})")
        print(f"{'='*70}")
        
        # Enable policy training after warmup
        if iteration == config.policy_warmup + 1 and not policy_frozen:
            trainer.enable_policy_training()
        
        # Policy training schedule
        policy_phase = iteration > config.policy_warmup and not policy_frozen
        iters_since_warmup = iteration - config.policy_warmup
        should_train_policy = policy_phase and (iters_since_warmup == 1 or iters_since_warmup % config.policy_update_interval == 0)
        
        # Buffer refresh
        if iteration > 1 and iteration % config.buffer_refresh_interval == 0:
            old_p, old_d = len(policy_buffer), len(distill_buffer)
            policy_buffer.clear()
            distill_buffer.clear()
            print(f"🔄 Buffers refreshed (policy: {old_p}→0, distill: {old_d}→0)")
        
        # 1. Endgame positions (reduced)
        print(f"\n♟️ Generating endgame positions...")
        endgame_positions = generate_endgame_positions(config.endgame_positions_per_iter, sf_eval)
        buffer.add_batch(endgame_positions, weight=1.0)
        print(f"   Added {len(endgame_positions)} positions")
        
        # 2. Play vs Stockfish
        if sf_play:
            print(f"\n🎮 Playing vs Stockfish (skill={current_skill})...")
            sf_results = run_games_vs_stockfish(
                network, sf_play, sf_eval, buffer, policy_buffer, distill_buffer,
                config.games_vs_stockfish, config, current_skill
            )
            res = sf_results['results']
            training_wr = (res['wins'] + 0.5 * res['draws']) / max(1, sum(res.values()))
            print(f"   W={res['wins']} D={res['draws']} L={res['losses']} (WR={training_wr:.0%})")
            print(f"   Value: {len(buffer)} | Policy: {len(policy_buffer)} | Distill: {len(distill_buffer)}")
            history['training_wr'].append(training_wr)
        else:
            history['training_wr'].append(0)
        
        # 3. Training
        if len(buffer) >= config.min_buffer_size:
            mode = "POLICY+DISTILL+VALUE" if should_train_policy else "value only"
            print(f"\n📚 Training ({mode})...")
            metrics = trainer.train_epoch(train_policy=should_train_policy)
            print(f"   Value Loss: {metrics['value_loss']:.4f}")
            print(f"   Sign Accuracy: {metrics['sign_acc']:.1%}")
            if should_train_policy and (metrics.get('policy_loss', 0) > 0 or metrics.get('distill_loss', 0) > 0):
                print(f"   Policy Loss: {metrics['policy_loss']:.4f} | Distill Loss: {metrics['distill_loss']:.4f}")
            
            history['iterations'].append(iteration)
            history['value_loss'].append(metrics['value_loss'])
            history['sign_accuracy'].append(metrics['sign_acc'])
            history['policy_loss'].append(metrics.get('policy_loss', 0))
            history['distill_loss'].append(metrics.get('distill_loss', 0))
        
        # 4. Evaluation
        if iteration % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            
            wr_random, _ = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr_random)
            print(f"   WR vs Random: {wr_random:.0%}")
            
            if sf_play:
                sf_play.set_skill_level(0)  # Always eval vs skill 0
                wr_sf, det_sf = evaluate_vs_stockfish(network, sf_play, config.stockfish_eval_games)
                history['wr_stockfish'].append(wr_sf)
                history['stockfish_skill'].append(0)
                print(f"   🐟 WR vs SF(skill=0): {wr_sf:.0%} (W:{det_sf['wins']} D:{det_sf['draws']} L:{det_sf['losses']})")
                
                if wr_sf > best_sf_wr:
                    best_sf_wr = wr_sf
                    torch.save(network.state_dict(), f'{SAVE_DIR}/chess_v27_best_sf.pt')
                    print(f"   ✨ New best! Saved.")
                
                if trainer.policy_training_enabled and not policy_frozen:
                    if trainer.check_should_freeze(wr_sf):
                        print(f"\n⚠️ WR drop detected! Freezing policy.")
                        trainer.disable_policy_training(permanent=True)
                        policy_frozen = True
        
        # Checkpoint
        if iteration % config.checkpoint_interval == 0:
            torch.save(network.state_dict(), f'{SAVE_DIR}/chess_v27_iter{iteration}.pt')
            print(f"   💾 Checkpoint: chess_v27_iter{iteration}.pt")
        
        iter_time = time.time() - iter_start
        eta = (config.rl_iterations - iteration) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), f'{SAVE_DIR}/chess_v27_final.pt')
    print(f"\n💾 Saved: chess_v27_final.pt")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    final_wr, _ = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {final_wr:.0%}")
    
    if sf_play:
        sf_play.set_skill_level(0)
        final_sf, _ = evaluate_vs_stockfish(network, sf_play, config.stockfish_eval_games)
        print(f"   WR vs Stockfish: {final_sf:.0%}")
        print(f"\n🏆 Best Stockfish WR: {best_sf_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 14: Run
# ==============================================================================

if __name__ == "__main__":
    history = train_v27()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if history['value_loss']:
        axes[0, 0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0, 0].set_title('Value Loss (SF Eval Target)')
        axes[0, 0].grid(True, alpha=0.3)
    
    if history['sign_accuracy']:
        axes[0, 1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--')
        axes[0, 1].set_title('Sign Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    if history['wr_stockfish']:
        axes[0, 2].plot(history['wr_stockfish'], 'r-o', linewidth=2, label='Eval WR')
        if history['training_wr']:
            axes[0, 2].plot(history['training_wr'], 'b--', alpha=0.5, label='Training WR')
        axes[0, 2].set_title('Win Rate vs Stockfish')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    if history['wr_random']:
        axes[1, 0].plot(history['wr_random'], 'g-o')
        axes[1, 0].set_title('Win Rate vs Random')
        axes[1, 0].axhline(y=0.9, color='r', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)
    
    if history['policy_loss'] and history['distill_loss']:
        axes[1, 1].plot(history['iterations'], history['policy_loss'], 'purple', label='REINFORCE')
        axes[1, 1].plot(history['iterations'], history['distill_loss'], 'orange', label='Distill')
        axes[1, 1].set_title('Policy Losses')
        axes[1, 1].legend()
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
    plt.savefig(f'{SAVE_DIR}/training_v27.png', dpi=150)
    plt.show()
    
    print("\n🎉 v27 TRAINING COMPLETE!")
    print("=" * 70)
