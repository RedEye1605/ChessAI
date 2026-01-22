"""
================================================================================
⚡ CHESS AI v20 - EXTENDED SELF-PLAY + CONSERVATIVE RL
================================================================================

🎯 COMPREHENSIVE IMPROVEMENTS:

1. EXTENDED SELF-PLAY
   - 100+ games per iteration
   - Varied starting positions for diversity
   - Temperature scheduling for exploration

2. CONSERVATIVE VALUE TRAINING
   - Frozen policy (no catastrophic forgetting)
   - Game outcomes as targets {-1, 0, +1}
   - Very low learning rate (1e-6)

3. COMPREHENSIVE MONITORING
   - Value loss tracking
   - Sign accuracy (predicting winner)
   - Win/Draw/Loss distribution
   - Early warning if metrics degrade

4. SAFETY FEATURES
   - Auto-save best model
   - Stop if WR vs random drops significantly
   - Checkpoint every N iterations

5. STOCKFISH EVALUATION (Optional, at end)
   - Evaluasi vs SF Level 0 di akhir training
   - Sebagai benchmark, bukan training signal

Training time: ~2-4 hours on P100
================================================================================
"""

# ==============================================================================
# Cell 1: Dependencies
# ==============================================================================

try:
    get_ipython().system('pip install -q python-chess tqdm matplotlib pandas')
except:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'python-chess', 'tqdm', 'matplotlib', 'pandas'], capture_output=True)

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

print("=" * 70)
print("⚡ CHESS AI v20 - EXTENDED SELF-PLAY + CONSERVATIVE RL")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Configuration
# ==============================================================================

@dataclass
class Config:
    # Network (must match v17!)
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Extended Self-Play
    self_play_games: int = 100       # Games per iteration (increased!)
    max_moves_per_game: int = 150    
    use_varied_openings: bool = True # Start from different positions
    
    # Temperature Scheduling
    temp_initial: float = 0.8        # More exploration early
    temp_final: float = 0.3          # Less exploration late
    temp_decay_iters: int = 50       # Iterations to reach final temp
    temp_move_threshold: int = 30    # After this move, use temp=0.1
    
    # Conservative Training
    rl_iterations: int = 100         # Banyak iterasi untuk self-play intensif
    batch_size: int = 256
    lr_value: float = 1e-6           # Very low LR
    weight_decay: float = 1e-5
    batches_per_iter: int = 20       # Training batches per iteration
    
    # Buffer
    buffer_size: int = 100000        # Large buffer
    min_buffer_size: int = 2000      # Start training after this
    
    # Evaluation
    eval_games: int = 50
    eval_interval: int = 5           # Evaluate every N iterations
    checkpoint_interval: int = 10    # Save checkpoint every N iterations
    
    # Safety
    min_wr_threshold: float = 0.85   # Stop if WR drops below this
    max_draw_ratio: float = 0.95     # Warning if draws > this

config = Config()
print(f"\n📋 Configuration:")
print(f"   Self-play: {config.self_play_games} games × {config.rl_iterations} iterations")
print(f"   Buffer: {config.buffer_size} positions")
print(f"   LR: {config.lr_value}")
print(f"   Varied openings: {config.use_varied_openings}")

# ==============================================================================
# Cell 3: State Encoding (12 channels)
# ==============================================================================

NUM_ACTIONS = 4096

def encode_board(board: chess.Board) -> np.ndarray:
    """12-channel board encoding."""
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
# Cell 4: Varied Openings (For Game Diversity)
# ==============================================================================

COMMON_OPENINGS = [
    # Italian Game
    "e2e4 e7e5 g1f3 b8c6 f1c4",
    # Sicilian Defense
    "e2e4 c7c5 g1f3 d7d6",
    # French Defense
    "e2e4 e7e6 d2d4 d7d5",
    # Caro-Kann
    "e2e4 c7c6 d2d4 d7d5",
    # Queen's Gambit
    "d2d4 d7d5 c2c4 e7e6",
    # King's Indian
    "d2d4 g8f6 c2c4 g7g6",
    # London System
    "d2d4 d7d5 c1f4",
    # Scotch Game
    "e2e4 e7e5 g1f3 b8c6 d2d4",
    # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    # Starting position (no moves)
    "",
]

def get_random_opening_position() -> chess.Board:
    """Get a board after playing a random opening sequence."""
    board = chess.Board()
    
    if not config.use_varied_openings or random.random() < 0.3:
        return board  # 30% start from initial position
    
    opening = random.choice(COMMON_OPENINGS)
    if opening:
        for uci in opening.split():
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    board.push(move)
            except:
                break
    
    return board

print(f"✅ {len(COMMON_OPENINGS)} varied opening positions available")

# ==============================================================================
# Cell 5: Neural Network
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
    
    def predict_move(self, board: chess.Board, temperature: float = 0.5) -> Tuple[chess.Move, float]:
        """Predict move using policy head."""
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
# Cell 6: Load Model and Freeze
# ==============================================================================

SUPERVISED_MODEL_PATH = '/kaggle/input/supervisedmodel/pytorch/default/1/chess_v17_supervised.pt'

network = ChessNet(in_channels=config.input_channels, 
                   filters=config.filters, 
                   blocks=config.blocks).to(device)

if os.path.exists(SUPERVISED_MODEL_PATH):
    network.load_state_dict(torch.load(SUPERVISED_MODEL_PATH, map_location=device))
    print(f"✅ Loaded supervised model from: {SUPERVISED_MODEL_PATH}")
else:
    print(f"⚠️ Model not found at: {SUPERVISED_MODEL_PATH}")
    print("   Starting with random weights")

# Freeze everything except value head
for param in network.stem.parameters():
    param.requires_grad = False
for param in network.tower.parameters():
    param.requires_grad = False
for param in network.policy_head.parameters():
    param.requires_grad = False
for param in network.value_head.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
total = sum(p.numel() for p in network.parameters())
print(f"✅ ChessNet: {total:,} params ({trainable:,} trainable)")
print(f"   Policy head: FROZEN")
print(f"   Value head: TRAINABLE")

# ==============================================================================
# Cell 7: Replay Buffer with Statistics
# ==============================================================================

class ReplayBuffer:
    """Replay buffer with statistics tracking."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.stats = {'wins': 0, 'draws': 0, 'losses': 0}
    
    def add(self, state: np.ndarray, value: float):
        self.buffer.append((state, value))
        
        # Track distribution
        if value > 0.5:
            self.stats['wins'] += 1
        elif value < -0.5:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states, values = zip(*batch)
        return np.array(states), np.array(values, dtype=np.float32)
    
    def get_distribution(self) -> Dict[str, float]:
        total = sum(self.stats.values())
        if total == 0:
            return {'wins': 0, 'draws': 0, 'losses': 0}
        return {k: v/total for k, v in self.stats.items()}
    
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(config.buffer_size)

# ==============================================================================
# Cell 8: Self-Play Engine
# ==============================================================================

class SelfPlayEngine:
    """Engine for running self-play games."""
    
    def __init__(self, network: ChessNet, config: Config):
        self.network = network
        self.config = config
        self.game_stats = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
        self.avg_game_length = 0
    
    def get_temperature(self, iteration: int, move_num: int) -> float:
        """Get temperature based on training progress and game phase."""
        # Iteration-based decay
        progress = min(1.0, iteration / self.config.temp_decay_iters)
        base_temp = self.config.temp_initial + (self.config.temp_final - self.config.temp_initial) * progress
        
        # Move-based adjustment (lower temp in endgame)
        if move_num > self.config.temp_move_threshold:
            return 0.1
        
        return base_temp
    
    def play_game(self, iteration: int) -> List[Tuple[np.ndarray, float]]:
        """Play a single self-play game."""
        board = get_random_opening_position()
        game_history = []
        move_count = 0
        
        while not board.is_game_over() and move_count < self.config.max_moves_per_game:
            temp = self.get_temperature(iteration, move_count)
            
            state = encode_board(board)
            player = 1 if board.turn == chess.WHITE else -1
            game_history.append((state, player))
            
            move, _ = self.network.predict_move(board, temperature=temp)
            board.push(move)
            move_count += 1
        
        # Get outcome
        result = board.result()
        if result == '1-0':
            outcome = 1
        elif result == '0-1':
            outcome = -1
        else:
            outcome = 0
        
        self.game_stats[result] = self.game_stats.get(result, 0) + 1
        self.avg_game_length = 0.9 * self.avg_game_length + 0.1 * move_count
        
        # Create training examples
        examples = []
        for state, player in game_history:
            value = outcome * player
            examples.append((state, value))
        
        return examples, result
    
    def run_games(self, n_games: int, iteration: int, buffer: ReplayBuffer) -> Dict:
        """Run multiple self-play games."""
        self.network.eval()
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
        total_positions = 0
        
        for _ in tqdm(range(n_games), desc="Self-play", leave=False):
            examples, result = self.play_game(iteration)
            
            for state, value in examples:
                buffer.add(state, value)
            
            results[result] = results.get(result, 0) + 1
            total_positions += len(examples)
        
        return {
            'results': results,
            'positions': total_positions,
            'avg_game_length': self.avg_game_length
        }

engine = SelfPlayEngine(network, config)
print(f"✅ Self-play engine initialized")

# ==============================================================================
# Cell 9: Trainer
# ==============================================================================

class Trainer:
    """Training manager with comprehensive monitoring."""
    
    def __init__(self, network: ChessNet, config: Config, buffer: ReplayBuffer):
        self.network = network
        self.config = config
        self.buffer = buffer
        
        self.optimizer = torch.optim.AdamW(
            network.value_head.parameters(),
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        
        self.loss_history = []
        self.sign_acc_history = []
    
    def train_batch(self) -> Dict[str, float]:
        """Train on a single batch."""
        self.network.train()
        
        states, target_values = self.buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        target_values = torch.FloatTensor(target_values).to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _, pred_values = self.network(states)
            loss = F.mse_loss(pred_values.squeeze(-1), target_values)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.network.value_head.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Compute sign accuracy
        with torch.no_grad():
            pred_sign = (pred_values.squeeze(-1) > 0).float()
            target_sign = (target_values > 0).float()
            sign_acc = (pred_sign == target_sign).float().mean().item()
        
        return {'loss': loss.item(), 'sign_acc': sign_acc}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for multiple batches."""
        total_loss = 0
        total_sign_acc = 0
        
        for _ in range(self.config.batches_per_iter):
            metrics = self.train_batch()
            total_loss += metrics['loss']
            total_sign_acc += metrics['sign_acc']
        
        n = self.config.batches_per_iter
        avg_loss = total_loss / n
        avg_sign_acc = total_sign_acc / n
        
        self.loss_history.append(avg_loss)
        self.sign_acc_history.append(avg_sign_acc)
        
        return {'loss': avg_loss, 'sign_acc': avg_sign_acc}

trainer = Trainer(network, config, buffer)
print(f"✅ Trainer initialized (LR={config.lr_value})")

# ==============================================================================
# Cell 10: Evaluator
# ==============================================================================

class Evaluator:
    """Evaluation manager."""
    
    def __init__(self, network: ChessNet):
        self.network = network
    
    def vs_random(self, n_games: int) -> Tuple[float, Dict]:
        """Evaluate against random opponent."""
        self.network.eval()
        results = {'wins': 0, 'draws': 0, 'losses': 0}
        
        for _ in range(n_games):
            board = chess.Board()
            move_count = 0
            
            while not board.is_game_over() and move_count < 200:
                if board.turn == chess.WHITE:
                    move, _ = self.network.predict_move(board, temperature=0.1)
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
    
    def vs_stockfish(self, n_games: int, sf_level: int = 0) -> Tuple[float, Dict]:
        """Evaluate against Stockfish (if available)."""
        try:
            import chess.engine
            
            # Try to find Stockfish
            sf_paths = [
                '/usr/games/stockfish',
                '/usr/bin/stockfish',
                'stockfish',
            ]
            
            engine = None
            for path in sf_paths:
                try:
                    engine = chess.engine.SimpleEngine.popen_uci(path)
                    break
                except:
                    continue
            
            if engine is None:
                print("   ⚠️ Stockfish not found")
                return -1, {}
            
            engine.configure({"Skill Level": sf_level})
            
            self.network.eval()
            results = {'wins': 0, 'draws': 0, 'losses': 0}
            
            for _ in range(n_games):
                board = chess.Board()
                move_count = 0
                
                while not board.is_game_over() and move_count < 200:
                    if board.turn == chess.WHITE:
                        move, _ = self.network.predict_move(board, temperature=0.1)
                    else:
                        result = engine.play(board, chess.engine.Limit(time=0.1))
                        move = result.move
                    
                    board.push(move)
                    move_count += 1
                
                result = board.result()
                if result == '1-0':
                    results['wins'] += 1
                elif result == '0-1':
                    results['losses'] += 1
                else:
                    results['draws'] += 1
            
            engine.quit()
            wr = (results['wins'] + 0.5 * results['draws']) / n_games
            return wr, results
            
        except Exception as e:
            print(f"   ⚠️ Stockfish evaluation failed: {e}")
            return -1, {}

evaluator = Evaluator(network)
print(f"✅ Evaluator initialized")

# ==============================================================================
# Cell 11: Main Training Loop
# ==============================================================================

def train_v20():
    """Main training loop with comprehensive monitoring."""
    print("\n" + "=" * 70)
    print("🚀 STARTING v20 - EXTENDED SELF-PLAY + CONSERVATIVE RL")
    print("=" * 70)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr, details = evaluator.vs_random(config.eval_games)
    print(f"   WR vs Random: {initial_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    # History
    history = {
        'iterations': [],
        'value_loss': [],
        'sign_accuracy': [],
        'wr_random': [initial_wr],
        'white_wins': [],
        'black_wins': [],
        'draws': [],
        'buffer_size': [],
        'temperature': [],
    }
    
    best_wr = initial_wr
    start_time = time.time()
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration + 1}/{config.rl_iterations}")
        print(f"{'='*70}")
        
        # 1. Self-Play
        current_temp = engine.get_temperature(iteration, 0)
        print(f"\n🎮 Self-Play ({config.self_play_games} games, temp={current_temp:.2f})...")
        
        sp_results = engine.run_games(config.self_play_games, iteration, buffer)
        results = sp_results['results']
        
        print(f"   Results: W={results.get('1-0',0)} D={results.get('1/2-1/2',0)} L={results.get('0-1',0)}")
        print(f"   Positions: {sp_results['positions']}, Buffer: {len(buffer)}")
        print(f"   Avg game length: {sp_results['avg_game_length']:.0f} moves")
        
        history['white_wins'].append(results.get('1-0', 0))
        history['black_wins'].append(results.get('0-1', 0))
        history['draws'].append(results.get('1/2-1/2', 0))
        history['buffer_size'].append(len(buffer))
        history['temperature'].append(current_temp)
        
        # Check draw ratio
        total_games = sum(results.values())
        draw_ratio = results.get('1/2-1/2', 0) / max(1, total_games)
        if draw_ratio > config.max_draw_ratio:
            print(f"   ⚠️ High draw ratio: {draw_ratio:.0%}")
        
        # 2. Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training (value head only)...")
            train_metrics = trainer.train_epoch()
            
            print(f"   Value Loss: {train_metrics['loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            
            history['iterations'].append(iteration + 1)
            history['value_loss'].append(train_metrics['loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
        else:
            print(f"\n⏳ Waiting for buffer ({len(buffer)}/{config.min_buffer_size})")
        
        # 3. Evaluation
        if (iteration + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr, details = evaluator.vs_random(config.eval_games)
            history['wr_random'].append(wr)
            
            delta = (wr - initial_wr) * 100
            status = "✅" if wr >= initial_wr else "⚠️"
            print(f"   {status} WR vs Random: {wr:.0%} ({delta:+.1f}%)")
            print(f"      W:{details['wins']} D:{details['draws']} L:{details['losses']}")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v20_best.pt')
                print(f"   ✨ New best! Saved.")
            
            # Safety check
            if wr < config.min_wr_threshold:
                print(f"\n🛑 WR dropped below threshold ({config.min_wr_threshold:.0%})")
                print(f"   Stopping training to preserve model")
                break
        
        # 4. Checkpoint
        if (iteration + 1) % config.checkpoint_interval == 0:
            torch.save(network.state_dict(), f'/kaggle/working/chess_v20_iter{iteration+1}.pt')
            print(f"\n💾 Checkpoint saved: iter{iteration+1}")
        
        # Timing
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        eta = (config.rl_iterations - iteration - 1) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | Total: {total_time/60:.1f}min | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), '/kaggle/working/chess_v20_final.pt')
    print(f"\n💾 Final model saved: chess_v20_final.pt")
    
    # Final evaluation
    print(f"\n" + "=" * 70)
    print(f"📊 FINAL EVALUATION")
    print(f"=" * 70)
    
    print(f"\n🎯 vs Random:")
    final_wr, details = evaluator.vs_random(config.eval_games)
    print(f"   WR: {final_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    print(f"\n🎯 vs Stockfish Level 0:")
    sf_wr, sf_details = evaluator.vs_stockfish(10, sf_level=0)
    if sf_wr >= 0:
        print(f"   WR: {sf_wr:.0%} (W:{sf_details['wins']} D:{sf_details['draws']} L:{sf_details['losses']})")
    
    return history

# ==============================================================================
# Cell 12: Run Training
# ==============================================================================

if __name__ == "__main__":
    history = train_v20()
    
    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Value Loss
    if history['value_loss']:
        axes[0, 0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0, 0].set_title('Value Loss', fontsize=12)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Sign Accuracy
    if history['sign_accuracy']:
        axes[0, 1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random')
        axes[0, 1].set_title('Value Sign Accuracy', fontsize=12)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Win Rate
    if history['wr_random']:
        x = range(len(history['wr_random']))
        axes[0, 2].plot(x, history['wr_random'], 'b-o', markersize=4)
        axes[0, 2].axhline(y=history['wr_random'][0], color='g', linestyle='--', 
                          label=f'Initial ({history["wr_random"][0]:.0%})')
        axes[0, 2].set_title('Win Rate vs Random', fontsize=12)
        axes[0, 2].set_xlabel('Evaluation')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Self-Play Results (Stacked Bar)
    if history['white_wins']:
        x = range(len(history['white_wins']))
        axes[1, 0].bar(x, history['white_wins'], label='White', alpha=0.8, color='white', edgecolor='black')
        axes[1, 0].bar(x, history['draws'], bottom=history['white_wins'], label='Draw', alpha=0.8, color='gray')
        black_bottom = [w + d for w, d in zip(history['white_wins'], history['draws'])]
        axes[1, 0].bar(x, history['black_wins'], bottom=black_bottom, label='Black', alpha=0.8, color='black')
        axes[1, 0].set_title('Self-Play Results', fontsize=12)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Games')
        axes[1, 0].legend()
    
    # Buffer Size
    if history['buffer_size']:
        axes[1, 1].plot(history['buffer_size'], 'purple')
        axes[1, 1].set_title('Buffer Size', fontsize=12)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Positions')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Temperature
    if history['temperature']:
        axes[1, 2].plot(history['temperature'], 'orange')
        axes[1, 2].set_title('Exploration Temperature', fontsize=12)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Temperature')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v20.png', dpi=150)
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 v20 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Initial WR: {history['wr_random'][0]:.0%}")
    print(f"   Final WR:   {history['wr_random'][-1]:.0%}")
    print(f"   Change:     {(history['wr_random'][-1] - history['wr_random'][0])*100:+.1f}%")
    print(f"   Total positions: {history['buffer_size'][-1] if history['buffer_size'] else 0:,}")
    print(f"\n📁 Files saved:")
    print(f"   chess_v20_best.pt  - Best model during training")
    print(f"   chess_v20_final.pt - Final model")
    print(f"   chess_v20_iter*.pt - Checkpoints")
    print(f"   training_v20.png   - Training plots")
