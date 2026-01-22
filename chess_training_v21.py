"""
================================================================================
⚡ CHESS AI v21 - FIXED BATCHNORM + ANTI-DRAW
================================================================================

🔧 FIXES FROM v20:
1. BatchNorm properly frozen (tidak update running stats)
2. Policy completely preserved

🎯 ANTI-DRAW STRATEGIES:
1. Asymmetric self-play (white vs slightly weaker black)
2. Shorter max game length (force decisive results)
3. Skip/downweight draw positions in training
4. Higher temperature for more tactical games
5. Aggressive opening positions

Training time: ~2-3 hours on P100
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
import copy
warnings.filterwarnings('ignore')

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
print("⚡ CHESS AI v21 - FIXED BATCHNORM + ANTI-DRAW")
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
    
    # Self-Play - ANTI-DRAW settings
    self_play_games: int = 100
    max_moves_per_game: int = 100   # Shorter! Force decisive games
    
    # Asymmetric play (white stronger, black weaker = creates wins/losses)
    white_temperature: float = 0.3  # White plays stronger
    black_temperature: float = 0.8  # Black plays weaker (more random)
    black_noise_prob: float = 0.1   # 10% chance black makes random move
    
    # Training
    rl_iterations: int = 100
    batch_size: int = 256
    lr_value: float = 1e-6
    weight_decay: float = 1e-5
    batches_per_iter: int = 20
    
    # Draw handling
    draw_weight: float = 0.3        # Downweight draw positions in training
    min_win_ratio: float = 0.15     # Warning if wins < this %
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    
    # Evaluation
    eval_games: int = 50
    eval_interval: int = 5
    min_wr_threshold: float = 0.90  # Stop if WR drops below 90%

config = Config()
print(f"\n📋 Configuration (ANTI-DRAW):")
print(f"   Max moves: {config.max_moves_per_game} (shorter games)")
print(f"   White temp: {config.white_temperature} (stronger)")
print(f"   Black temp: {config.black_temperature} (weaker)")
print(f"   Black noise: {config.black_noise_prob:.0%}")
print(f"   Draw weight: {config.draw_weight}")

# ==============================================================================
# Cell 3: State Encoding
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
# Cell 4: Aggressive Opening Positions (More Tactical)
# ==============================================================================

# These openings lead to sharper, more tactical games with fewer draws
SHARP_OPENINGS = [
    # Sicilian Dragon
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6",
    # King's Gambit (very aggressive)
    "e2e4 e7e5 f2f4",
    # Evan's Gambit
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4",
    # Scotch Gambit
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f1c4",
    # Danish Gambit
    "e2e4 e7e5 d2d4 e5d4 c2c3",
    # Fried Liver Attack setup
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4",
    # Alekhine Defense
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",
    # Scandinavian
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    # Dutch Defense
    "d2d4 f7f5",
    # Starting position (some variety)
    "",
]

def get_sharp_opening() -> chess.Board:
    """Get a board position that tends to be more tactical."""
    board = chess.Board()
    
    if random.random() < 0.2:  # 20% normal start
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

print(f"✅ {len(SHARP_OPENINGS)} sharp opening positions")

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
    
    def predict_move(self, board: chess.Board, temperature: float = 0.5, 
                     add_noise: bool = False) -> Tuple[chess.Move, float]:
        """Predict move with optional noise."""
        self.eval()  # Always eval for inference
        
        # Random move with noise probability
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
# Cell 6: Load and PROPERLY Freeze Model
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

def freeze_module_completely(module: nn.Module):
    """Freeze weights AND BatchNorm running stats."""
    module.eval()  # Set to eval mode (freezes BN running stats)
    for param in module.parameters():
        param.requires_grad = False

def setup_training_mode(network: ChessNet):
    """
    Setup proper training mode:
    - Backbone + Policy: COMPLETELY frozen (including BN)
    - Value head: Trainable
    """
    # Freeze backbone completely
    freeze_module_completely(network.stem)
    freeze_module_completely(network.tower)
    freeze_module_completely(network.policy_head)
    
    # Value head - trainable BUT keep BN in eval mode for stability
    for param in network.value_head.parameters():
        param.requires_grad = True
    
    # Keep value head BN in eval too for stability
    for module in network.value_head.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

setup_training_mode(network)

# Verify
trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
total = sum(p.numel() for p in network.parameters())
print(f"✅ ChessNet: {total:,} params ({trainable:,} trainable)")
print(f"   Stem: FROZEN (including BN)")
print(f"   Tower: FROZEN (including BN)")
print(f"   Policy head: FROZEN (including BN)")
print(f"   Value head: TRAINABLE (BN eval mode)")

# ==============================================================================
# Cell 7: Replay Buffer with Weighted Sampling
# ==============================================================================

class WeightedReplayBuffer:
    """Replay buffer that can weight samples by outcome."""
    
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
        """Sample with weights."""
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            # Weight-based sampling
            weights = np.array([item[2] for item in self.buffer])
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=weights)
            batch = [self.buffer[i] for i in indices]
        
        states, values, weights = zip(*batch)
        return np.array(states), np.array(values, dtype=np.float32), np.array(weights, dtype=np.float32)
    
    def get_win_ratio(self) -> float:
        total = self.stats['wins'] + self.stats['losses']
        if total == 0:
            return 0
        return self.stats['wins'] / total
    
    def __len__(self):
        return len(self.buffer)

buffer = WeightedReplayBuffer(config.buffer_size)

# ==============================================================================
# Cell 8: Asymmetric Self-Play (Forces Decisive Games)
# ==============================================================================

def play_asymmetric_game(network: ChessNet, config: Config) -> Tuple[List, str]:
    """
    Play asymmetric self-play:
    - White: Strong (low temp)
    - Black: Weaker (high temp + noise)
    
    This creates more wins and losses, fewer draws.
    """
    board = get_sharp_opening()
    game_history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        state = encode_board(board)
        player = 1 if board.turn == chess.WHITE else -1
        game_history.append((state, player))
        
        if board.turn == chess.WHITE:
            # White plays strong
            move, _ = network.predict_move(board, temperature=config.white_temperature)
        else:
            # Black plays weaker (higher temp + sometimes random)
            move, _ = network.predict_move(
                board, 
                temperature=config.black_temperature,
                add_noise=True
            )
        
        board.push(move)
        move_count += 1
    
    result = board.result()
    if result == '1-0':
        outcome = 1
    elif result == '0-1':
        outcome = -1
    else:
        outcome = 0
    
    # Create training examples with appropriate weights
    examples = []
    for state, player in game_history:
        value = outcome * player
        
        # Weight: decisive games matter more
        if outcome != 0:
            weight = 1.0
        else:
            weight = config.draw_weight  # Lower weight for draws
        
        examples.append((state, value, weight))
    
    return examples, result

def run_asymmetric_self_play(network: ChessNet, buffer: WeightedReplayBuffer,
                              n_games: int, config: Config) -> Dict:
    """Run self-play with asymmetric strength."""
    network.eval()  # Always eval mode!
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_positions = 0
    
    for _ in tqdm(range(n_games), desc="Self-play", leave=False):
        examples, result = play_asymmetric_game(network, config)
        
        for state, value, weight in examples:
            buffer.add(state, value, weight)
        
        results[result] = results.get(result, 0) + 1
        total_positions += len(examples)
    
    return {
        'results': results,
        'positions': total_positions,
        'win_ratio': buffer.get_win_ratio()
    }

print(f"✅ Asymmetric self-play initialized")

# ==============================================================================
# Cell 9: Trainer with Proper Forward Pass
# ==============================================================================

class Trainer:
    def __init__(self, network: ChessNet, config: Config, buffer: WeightedReplayBuffer):
        self.network = network
        self.config = config
        self.buffer = buffer
        
        # Only optimize value head parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in network.value_head.parameters() if p.requires_grad],
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        
        self.loss_history = []
        self.sign_acc_history = []
    
    def train_batch(self) -> Dict[str, float]:
        """Train on a single batch."""
        # IMPORTANT: Keep frozen parts in eval, only set value head Linear layers to train
        # But since we keep all BN in eval, we can just call network.train() 
        # and then re-freeze BN
        
        # Actually, let's be explicit: keep network in eval, 
        # gradients will still flow for trainable params
        self.network.eval()  # Keep in eval to preserve BN stats
        
        states, target_values, weights = self.buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        target_values = torch.FloatTensor(target_values).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Enable grad for this forward pass
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _, pred_values = self.network(states)
                
                # Weighted MSE loss
                loss = (weights * (pred_values.squeeze(-1) - target_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Only clip value head grads
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.network.value_head.parameters() if p.requires_grad], 
            1.0
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Compute sign accuracy
        with torch.no_grad():
            # Only count decisive positions
            decisive_mask = target_values.abs() > 0.5
            if decisive_mask.sum() > 0:
                pred_sign = (pred_values.squeeze(-1)[decisive_mask] > 0).float()
                target_sign = (target_values[decisive_mask] > 0).float()
                sign_acc = (pred_sign == target_sign).float().mean().item()
            else:
                sign_acc = 0.5
        
        return {'loss': loss.item(), 'sign_acc': sign_acc}
    
    def train_epoch(self) -> Dict[str, float]:
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

# ==============================================================================
# Cell 10: Evaluator
# ==============================================================================

def evaluate_vs_random(network: ChessNet, n_games: int) -> Tuple[float, Dict]:
    """Evaluate against random opponent."""
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

def train_v21():
    print("\n" + "=" * 70)
    print("🚀 STARTING v21 - FIXED BATCHNORM + ANTI-DRAW")
    print("=" * 70)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    history = {
        'iterations': [],
        'value_loss': [],
        'sign_accuracy': [],
        'wr_random': [initial_wr],
        'white_wins': [],
        'black_wins': [],
        'draws': [],
        'buffer_win_ratio': [],
    }
    
    best_wr = initial_wr
    start_time = time.time()
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration + 1}/{config.rl_iterations}")
        print(f"{'='*70}")
        
        # 1. Asymmetric Self-Play
        print(f"\n🎮 Asymmetric Self-Play ({config.self_play_games} games)...")
        sp_results = run_asymmetric_self_play(network, buffer, config.self_play_games, config)
        results = sp_results['results']
        
        total = sum(results.values())
        win_pct = results.get('1-0', 0) / max(1, total)
        draw_pct = results.get('1/2-1/2', 0) / max(1, total)
        
        print(f"   Results: W={results.get('1-0',0)} ({win_pct:.0%}) D={results.get('1/2-1/2',0)} ({draw_pct:.0%}) L={results.get('0-1',0)}")
        print(f"   Positions: {sp_results['positions']}, Buffer: {len(buffer)}")
        
        history['white_wins'].append(results.get('1-0', 0))
        history['black_wins'].append(results.get('0-1', 0))
        history['draws'].append(results.get('1/2-1/2', 0))
        history['buffer_win_ratio'].append(sp_results['win_ratio'])
        
        # Check if enough decisive games
        if win_pct < config.min_win_ratio:
            print(f"   ⚠️ Low win ratio: {win_pct:.0%} (need > {config.min_win_ratio:.0%})")
        
        # 2. Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training (value head only, network in EVAL mode)...")
            train_metrics = trainer.train_epoch()
            
            print(f"   Value Loss: {train_metrics['loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            
            history['iterations'].append(iteration + 1)
            history['value_loss'].append(train_metrics['loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
        
        # 3. Evaluation
        if (iteration + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr, details = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr)
            
            delta = (wr - initial_wr) * 100
            status = "✅" if wr >= initial_wr - 0.02 else "⚠️"
            print(f"   {status} WR vs Random: {wr:.0%} ({delta:+.1f}%)")
            print(f"      W:{details['wins']} D:{details['draws']} L:{details['losses']}")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v21_best.pt')
                print(f"   ✨ New best! Saved.")
            
            # Safety check
            if wr < config.min_wr_threshold:
                print(f"\n🛑 WR dropped below {config.min_wr_threshold:.0%}! Stopping.")
                break
        
        # Timing
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        eta = (config.rl_iterations - iteration - 1) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | Total: {total_time/60:.1f}min | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), '/kaggle/working/chess_v21_final.pt')
    print(f"\n💾 Saved: chess_v21_final.pt")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    final_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {final_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    return history

# ==============================================================================
# Cell 12: Run Training
# ==============================================================================

if __name__ == "__main__":
    history = train_v21()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if history['value_loss']:
        axes[0, 0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0, 0].set_title('Value Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)
    
    if history['sign_accuracy']:
        axes[0, 1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random')
        axes[0, 1].set_title('Sign Accuracy (Decisive Positions)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if history['wr_random']:
        axes[0, 2].plot(history['wr_random'], 'b-o')
        axes[0, 2].axhline(y=history['wr_random'][0], color='g', linestyle='--', label='Initial')
        axes[0, 2].set_title('Win Rate vs Random')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    if history['white_wins']:
        x = range(len(history['white_wins']))
        axes[1, 0].bar(x, history['white_wins'], label='White', alpha=0.8, color='green')
        axes[1, 0].bar(x, history['draws'], bottom=history['white_wins'], label='Draw', alpha=0.8, color='gray')
        axes[1, 0].set_title('Self-Play Results')
        axes[1, 0].legend()
    
    if history['buffer_win_ratio']:
        axes[1, 1].plot(history['buffer_win_ratio'], 'purple')
        axes[1, 1].set_title('Buffer Win Ratio')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].text(0.5, 0.5, f"Initial: {history['wr_random'][0]:.0%}\nFinal: {history['wr_random'][-1]:.0%}", 
                    ha='center', va='center', fontsize=14, transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v21.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("🎉 v21 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   Initial WR: {history['wr_random'][0]:.0%}")
    print(f"   Final WR:   {history['wr_random'][-1]:.0%}")
    print(f"   Change:     {(history['wr_random'][-1] - history['wr_random'][0])*100:+.1f}%")
