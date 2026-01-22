"""
================================================================================
⚡ CHESS AI v22 - CONTINUE TRAINING FROM v21
================================================================================

Melanjutkan training dari model v21_final.pt
- Load model dari v21 (bukan v17 supervised)
- Lanjutkan training value head
- Start iteration dari 101 (melanjutkan dari 100)

Training time: ~1 hour for 50 more iterations
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
print("⚡ CHESS AI v22 - CONTINUE FROM v21")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")

# ==============================================================================
# Cell 2: Configuration - CONTINUE FROM v21
# ==============================================================================

@dataclass
class Config:
    # Network
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Self-Play
    self_play_games: int = 100
    max_moves_per_game: int = 100
    
    # Asymmetric play
    white_temperature: float = 0.3
    black_temperature: float = 0.8
    black_noise_prob: float = 0.1
    
    # Training - CONTINUING
    start_iteration: int = 101      # Continue from iteration 101
    rl_iterations: int = 50         # Run 50 more iterations (101-150)
    batch_size: int = 256
    lr_value: float = 5e-7          # Even lower LR for fine-tuning
    weight_decay: float = 1e-5
    batches_per_iter: int = 20
    
    # Draw handling
    draw_weight: float = 0.3
    min_win_ratio: float = 0.15
    
    # Buffer
    buffer_size: int = 100000
    min_buffer_size: int = 2000
    
    # Evaluation
    eval_games: int = 50
    eval_interval: int = 5
    min_wr_threshold: float = 0.90

config = Config()
print(f"\n📋 Configuration (CONTINUE TRAINING):")
print(f"   Starting from iteration: {config.start_iteration}")
print(f"   Running {config.rl_iterations} more iterations")
print(f"   LR: {config.lr_value} (lowered for fine-tuning)")

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
# Cell 5: Load v21 Model (NOT v17!)
# ==============================================================================

# Try multiple paths for the v21 model
V21_MODEL_PATHS = [
    '/kaggle/input/supervisedmodel/pytorch/default/4/chess_v21_final.pt',  # User's dataset
    '/kaggle/input/v21model/pytorch/default/1/chess_v21_final.pt',  # Kaggle dataset
    '/kaggle/working/chess_v21_final.pt',                            # Previous run output
    '/kaggle/input/v21model/chess_v21_final.pt',
    '/kaggle/input/chess-v21/chess_v21_final.pt',
]

network = ChessNet(in_channels=config.input_channels, 
                   filters=config.filters, 
                   blocks=config.blocks).to(device)

loaded = False
for path in V21_MODEL_PATHS:
    if os.path.exists(path):
        network.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded v21 model from: {path}")
        loaded = True
        break

if not loaded:
    print("⚠️ v21 model not found! Trying fallback paths...")
    # List available files
    for search_dir in ['/kaggle/input', '/kaggle/working']:
        if os.path.exists(search_dir):
            print(f"   Files in {search_dir}:")
            for f in os.listdir(search_dir):
                print(f"      {f}")

# Freeze everything except value head
def freeze_module_completely(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

freeze_module_completely(network.stem)
freeze_module_completely(network.tower)
freeze_module_completely(network.policy_head)

for param in network.value_head.parameters():
    param.requires_grad = True

for module in network.value_head.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"✅ Trainable params: {trainable:,}")

# ==============================================================================
# Cell 6: Buffer
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
# Cell 7: Self-Play & Training
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
# Cell 8: Main Training Loop (CONTINUE)
# ==============================================================================

def train_v22():
    print("\n" + "=" * 70)
    print(f"🚀 CONTINUING TRAINING FROM ITERATION {config.start_iteration}")
    print("=" * 70)
    
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
    }
    
    best_wr = initial_wr
    start_time = time.time()
    
    for i in range(config.rl_iterations):
        iteration = config.start_iteration + i
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📌 ITERATION {iteration}/{config.start_iteration + config.rl_iterations - 1}")
        print(f"{'='*70}")
        
        # Self-Play
        print(f"\n🎮 Self-Play ({config.self_play_games} games)...")
        sp_results = run_self_play(network, buffer, config.self_play_games, config)
        results = sp_results['results']
        
        total = sum(results.values())
        win_pct = results.get('1-0', 0) / max(1, total)
        
        print(f"   W={results.get('1-0',0)} ({win_pct:.0%}) D={results.get('1/2-1/2',0)} L={results.get('0-1',0)}")
        print(f"   Buffer: {len(buffer)}")
        
        history['white_wins'].append(results.get('1-0', 0))
        history['black_wins'].append(results.get('0-1', 0))
        history['draws'].append(results.get('1/2-1/2', 0))
        
        # Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training...")
            train_metrics = trainer.train_epoch()
            print(f"   Value Loss: {train_metrics['loss']:.4f}")
            print(f"   Sign Accuracy: {train_metrics['sign_acc']:.1%}")
            
            history['iterations'].append(iteration)
            history['value_loss'].append(train_metrics['loss'])
            history['sign_accuracy'].append(train_metrics['sign_acc'])
        
        # Evaluation
        if (i + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr, details = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr)
            
            delta = (wr - initial_wr) * 100
            status = "✅" if wr >= initial_wr - 0.02 else "⚠️"
            print(f"   {status} WR: {wr:.0%} ({delta:+.1f}%)")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v22_best.pt')
                print(f"   ✨ New best! Saved.")
            
            if wr < config.min_wr_threshold:
                print(f"\n🛑 WR dropped below {config.min_wr_threshold:.0%}! Stopping.")
                break
        
        iter_time = time.time() - iter_start
        eta = (config.rl_iterations - i - 1) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.0f}s | ETA: {eta/60:.1f}min")
    
    torch.save(network.state_dict(), '/kaggle/working/chess_v22_final.pt')
    print(f"\n💾 Saved: chess_v22_final.pt")
    
    final_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"\n📊 Final: WR {final_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 9: Run
# ==============================================================================

if __name__ == "__main__":
    history = train_v22()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if history['value_loss']:
        axes[0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0].set_title('Value Loss')
        axes[0].set_xlabel('Iteration')
    
    if history['sign_accuracy']:
        axes[1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[1].axhline(y=0.5, color='r', linestyle='--')
        axes[1].set_title('Sign Accuracy')
    
    if history['wr_random']:
        axes[2].plot(history['wr_random'], 'b-o')
        axes[2].set_title('Win Rate vs Random')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v22.png', dpi=150)
    plt.show()
    
    print("\n🎉 v22 TRAINING COMPLETE!")
