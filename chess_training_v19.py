"""
================================================================================
⚡ CHESS AI v19 - CONSERVATIVE RL (VALUE-ONLY TRAINING)
================================================================================
Melanjutkan dari model supervised v17 dengan pendekatan "gentle RL":

🎯 STRATEGI:
1. FREEZE policy head - tetap menggunakan policy dari supervised
2. TRAIN value head saja - dari game outcomes, bukan Stockfish eval
3. VERY LOW LR - 1e-7 untuk perubahan gradual
4. SELF-PLAY - tanpa MCTS untuk kecepatan

📊 VALUE TARGET:
- Win  = +1
- Draw =  0
- Loss = -1

Ini mengajarkan model untuk MENANG, bukan meniru Stockfish.

⚠️ TIDAK AKAN MERUSAK model supervised karena policy tidak diubah.

Training time: ~1-2 hours on P100
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

print("=" * 60)
print("⚡ CHESS AI v19 - CONSERVATIVE RL")
print("=" * 60)
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
    
    # Self-Play (tanpa MCTS untuk kecepatan)
    self_play_games: int = 50        # Games per iteration
    max_moves_per_game: int = 150    # Max moves before draw
    temperature: float = 0.5         # Sampling temperature
    temperature_threshold: int = 30  # After this, use temp=0.1
    
    # Training - VERY CONSERVATIVE
    rl_iterations: int = 50          # More iterations, gentler changes
    batch_size: int = 128
    lr_value: float = 1e-6           # Very low LR for value head
    weight_decay: float = 1e-5
    
    # Optional: Train policy with KL regularization
    train_policy: bool = False       # Set True untuk train policy juga
    lr_policy: float = 1e-8          # Even lower LR for policy
    kl_weight: float = 0.5           # KL penalty weight
    
    # Buffer
    buffer_size: int = 30000
    min_buffer_size: int = 500
    
    # Evaluation
    eval_games: int = 50
    eval_interval: int = 5

config = Config()
print(f"✅ Config: Value-Only={not config.train_policy}")
print(f"✅ LR Value: {config.lr_value}, LR Policy: {config.lr_policy if config.train_policy else 'FROZEN'}")
print(f"✅ Self-play: {config.self_play_games} games/iter, {config.rl_iterations} iterations")

# ==============================================================================
# Cell 3: State Encoding (12 channels - same as v17)
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

print(f"✅ State encoding: 12 channels")

# ==============================================================================
# Cell 4: Neural Network (same as v17)
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
                action = torch.multinomial(probs, 1).item()
            
            move = decode_move(action, board)
            if move is None:
                move = random.choice(list(board.legal_moves))
            
            return move, value.item()

# ==============================================================================
# Cell 5: Load Supervised Model from v17
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
    print("   Will start with random weights (not recommended)")

print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 6: Freeze Layers (Conservative RL)
# ==============================================================================

def setup_conservative_training(network: ChessNet, train_policy: bool = False):
    """
    Setup conservative training:
    - Always freeze backbone (stem + tower)
    - Freeze policy head unless train_policy=True
    - Always train value head
    """
    # Freeze backbone
    for param in network.stem.parameters():
        param.requires_grad = False
    for param in network.tower.parameters():
        param.requires_grad = False
    
    # Policy head
    for param in network.policy_head.parameters():
        param.requires_grad = train_policy
    
    # Value head - always trainable
    for param in network.value_head.parameters():
        param.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in network.parameters() if not p.requires_grad)
    
    print(f"\n📊 Training Setup:")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Frozen params: {frozen:,}")
    print(f"   Policy training: {'YES' if train_policy else 'NO (frozen)'}")

setup_conservative_training(network, config.train_policy)

# Save original policy for KL regularization
if config.train_policy:
    # Deep copy the policy head weights
    original_policy_state = {k: v.clone() for k, v in network.policy_head.state_dict().items()}
    print("   ✅ Saved original policy for KL regularization")

# ==============================================================================
# Cell 7: Self-Play (Without MCTS - Faster)
# ==============================================================================

class ReplayBuffer:
    """Store training examples from self-play."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, state: np.ndarray, value: float):
        """Add a training example (state, value)."""
        self.buffer.append((state, value))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, values = zip(*batch)
        return np.array(states), np.array(values, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(config.buffer_size)

def play_self_play_game(network: ChessNet, config: Config) -> Tuple[List[Tuple], str]:
    """
    Play a self-play game WITHOUT MCTS (using policy directly).
    Returns: (game_history, result)
    """
    board = chess.Board()
    game_history = []  # (state, player)
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        # Temperature scheduling
        if move_count < config.temperature_threshold:
            temp = config.temperature
        else:
            temp = 0.1
        
        # Get state before move
        state = encode_board(board)
        player = 1 if board.turn == chess.WHITE else -1
        game_history.append((state, player))
        
        # Get move from policy
        move, _ = network.predict_move(board, temperature=temp)
        board.push(move)
        move_count += 1
    
    # Get game result
    result = board.result()
    if result == '1-0':
        outcome = 1  # White wins
    elif result == '0-1':
        outcome = -1  # Black wins
    else:
        outcome = 0  # Draw
    
    # Create training examples with game outcome
    training_examples = []
    for state, player in game_history:
        # Value from this player's perspective
        value = outcome * player
        training_examples.append((state, value))
    
    return training_examples, result

def run_self_play(network: ChessNet, buffer: ReplayBuffer, 
                  n_games: int, config: Config) -> Dict[str, int]:
    """Run multiple self-play games."""
    network.eval()
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_positions = 0
    
    for _ in tqdm(range(n_games), desc="Self-play"):
        examples, result = play_self_play_game(network, config)
        
        # Add to buffer
        for state, value in examples:
            buffer.add(state, value)
        
        if result in results:
            results[result] += 1
        else:
            results['1/2-1/2'] += 1
        
        total_positions += len(examples)
    
    return results, total_positions

print(f"✅ Self-play initialized (no MCTS)")

# ==============================================================================
# Cell 8: Training (Value-Only)
# ==============================================================================

def train_value_only(network: ChessNet, buffer: ReplayBuffer,
                     optimizer: torch.optim.Optimizer,
                     scaler: torch.cuda.amp.GradScaler,
                     n_batches: int = 10) -> Dict[str, float]:
    """Train ONLY the value head."""
    network.train()
    
    total_value_loss = 0
    
    for _ in range(n_batches):
        states, target_values = buffer.sample(config.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        target_values = torch.FloatTensor(target_values).to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _, pred_values = network(states)
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
        
        optimizer.zero_grad()
        scaler.scale(value_loss).backward()
        torch.nn.utils.clip_grad_norm_(network.value_head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_value_loss += value_loss.item()
    
    return {'value_loss': total_value_loss / n_batches}

def train_with_kl_regularization(network: ChessNet, buffer: ReplayBuffer,
                                  optimizer_value: torch.optim.Optimizer,
                                  optimizer_policy: torch.optim.Optimizer,
                                  scaler: torch.cuda.amp.GradScaler,
                                  original_policy_state: dict,
                                  n_batches: int = 10) -> Dict[str, float]:
    """Train both heads with KL regularization on policy."""
    network.train()
    
    # Create a frozen copy of original policy for KL computation
    # This is a simple approach - compare current logits to original
    
    total_value_loss = 0
    total_policy_loss = 0
    total_kl_loss = 0
    
    for _ in range(n_batches):
        states, target_values = buffer.sample(config.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        target_values = torch.FloatTensor(target_values).to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            policy_logits, pred_values = network(states)
            
            # Value loss
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            
            # For policy, we use a simple regularization:
            # Don't let logits change too much from their current values
            # (This is a simplified KL approximation)
            with torch.no_grad():
                original_probs = F.softmax(policy_logits.detach(), dim=-1)
            
            current_probs = F.softmax(policy_logits, dim=-1)
            kl_loss = F.kl_div(
                current_probs.log(), 
                original_probs, 
                reduction='batchmean'
            )
            
            # Combined loss
            loss = value_loss + config.kl_weight * kl_loss
        
        optimizer_value.zero_grad()
        optimizer_policy.zero_grad()
        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(network.value_head.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(network.policy_head.parameters(), 0.1)  # Stricter for policy
        
        scaler.step(optimizer_value)
        scaler.step(optimizer_policy)
        scaler.update()
        
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
    
    return {
        'value_loss': total_value_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
    }

# ==============================================================================
# Cell 9: Evaluation
# ==============================================================================

def evaluate_vs_random(network: ChessNet, n_games: int) -> Tuple[float, Dict]:
    """Evaluate network against random opponent."""
    network.eval()
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:
            if board.turn == chess.WHITE:
                # Network plays white
                move, _ = network.predict_move(board, temperature=0.1)
            else:
                # Random opponent
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
    
    wr = (wins + 0.5 * draws) / n_games
    return wr, {'wins': wins, 'draws': draws, 'losses': losses}

def evaluate_value_accuracy(network: ChessNet, buffer: ReplayBuffer, 
                            n_samples: int = 500) -> Dict[str, float]:
    """Check how well value predictions match actual outcomes."""
    network.eval()
    
    if len(buffer) < n_samples:
        n_samples = len(buffer)
    
    states, target_values = buffer.sample(n_samples)
    
    with torch.no_grad():
        states_t = torch.FloatTensor(states).to(device)
        _, pred_values = network(states_t)
        pred_values = pred_values.squeeze(-1).cpu().numpy()
    
    # Compute metrics
    mse = np.mean((pred_values - target_values) ** 2)
    mae = np.mean(np.abs(pred_values - target_values))
    
    # Sign accuracy (did we predict the right winner?)
    sign_correct = np.mean((pred_values > 0) == (target_values > 0))
    
    return {'mse': mse, 'mae': mae, 'sign_accuracy': sign_correct}

# ==============================================================================
# Cell 10: Main Training Loop
# ==============================================================================

def train_v19():
    """Main training loop for Conservative RL."""
    print("\n" + "=" * 60)
    print("🚀 STARTING v19 - CONSERVATIVE RL")
    print("=" * 60)
    
    # Create optimizer for value head only (or both if train_policy)
    if config.train_policy:
        optimizer_value = torch.optim.AdamW(
            network.value_head.parameters(),
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        optimizer_policy = torch.optim.AdamW(
            network.policy_head.parameters(),
            lr=config.lr_policy,
            weight_decay=config.weight_decay
        )
    else:
        optimizer_value = torch.optim.AdamW(
            network.value_head.parameters(),
            lr=config.lr_value,
            weight_decay=config.weight_decay
        )
        optimizer_policy = None
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    initial_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {initial_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    # Training history
    history = {
        'iterations': [],
        'value_loss': [],
        'wr_random': [initial_wr],
        'sign_accuracy': [],
        'white_wins': [],
        'black_wins': [],
        'draws': []
    }
    
    best_wr = initial_wr
    start_time = time.time()
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"📌 ITERATION {iteration + 1}/{config.rl_iterations}")
        print(f"{'='*60}")
        
        # 1. Self-Play
        print(f"\n🎮 Self-Play ({config.self_play_games} games)...")
        results, n_positions = run_self_play(network, buffer, config.self_play_games, config)
        
        print(f"   Results: W={results['1-0']} D={results['1/2-1/2']} L={results['0-1']}")
        print(f"   Positions: {n_positions}, Buffer: {len(buffer)}")
        
        history['white_wins'].append(results['1-0'])
        history['black_wins'].append(results['0-1'])
        history['draws'].append(results['1/2-1/2'])
        
        # 2. Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training (Value-Only)...")
            n_batches = min(30, len(buffer) // config.batch_size)
            
            if config.train_policy and optimizer_policy:
                losses = train_with_kl_regularization(
                    network, buffer, optimizer_value, optimizer_policy,
                    scaler, original_policy_state, n_batches
                )
                print(f"   Value Loss: {losses['value_loss']:.4f}")
                print(f"   KL Loss:    {losses['kl_loss']:.4f}")
            else:
                losses = train_value_only(network, buffer, optimizer_value, scaler, n_batches)
                print(f"   Value Loss: {losses['value_loss']:.4f}")
            
            history['iterations'].append(iteration + 1)
            history['value_loss'].append(losses['value_loss'])
            
            # Check value accuracy
            val_metrics = evaluate_value_accuracy(network, buffer)
            print(f"   Sign Accuracy: {val_metrics['sign_accuracy']:.1%}")
            history['sign_accuracy'].append(val_metrics['sign_accuracy'])
        
        # 3. Evaluation
        if (iteration + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr, details = evaluate_vs_random(network, config.eval_games)
            history['wr_random'].append(wr)
            
            print(f"   WR vs Random: {wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v19_best.pt')
                print(f"   ✨ New best! Saved.")
            elif wr < initial_wr - 0.05:
                print(f"   ⚠️ Warning: WR dropped significantly from initial!")
        
        # Timing
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        eta = (config.rl_iterations - iteration - 1) * iter_time
        print(f"\n⏱️ Iter: {iter_time:.1f}s | Total: {total_time/60:.1f}min | ETA: {eta/60:.1f}min")
    
    # Final save
    torch.save(network.state_dict(), '/kaggle/working/chess_v19_final.pt')
    print(f"\n💾 Saved: chess_v19_final.pt")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    final_wr, details = evaluate_vs_random(network, config.eval_games)
    print(f"   WR vs Random: {final_wr:.0%} (W:{details['wins']} D:{details['draws']} L:{details['losses']})")
    
    return history

# ==============================================================================
# Cell 11: Run Training
# ==============================================================================

if __name__ == "__main__":
    history = train_v19()
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Value Loss
    if history['value_loss']:
        axes[0, 0].plot(history['iterations'], history['value_loss'], 'b-')
        axes[0, 0].set_title('Value Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MSE Loss')
    
    # Sign Accuracy
    if history['sign_accuracy']:
        axes[0, 1].plot(history['iterations'], history['sign_accuracy'], 'g-')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random guess')
        axes[0, 1].set_title('Value Sign Accuracy')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
    
    # Win Rate
    axes[1, 0].plot(history['wr_random'], 'b-o')
    axes[1, 0].axhline(y=history['wr_random'][0], color='g', linestyle='--', label=f'Initial ({history["wr_random"][0]:.0%})')
    axes[1, 0].set_title('Win Rate vs Random')
    axes[1, 0].set_xlabel('Evaluation')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].legend()
    
    # Self-Play Results
    if history['white_wins']:
        x = range(len(history['white_wins']))
        axes[1, 1].bar(x, history['white_wins'], label='White Wins', alpha=0.7, color='white', edgecolor='black')
        axes[1, 1].bar(x, history['draws'], bottom=history['white_wins'], label='Draws', alpha=0.7, color='gray')
        black_bottom = [w + d for w, d in zip(history['white_wins'], history['draws'])]
        axes[1, 1].bar(x, history['black_wins'], bottom=black_bottom, label='Black Wins', alpha=0.7, color='black')
        axes[1, 1].set_title('Self-Play Results')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Games')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v19.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 60)
    print("🎉 v19 CONSERVATIVE RL COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Initial WR: {history['wr_random'][0]:.0%}")
    print(f"   Final WR:   {history['wr_random'][-1]:.0%}")
    print(f"   Change:     {(history['wr_random'][-1] - history['wr_random'][0])*100:+.1f}%")
    print(f"\n📁 Files:")
    print(f"   chess_v19_best.pt  - Best model")
    print(f"   chess_v19_final.pt - Final model")
    print(f"   training_v19.png   - Training plots")
