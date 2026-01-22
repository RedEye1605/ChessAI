"""
================================================================================
⚡ CHESS AI v18 - SELF-PLAY + MCTS
================================================================================
Melanjutkan dari model supervised v17 dengan:

1. LOAD MODEL SUPERVISED v17:
   - Tidak training dari nol
   - Melanjutkan dari model 94% WR vs random

2. MONTE CARLO TREE SEARCH (MCTS):
   - 50-100 simulations per move
   - UCB1 selection for exploration/exploitation
   - Network sebagai leaf evaluator
   - Policy target = MCTS visit distribution

3. SELF-PLAY REINFORCEMENT LEARNING:
   - Model bermain melawan dirinya sendiri
   - Selalu dapat sinyal seimbang (50% win/50% lose)
   - Value target = game outcome {-1, 0, +1}

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
import math
import chess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
import pickle
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
print("⚡ CHESS AI v18 - SELF-PLAY + MCTS")
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
    
    # MCTS - REDUCED for speed
    mcts_simulations: int = 15       # 50→15 for ~3x speedup
    mcts_cpuct: float = 1.5          # Exploration constant
    mcts_temperature: float = 1.0    # Temperature for move selection
    mcts_temperature_threshold: int = 30  # After this many moves, use temp=0
    
    # Self-Play - REDUCED for speed
    self_play_games: int = 30        # 100→30 for ~3x speedup
    max_moves_per_game: int = 150    # 200→150 to end games faster
    
    # Training
    rl_iterations: int = 30          # More iterations to compensate
    batch_size: int = 256
    lr: float = 1e-5                 # Low LR for fine-tuning
    weight_decay: float = 1e-4
    
    # Buffer
    buffer_size: int = 50000         # Smaller buffer for faster sampling
    min_buffer_size: int = 500       # Start training sooner
    
    # Evaluation
    eval_games: int = 30             # Faster evaluation
    eval_interval: int = 3           # Evaluate every 3 iterations

config = Config()
print(f"✅ Config: {config.input_channels} channels, {config.blocks} blocks")
print(f"✅ MCTS: {config.mcts_simulations} simulations, cpuct={config.mcts_cpuct}")
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

def get_legal_actions(board: chess.Board) -> List[int]:
    """Get list of legal action indices."""
    return [encode_move(m) for m in board.legal_moves]

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
    
    def predict(self, state: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get policy probabilities and value for a single state."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            return probs, value.item()

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
# Cell 6: MCTS Implementation
# ==============================================================================

class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, board: chess.Board, parent=None, action=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior    # Prior probability from policy network
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, cpuct: float) -> float:
        """Upper Confidence Bound score for selection."""
        prior_score = cpuct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        value_score = -self.value  # Negate because we want opponent's perspective
        return value_score + prior_score
    
    def select_child(self, cpuct: float) -> 'MCTSNode':
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(self.visit_count, cpuct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray):
        """Expand node by creating children for all legal moves."""
        for move in self.board.legal_moves:
            action = encode_move(move)
            if action not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[action] = MCTSNode(
                    board=new_board,
                    parent=self,
                    action=action,
                    prior=policy_probs[action]
                )
        self.is_expanded = True
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)  # Negate for opponent

class MCTS:
    """Monte Carlo Tree Search."""
    
    def __init__(self, network: ChessNet, config: Config):
        self.network = network
        self.config = config
    
    def search(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from the given board position.
        Returns: (policy, value)
            - policy: visit count distribution over actions
            - value: estimated value of root position
        """
        # Create root node
        root = MCTSNode(board)
        
        # Get initial policy and value from network
        state = encode_board(board)
        mask = get_legal_mask(board)
        policy_probs, _ = self.network.predict(state, mask)
        
        # Expand root
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(self.config.mcts_simulations):
            node = root
            path = [node]
            
            # 1. SELECT: Traverse tree until we reach unexpanded node
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child(self.config.mcts_cpuct)
                path.append(node)
            
            # 2. EVALUATE: Get value from network or terminal state
            if node.board.is_game_over():
                # Terminal state - get actual game result
                result = node.board.result()
                if result == '1-0':
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == '0-1':
                    value = -1.0 if node.board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0
            else:
                # Non-terminal - evaluate with network
                state = encode_board(node.board)
                mask = get_legal_mask(node.board)
                policy_probs, value = self.network.predict(state, mask)
                
                # 3. EXPAND
                node.expand(policy_probs)
            
            # 4. BACKPROPAGATE
            node.backpropagate(value)
        
        # Compute policy from visit counts
        visit_counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # Normalize to get probability distribution
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        return policy, root.value
    
    def select_action(self, board: chess.Board, temperature: float = 1.0) -> Tuple[int, np.ndarray, float]:
        """
        Run MCTS and select action based on temperature.
        Returns: (action, policy, value)
        """
        policy, value = self.search(board)
        
        # Get legal actions
        legal_actions = get_legal_actions(board)
        
        if temperature < 0.1 or len(legal_actions) == 1:
            # Greedy selection or only one legal move
            action = int(np.argmax(policy))
        else:
            # Sample from policy with temperature
            # Use float64 for numerical stability
            legal_probs = policy[legal_actions].astype(np.float64)
            
            # Handle all-zero case (use uniform)
            if legal_probs.sum() < 1e-10:
                legal_probs = np.ones(len(legal_actions), dtype=np.float64)
            
            # Apply temperature
            legal_probs = np.power(legal_probs, 1.0 / temperature)
            
            # Normalize and clip for numerical stability
            legal_probs = legal_probs / legal_probs.sum()
            legal_probs = np.clip(legal_probs, 0, 1)
            legal_probs = legal_probs / legal_probs.sum()  # Re-normalize after clip
            
            # Final safety check
            if not np.isclose(legal_probs.sum(), 1.0):
                legal_probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            action = legal_actions[action_idx]
        
        return action, policy, value

print(f"✅ MCTS initialized: {config.mcts_simulations} simulations/move")

# ==============================================================================
# Cell 7: Self-Play Data Generation
# ==============================================================================

class ReplayBuffer:
    """Store training examples from self-play."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a training example."""
        self.buffer.append((state, policy, value))
        
        # Remove oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (np.array(states), np.array(policies), np.array(values, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

buffer = ReplayBuffer(config.buffer_size)

def play_self_play_game(network: ChessNet, mcts: MCTS, config: Config) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Play a self-play game and collect training data.
    Returns: list of (state, policy, outcome)
    """
    board = chess.Board()
    game_history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < config.max_moves_per_game:
        # Determine temperature
        if move_count < config.mcts_temperature_threshold:
            temperature = config.mcts_temperature
        else:
            temperature = 0.1  # Near-greedy in endgame
        
        # Get state before move
        state = encode_board(board)
        
        # Run MCTS to get policy
        action, policy, _ = mcts.select_action(board, temperature)
        
        # Store (state, policy, player)
        # Player: 1 for white, -1 for black
        player = 1 if board.turn == chess.WHITE else -1
        game_history.append((state, policy, player))
        
        # Make move
        move = decode_move(action, board)
        if move is None:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        move_count += 1
    
    # Determine game outcome
    result = board.result()
    if result == '1-0':
        outcome = 1  # White wins
    elif result == '0-1':
        outcome = -1  # Black wins
    else:
        outcome = 0  # Draw
    
    # Convert to training examples with final outcomes
    training_examples = []
    for state, policy, player in game_history:
        # Value from this player's perspective
        value = outcome * player
        training_examples.append((state, policy, value))
    
    return training_examples, result

def run_self_play(network: ChessNet, mcts: MCTS, buffer: ReplayBuffer, 
                  n_games: int) -> Dict[str, int]:
    """Run multiple self-play games and add to buffer."""
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_positions = 0
    
    for game_idx in tqdm(range(n_games), desc="Self-play"):
        examples, result = play_self_play_game(network, mcts, config)
        
        # Add to buffer
        for state, policy, value in examples:
            buffer.add(state, policy, value)
        
        # Track result
        if result in results:
            results[result] += 1
        else:
            results['1/2-1/2'] += 1
        
        total_positions += len(examples)
    
    return results, total_positions

# ==============================================================================
# Cell 8: Training
# ==============================================================================

def train_on_buffer(network: ChessNet, buffer: ReplayBuffer, 
                    optimizer: torch.optim.Optimizer, 
                    scaler: torch.cuda.amp.GradScaler,
                    n_batches: int = 10) -> Dict[str, float]:
    """Train network on data from replay buffer."""
    network.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    
    for _ in range(n_batches):
        states, target_policies, target_values = buffer.sample(config.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        target_policies = torch.FloatTensor(target_policies).to(device)
        target_values = torch.FloatTensor(target_values).to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            # Forward pass
            policy_logits, pred_values = network(states)
            
            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()
            
            # Value loss: MSE
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            
            # Total loss
            loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
    
    return {
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches,
        'total_loss': total_loss / n_batches
    }

# ==============================================================================
# Cell 9: Evaluation
# ==============================================================================

def evaluate_vs_random(network: ChessNet, mcts: MCTS, n_games: int) -> float:
    """Evaluate network against random opponent."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:
            if board.turn == chess.WHITE:
                # Network plays white with MCTS (but faster - fewer simulations)
                action, _, _ = mcts.select_action(board, temperature=0.1)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                # Random opponent
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            wins += 0.5  # Count draw as half win
    
    return wins / n_games

def quick_evaluate(network: ChessNet, n_games: int = 20) -> float:
    """Quick evaluation without MCTS (just policy)."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:
            if board.turn == chess.WHITE:
                # Network plays white (greedy policy)
                state = encode_board(board)
                mask = get_legal_mask(board)
                probs, _ = network.predict(state, mask)
                action = int(np.argmax(probs * mask))
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                # Random opponent
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            wins += 0.5
    
    return wins / n_games

# ==============================================================================
# Cell 10: Main Training Loop
# ==============================================================================

def train_v18():
    """Main training loop for v18."""
    print("\n" + "=" * 60)
    print("🚀 STARTING v18 TRAINING - SELF-PLAY + MCTS")
    print("=" * 60)
    
    # Initialize
    mcts = MCTS(network, config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Initial evaluation
    print("\n📊 Initial Evaluation (without MCTS):")
    initial_wr = quick_evaluate(network, 50)
    print(f"   WR vs Random: {initial_wr:.0%}")
    
    # Training history
    history = {
        'iterations': [],
        'policy_loss': [],
        'value_loss': [],
        'wr_random': [initial_wr],
        'buffer_size': [],
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
        results, n_positions = run_self_play(network, mcts, buffer, config.self_play_games)
        
        print(f"   Results: W={results['1-0']} D={results['1/2-1/2']} L={results['0-1']}")
        print(f"   Positions collected: {n_positions}")
        print(f"   Buffer size: {len(buffer)}")
        
        history['white_wins'].append(results['1-0'])
        history['black_wins'].append(results['0-1'])
        history['draws'].append(results['1/2-1/2'])
        history['buffer_size'].append(len(buffer))
        
        # 2. Training
        if len(buffer) >= config.min_buffer_size:
            print(f"\n📚 Training...")
            n_batches = max(10, len(buffer) // config.batch_size)
            n_batches = min(n_batches, 50)  # Cap at 50 batches
            
            losses = train_on_buffer(network, buffer, optimizer, scaler, n_batches)
            
            print(f"   Policy Loss: {losses['policy_loss']:.4f}")
            print(f"   Value Loss:  {losses['value_loss']:.4f}")
            print(f"   Total Loss:  {losses['total_loss']:.4f}")
            
            history['iterations'].append(iteration + 1)
            history['policy_loss'].append(losses['policy_loss'])
            history['value_loss'].append(losses['value_loss'])
        else:
            print(f"\n⏳ Buffer too small ({len(buffer)}/{config.min_buffer_size}), skipping training")
        
        # 3. Evaluation
        if (iteration + 1) % config.eval_interval == 0:
            print(f"\n📊 Evaluation...")
            wr = quick_evaluate(network, config.eval_games)
            history['wr_random'].append(wr)
            
            print(f"   WR vs Random: {wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                torch.save(network.state_dict(), '/kaggle/working/chess_v18_best.pt')
                print(f"   ✨ New best! Saved.")
        
        # Timing
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        print(f"\n⏱️ Iteration time: {iter_time/60:.1f} min | Total: {total_time/60:.1f} min")
    
    # Final save
    torch.save(network.state_dict(), '/kaggle/working/chess_v18_final.pt')
    print(f"\n💾 Saved: chess_v18_final.pt")
    
    # Final evaluation with MCTS
    print(f"\n📊 Final Evaluation (with MCTS):")
    # Use fewer simulations for faster eval
    mcts_eval = MCTS(network, Config(mcts_simulations=20))
    final_wr = evaluate_vs_random(network, mcts_eval, 30)
    print(f"   WR vs Random (with MCTS): {final_wr:.0%}")
    
    return history

# ==============================================================================
# Cell 11: Run Training
# ==============================================================================

if __name__ == "__main__":
    history = train_v18()
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Policy Loss
    if history['policy_loss']:
        axes[0, 0].plot(history['iterations'], history['policy_loss'], 'b-')
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
    
    # Value Loss
    if history['value_loss']:
        axes[0, 1].plot(history['iterations'], history['value_loss'], 'r-')
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
    
    # Win Rate
    axes[1, 0].plot(history['wr_random'], 'g-o')
    axes[1, 0].axhline(y=0.9, color='b', linestyle='--', label='Target 90%')
    axes[1, 0].set_title('Win Rate vs Random')
    axes[1, 0].set_xlabel('Evaluation')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].legend()
    
    # Self-Play Results
    if history['white_wins']:
        x = range(len(history['white_wins']))
        axes[1, 1].bar(x, history['white_wins'], label='White Wins', alpha=0.7)
        axes[1, 1].bar(x, history['draws'], bottom=history['white_wins'], label='Draws', alpha=0.7)
        black_bottom = [w + d for w, d in zip(history['white_wins'], history['draws'])]
        axes[1, 1].bar(x, history['black_wins'], bottom=black_bottom, label='Black Wins', alpha=0.7)
        axes[1, 1].set_title('Self-Play Results')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Games')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_v18.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 60)
    print("🎉 v18 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results Summary:")
    print(f"   Initial WR: {history['wr_random'][0]:.0%}")
    print(f"   Final WR: {history['wr_random'][-1]:.0%}")
    print(f"\n📁 Files saved:")
    print(f"   chess_v18_best.pt - Best during training")
    print(f"   chess_v18_final.pt - Final model")
    print(f"   training_v18.png - Training plots")
