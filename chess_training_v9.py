"""
================================================================================
⚡ CHESS AI v9 - COMPREHENSIVE TRAINING
================================================================================
Complete training pipeline with:
1. Opening Book - Standard openings for first 10 moves
2. Supervised Pre-training - All Lichess + Stockfish data
3. MCTS Self-Play - Model plays against itself with search
4. Value Network - Position evaluation
5. Temperature - Variety in moves to avoid repetitive draws

Target: Intelligent chess play within 12 hours on P100

Author: AI Assistant
Date: 2026-01-18 (v9)
================================================================================
"""

# ==============================================================================
# Cell 1: Setup
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import time
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = torch.cuda.is_available()

print("=" * 60)
print("⚡ CHESS AI v9 - COMPREHENSIVE TRAINING")
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
    filters: int = 128
    blocks: int = 6
    
    # Training
    supervised_epochs: int = 20
    selfplay_iterations: int = 50
    games_per_iteration: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    
    # MCTS
    mcts_simulations: int = 50
    c_puct: float = 1.5
    
    # Temperature (for variety in moves)
    temp_initial: float = 1.0  # High temp = more exploration
    temp_final: float = 0.1   # Low temp = more exploitation
    temp_threshold: int = 30  # After this move, use low temp
    
    # Buffer
    buffer_size: int = 200000
    
    # Opening book
    use_opening_book: bool = True
    opening_book_moves: int = 10

config = Config()
print(f"✅ Config loaded")

# ==============================================================================
# Cell 3: Dependencies
# ==============================================================================

try:
    import chess
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    print("✅ Dependencies loaded!")
except ImportError:
    print("Run: pip install python-chess tqdm matplotlib pandas")

# ==============================================================================
# Cell 4: Opening Book
# ==============================================================================

# Common opening moves for White and Black
OPENING_BOOK = {
    # Starting position - common first moves for White
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -': [
        'e2e4', 'd2d4', 'c2c4', 'g1f3'  # e4, d4, c4, Nf3
    ],
    # After 1.e4 - common responses for Black
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -': [
        'e7e5', 'c7c5', 'e7e6', 'c7c6', 'g8f6'  # e5, Sicilian, French, Caro-Kann, Alekhine
    ],
    # After 1.d4 - common responses
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -': [
        'd7d5', 'g8f6', 'e7e6', 'f7f5'  # d5, Indian, e6, Dutch
    ],
    # After 1.e4 e5 - White's responses
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': [
        'g1f3', 'f1c4', 'b1c3'  # Nf3, Italian setup, Nc3
    ],
    # After 1.e4 e5 2.Nf3 - Black's responses
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -': [
        'b8c6', 'g8f6', 'd7d6'  # Nc6, Petrov, Philidor
    ],
    # After 1.e4 e5 2.Nf3 Nc6 - White's responses
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -': [
        'f1b5', 'f1c4', 'd2d4', 'b1c3'  # Ruy Lopez, Italian, Scotch, Four Knights
    ],
    # Italian Game setup
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -': [
        'f8c5', 'g8f6', 'f8e7'  # Italian responses
    ],
    # After 1.d4 d5 - White's responses
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': [
        'c2c4', 'g1f3', 'c1f4', 'e2e3'  # Queen's Gambit, etc
    ],
    # Queen's Gambit
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -': [
        'e7e6', 'c7c6', 'd5c4'  # QGD, Slav, QGA
    ],
    # After 1.d4 Nf6 - White's responses (Indian systems)
    'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': [
        'c2c4', 'g1f3', 'c1f4', 'c1g5'
    ],
    # Sicilian Defense after 1.e4 c5
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': [
        'g1f3', 'b1c3', 'c2c3', 'd2d4'  # Open Sicilian, Closed, Alapin
    ],
}

def get_opening_book_move(board: 'chess.Board') -> Optional[str]:
    """Get a move from opening book if available."""
    # Get FEN without move counters
    fen_key = ' '.join(board.fen().split()[:4])
    
    if fen_key in OPENING_BOOK:
        moves = OPENING_BOOK[fen_key]
        # Filter legal moves
        legal = [m for m in moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal:
            return random.choice(legal)
    return None

print(f"✅ Opening book: {len(OPENING_BOOK)} positions")

# ==============================================================================
# Cell 5: State Encoder & Action Space
# ==============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board to 8x8x8 tensor."""
    state = np.zeros((8, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_to_channel[piece.piece_type]
            rank = sq // 8
            file = sq % 8
            state[ch, rank, file] = 1.0 if piece.color == chess.WHITE else -1.0
    
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    
    return state

NUM_ACTIONS = 64 * 64

def encode_move(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    from_sq = action // 64
    to_sq = action % 64
    for m in board.legal_moves:
        if m.from_square == from_sq and m.to_square == to_sq:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask

# ==============================================================================
# Cell 6: Neural Network with Value Head
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//8), nn.ReLU(), nn.Linear(ch//8, ch), nn.Sigmoid())
    def forward(self, x):
        y = self.pool(x).view(x.size(0), -1)
        return x * self.fc(y).view(x.size(0), -1, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(self.se(out) + x)

class ChessNet(nn.Module):
    """Network with Policy and Value heads."""
    def __init__(self, in_ch=8, filters=128, blocks=6, actions=NUM_ACTIONS):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
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
    
    def forward(self, x, mask=None):
        x = self.res_blocks(self.input_conv(x))
        p = self.policy_head(x)
        v = self.value_head(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)
        return p, v
    
    def predict(self, state: np.ndarray, mask: np.ndarray, temperature: float = 1.0):
        """Predict with temperature for variety."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            logits, value = self(x, m)
            
            if temperature <= 0.01:
                # Deterministic
                probs = torch.zeros_like(logits)
                probs[0, logits.argmax()] = 1.0
            else:
                # Apply temperature
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
            
            return probs.squeeze(0).cpu().numpy(), value.item()

network = ChessNet(filters=config.filters, blocks=config.blocks).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 7: MCTS Implementation
# ==============================================================================

class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """Upper Confidence Bound score."""
        prior_score = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + prior_score

class MCTS:
    """Monte Carlo Tree Search."""
    def __init__(self, network: ChessNet, config: Config):
        self.network = network
        self.config = config
    
    def search(self, board: chess.Board, num_simulations: int = None) -> Tuple[int, Dict[int, float]]:
        """
        Run MCTS and return best action and visit distribution.
        """
        if num_simulations is None:
            num_simulations = self.config.mcts_simulations
        
        root = MCTSNode(prior=0)
        
        # Expand root
        state = encode_board(board)
        mask = get_legal_mask(board)
        policy, _ = self.network.predict(state, mask, temperature=1.0)
        
        legal_actions = np.where(mask)[0]
        for action in legal_actions:
            root.children[action] = MCTSNode(prior=policy[action])
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]
            
            # Selection
            while node.children and not sim_board.is_game_over():
                action, node = self._select_child(node)
                sim_board.push(decode_move(action, sim_board))
                search_path.append(node)
            
            # Expansion & Evaluation
            if sim_board.is_game_over():
                result = sim_board.result()
                if result == '1-0':
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
            else:
                # Expand
                state = encode_board(sim_board)
                mask = get_legal_mask(sim_board)
                policy, value = self.network.predict(state, mask, temperature=1.0)
                
                legal_actions = np.where(mask)[0]
                for action in legal_actions:
                    if action not in node.children:
                        node.children[action] = MCTSNode(prior=policy[action])
                
                # Flip value for opponent
                value = -value
            
            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value
        
        # Get visit counts
        visit_counts = {a: n.visit_count for a, n in root.children.items()}
        
        # Select best action
        best_action = max(visit_counts, key=visit_counts.get)
        
        return best_action, visit_counts
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

print("✅ MCTS initialized")

# ==============================================================================
# Cell 8: Experience Buffer
# ==============================================================================

class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)
        
        # Remove oldest if over capacity
        while len(self.states) > self.max_size:
            self.states.pop(0)
            self.policies.pop(0)
            self.values.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.policies[i] for i in indices]),
            np.array([self.values[i] for i in indices])
        )
    
    def __len__(self):
        return len(self.states)

buffer = ReplayBuffer(config.buffer_size)
print(f"✅ Replay buffer: max {config.buffer_size:,} positions")

# ==============================================================================
# Cell 9: Self-Play Game Generation
# ==============================================================================

def play_selfplay_game(network: ChessNet, mcts: MCTS, config: Config) -> List[Tuple]:
    """Play a self-play game and return training data."""
    board = chess.Board()
    game_data = []
    move_count = 0
    
    while not board.is_game_over() and move_count < 200:
        # Temperature schedule
        if move_count < config.temp_threshold:
            temperature = config.temp_initial
        else:
            temperature = config.temp_final
        
        # Check opening book first
        if config.use_opening_book and move_count < config.opening_book_moves:
            book_move = get_opening_book_move(board)
            if book_move:
                move = chess.Move.from_uci(book_move)
                board.push(move)
                move_count += 1
                continue
        
        # MCTS search
        state = encode_board(board)
        mask = get_legal_mask(board)
        
        action, visit_counts = mcts.search(board)
        
        # Create policy from visit counts
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        total_visits = sum(visit_counts.values())
        for a, v in visit_counts.items():
            policy[a] = v / total_visits
        
        # Store experience (value will be filled later)
        game_data.append((state, policy, board.turn))
        
        # Select move with temperature
        if temperature > 0.01:
            probs = policy ** (1 / temperature)
            probs = probs / probs.sum()
            action = np.random.choice(len(probs), p=probs)
        
        move = decode_move(action, board)
        if move is None:
            move = random.choice(list(board.legal_moves))
        
        board.push(move)
        move_count += 1
    
    # Determine game result
    result = board.result()
    if result == '1-0':
        winner = chess.WHITE
    elif result == '0-1':
        winner = chess.BLACK
    else:
        winner = None
    
    # Fill in values
    training_data = []
    for state, policy, turn in game_data:
        if winner is None:
            value = 0.0
        elif winner == turn:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state, policy, value))
    
    return training_data, result

# ==============================================================================
# Cell 10: Load Datasets
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading datasets...")

# Load Lichess games
print("   Loading Lichess games...")
try:
    lichess_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "datasnaek/chess",
        "games.csv"
    )
    print(f"   ✅ Lichess: {len(lichess_df):,} games")
except Exception as e:
    print(f"   ⚠️ Could not load Lichess: {e}")
    lichess_df = None

# Load Stockfish evaluations
print("   Loading Stockfish evaluations...")
try:
    evals_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ronakbadhe/chess-evaluations",
        "chessData.csv"
    )
    # Sample for memory
    if len(evals_df) > 1000000:
        evals_df = evals_df.sample(1000000, random_state=42)
    print(f"   ✅ Evaluations: {len(evals_df):,} positions")
except Exception as e:
    print(f"   ⚠️ Could not load evaluations: {e}")
    evals_df = None

# ==============================================================================
# Cell 11: Supervised Pre-training
# ==============================================================================

def parse_evaluation(eval_str) -> Optional[float]:
    """Parse evaluation string to [-1, 1]."""
    if pd.isna(eval_str):
        return None
    eval_str = str(eval_str).strip()
    
    if eval_str.startswith('#+'):
        return 1.0
    elif eval_str.startswith('#-') or eval_str.startswith('#'):
        return -1.0 if '-' in eval_str else 1.0
    
    try:
        cp = float(eval_str)
        return np.tanh(cp / 400)
    except:
        return None

def process_lichess_games(df, max_games=None):
    """Process Lichess games into training data."""
    states, actions, values = [], [], []
    
    games = df if max_games is None else df.sample(min(max_games, len(df)))
    
    for _, row in tqdm(games.iterrows(), total=len(games), desc="Processing Lichess"):
        try:
            board = chess.Board()
            moves_str = row['moves']
            winner = row['winner']
            
            if not isinstance(moves_str, str):
                continue
            
            for token in moves_str.split():
                if board.is_game_over():
                    break
                try:
                    move = board.parse_san(token)
                    
                    # Record position
                    state = encode_board(board)
                    action = encode_move(move)
                    
                    # Value based on winner
                    if winner == 'white':
                        v = 1.0 if board.turn == chess.WHITE else -1.0
                    elif winner == 'black':
                        v = -1.0 if board.turn == chess.WHITE else 1.0
                    else:
                        v = 0.0
                    
                    states.append(state)
                    actions.append(action)
                    values.append(v)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    return np.array(states), np.array(actions), np.array(values)

def process_stockfish_evals(df, max_positions=None):
    """Process Stockfish evaluations."""
    states, values = [], []
    
    positions = df if max_positions is None else df.sample(min(max_positions, len(df)))
    
    for _, row in tqdm(positions.iterrows(), total=len(positions), desc="Processing Evals"):
        try:
            board = chess.Board(row['FEN'])
            value = parse_evaluation(row['Evaluation'])
            
            if value is None:
                continue
            
            # Adjust for side to move
            if board.turn == chess.BLACK:
                value = -value
            
            states.append(encode_board(board))
            values.append(value)
        except:
            continue
    
    return np.array(states), np.array(values)

def supervised_train(network, lichess_df, evals_df, epochs=20):
    """Supervised pre-training."""
    print("\n📚 PHASE 1: Supervised Pre-training")
    print("=" * 50)
    
    # Process data
    if lichess_df is not None:
        l_states, l_actions, l_values = process_lichess_games(lichess_df)
        print(f"   Lichess data: {len(l_states):,} positions")
    else:
        l_states, l_actions, l_values = np.array([]), np.array([]), np.array([])
    
    if evals_df is not None:
        e_states, e_values = process_stockfish_evals(evals_df)
        print(f"   Eval data: {len(e_states):,} positions")
    else:
        e_states, e_values = np.array([]), np.array([])
    
    if len(l_states) == 0 and len(e_states) == 0:
        print("   ⚠️ No data available!")
        return
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    for epoch in range(epochs):
        network.train()
        total_ploss, total_vloss = 0, 0
        batches = 0
        
        # Train on Lichess data (policy + value)
        if len(l_states) > 0:
            indices = np.random.permutation(len(l_states))
            for i in range(0, len(l_states), config.batch_size):
                batch_idx = indices[i:i+config.batch_size]
                
                states = torch.FloatTensor(l_states[batch_idx]).to(device)
                actions = torch.LongTensor(l_actions[batch_idx]).to(device)
                values = torch.FloatTensor(l_values[batch_idx]).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits, pred_values = network(states)
                    ploss = F.cross_entropy(logits, actions)
                    vloss = F.mse_loss(pred_values.squeeze(-1), values)
                    loss = ploss + vloss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_ploss += ploss.item()
                total_vloss += vloss.item()
                batches += 1
        
        # Train on eval data (value only)
        if len(e_states) > 0:
            e_indices = np.random.permutation(len(e_states))
            for i in range(0, min(50000, len(e_states)), config.batch_size):
                batch_idx = e_indices[i:i+config.batch_size]
                
                states = torch.FloatTensor(e_states[batch_idx]).to(device)
                values = torch.FloatTensor(e_values[batch_idx]).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    _, pred_values = network(states)
                    vloss = F.mse_loss(pred_values.squeeze(-1), values)
                
                optimizer.zero_grad()
                scaler.scale(vloss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_vloss += vloss.item()
                batches += 1
        
        if batches > 0:
            avg_ploss = total_ploss / max(batches, 1)
            avg_vloss = total_vloss / batches
            
            if (epoch + 1) % 5 == 0:
                wr = evaluate_vs_random(network, 50)
                print(f"   Epoch {epoch+1}/{epochs}: PLoss={avg_ploss:.4f}, VLoss={avg_vloss:.4f}, WR={wr:.0%}")
    
    print("   ✅ Supervised training complete")

# ==============================================================================
# Cell 12: Self-Play Training
# ==============================================================================

def selfplay_train(network, config):
    """MCTS self-play training."""
    print("\n🤖 PHASE 2: MCTS Self-Play")
    print("=" * 50)
    
    mcts = MCTS(network, config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr * 0.1, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    for iteration in range(config.selfplay_iterations):
        # Generate games
        games_data = []
        wins, draws, losses = 0, 0, 0
        
        for _ in tqdm(range(config.games_per_iteration), desc=f"Iter {iteration+1} Games"):
            data, result = play_selfplay_game(network, mcts, config)
            for s, p, v in data:
                buffer.add(s, p, v)
            
            if result == '1-0':
                wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        
        # Train on buffer
        if len(buffer) > config.batch_size:
            network.train()
            for _ in range(100):
                states, policies, values = buffer.sample(config.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                policies = torch.FloatTensor(policies).to(device)
                values = torch.FloatTensor(values).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits, pred_values = network(states)
                    ploss = F.cross_entropy(logits, policies.argmax(dim=-1))
                    vloss = F.mse_loss(pred_values.squeeze(-1), values)
                    loss = ploss + vloss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Evaluate
        wr = evaluate_vs_random(network, 50)
        print(f"   Iter {iteration+1}/{config.selfplay_iterations}: W={wins} D={draws} L={losses}, WR={wr:.0%}, Buffer={len(buffer):,}")
        
        # Save checkpoint
        if (iteration + 1) % 10 == 0:
            torch.save(network.state_dict(), f'/kaggle/working/chess_v9_iter{iteration+1}.pt')
    
    print("   ✅ Self-play training complete")

# ==============================================================================
# Cell 13: Evaluation
# ==============================================================================

def evaluate_vs_random(network, n_games):
    """Evaluate against random opponent."""
    network.eval()
    wins = 0
    mcts = MCTS(network, config)
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 150:
            if board.turn == chess.WHITE:
                # Use opening book
                if move_count < config.opening_book_moves:
                    book_move = get_opening_book_move(board)
                    if book_move:
                        board.push(chess.Move.from_uci(book_move))
                        move_count += 1
                        continue
                
                # Use MCTS (fewer sims for speed)
                action, _ = mcts.search(board, num_simulations=10)
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
# Cell 14: Main Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING COMPREHENSIVE TRAINING")
print("=" * 60)

start_time = time.time()

# Phase 1: Supervised
supervised_train(network, lichess_df, evals_df, epochs=config.supervised_epochs)

# Checkpoint
torch.save(network.state_dict(), '/kaggle/working/chess_v9_supervised.pt')

# Phase 2: Self-play
selfplay_train(network, config)

# Final save
torch.save(network.state_dict(), '/kaggle/working/chess_v9_final.pt')

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/3600:.1f} hours")

# Final evaluation
print("\n📊 Final Evaluation...")
final_wr = evaluate_vs_random(network, 100)
print(f"   Win Rate vs Random: {final_wr:.0%}")

print(f"\n🎉 Training complete!")
print(f"   Model saved to /kaggle/working/chess_v9_final.pt")
