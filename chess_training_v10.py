"""
================================================================================
⚡ CHESS AI v10 - STABLE & OPTIMIZED
================================================================================
Fixed issues from v9:
1. Value loss now decreases properly
2. Faster self-play (reduced MCTS sims)
3. Better training stability
4. Progressive difficulty

Training time: ~6-10 hours on P100

Author: AI Assistant
Date: 2026-01-18 (v10 - Stable)
================================================================================
"""

# ==============================================================================
# Cell 1: Setup & Config
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

@dataclass
class Config:
    # Network
    filters: int = 128
    blocks: int = 6
    
    # Phase 1: Supervised
    supervised_epochs: int = 15
    
    # Phase 2: Self-play (lighter)
    selfplay_iterations: int = 30
    games_per_iteration: int = 50  # Reduced from 100
    mcts_simulations: int = 20      # Reduced from 50
    
    # Training
    batch_size: int = 256
    lr_supervised: float = 1e-3
    lr_selfplay: float = 5e-4
    
    # MCTS
    c_puct: float = 1.5
    
    # Temperature
    temp_high: float = 1.0
    temp_low: float = 0.2
    temp_drop_move: int = 20
    
    # Opening book
    opening_book_moves: int = 8
    
    # Buffer
    buffer_size: int = 100000

config = Config()

print("=" * 60)
print("⚡ CHESS AI v10 - STABLE & OPTIMIZED")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Device: {device}")
print(f"✅ MCTS sims: {config.mcts_simulations}")
print(f"✅ Games/iter: {config.games_per_iteration}")

# ==============================================================================
# Cell 2: Dependencies
# ==============================================================================

!pip install -q python-chess gymnasium tqdm matplotlib
print("✅ Dependencies installed!")

try:
    import chess
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    print("✅ Dependencies loaded!")
except ImportError:
    print("Run: pip install python-chess tqdm matplotlib pandas")

# ==============================================================================
# Cell 3: Opening Book (Expanded)
# ==============================================================================

OPENING_BOOK = {
    # Starting position
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    # After 1.e4
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b': ['e7e5', 'c7c5', 'e7e6', 'c7c6', 'd7d5'],
    # After 1.d4
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b': ['d7d5', 'g8f6', 'e7e6', 'f7f5'],
    # After 1.e4 e5
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'f1c4', 'b1c3', 'f2f4'],
    # After 1.e4 e5 2.Nf3
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b': ['b8c6', 'g8f6', 'd7d6'],
    # After 1.e4 e5 2.Nf3 Nc6
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w': ['f1b5', 'f1c4', 'd2d4', 'b1c3'],
    # Italian Game
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b': ['g8f6', 'f8c5', 'f8e7'],
    # Ruy Lopez
    'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b': ['a7a6', 'g8f6', 'f8c5'],
    # After 1.d4 d5
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w': ['c2c4', 'g1f3', 'c1f4', 'b1c3'],
    # Queen's Gambit
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b': ['e7e6', 'c7c6', 'd5c4', 'e7e5'],
    # After 1.d4 Nf6
    'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w': ['c2c4', 'g1f3', 'c1f4', 'b1c3'],
    # Sicilian
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w': ['g1f3', 'b1c3', 'c2c3', 'd2d4'],
    # French Defense
    'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w': ['d2d4', 'b1c3', 'd2d3', 'g1f3'],
    # Caro-Kann
    'rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w': ['d2d4', 'b1c3', 'g1f3', 'c2c4'],
}

def get_opening_move(board: chess.Board) -> Optional[str]:
    fen_key = ' '.join(board.fen().split()[:2])
    if fen_key in OPENING_BOOK:
        moves = [m for m in OPENING_BOOK[fen_key] if chess.Move.from_uci(m) in board.legal_moves]
        if moves:
            return random.choice(moves)
    return None

print(f"✅ Opening book: {len(OPENING_BOOK)} positions")

# ==============================================================================
# Cell 4: State Encoder & Actions
# ==============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    """8-channel encoding."""
    state = np.zeros((8, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_map[piece.piece_type]
            r, f = sq // 8, sq % 8
            state[ch, r, f] = 1.0 if piece.color == chess.WHITE else -1.0
    
    state[6, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    state[7, :, :] = min(board.fullmove_number / 100, 1.0)
    return state

NUM_ACTIONS = 4096

def encode_move(m: chess.Move) -> int:
    return m.from_square * 64 + m.to_square

def decode_move(a: int, board: chess.Board) -> Optional[chess.Move]:
    f, t = a // 64, a % 64
    for m in board.legal_moves:
        if m.from_square == f and m.to_square == t:
            return m
    return None

def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[encode_move(m)] = True
    return mask

# ==============================================================================
# Cell 5: Neural Network
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), 
                                nn.Linear(c, c//8), nn.ReLU(), nn.Linear(c//8, c), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(-1, x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), SEBlock(c)
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class ChessNet(nn.Module):
    def __init__(self, filters=128, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(8, filters, 3, padding=1, bias=False), nn.BatchNorm2d(filters), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(2048, NUM_ACTIONS))
        self.value = nn.Sequential(nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        x = self.blocks(self.stem(x))
        p = self.policy(x)
        v = self.value(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)
        return p, v
    
    def predict(self, state, mask, temp=0.5):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            logits, value = self(x, m)
            
            if temp < 0.1:
                probs = torch.zeros_like(logits)
                probs[0, logits.argmax()] = 1.0
            else:
                probs = F.softmax(logits / temp, dim=-1)
            
            return probs.squeeze(0).cpu().numpy(), value.item()

network = ChessNet(config.filters, config.blocks).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 6: Simple MCTS
# ==============================================================================

class Node:
    __slots__ = ['n', 'w', 'p', 'children']
    def __init__(self, prior):
        self.n, self.w, self.p, self.children = 0, 0, prior, {}

def mcts_search(board: chess.Board, network: ChessNet, num_sims: int, c_puct: float) -> Tuple[int, np.ndarray]:
    """Lightweight MCTS search."""
    root = Node(0)
    
    # Expand root
    state = encode_board(board)
    mask = get_legal_mask(board)
    policy, _ = network.predict(state, mask, temp=1.0)
    
    legal = np.where(mask)[0]
    for a in legal:
        root.children[a] = Node(policy[a])
    
    # Simulations
    for _ in range(num_sims):
        node = root
        sim_board = board.copy()
        path = [node]
        
        # Select
        while node.children and not sim_board.is_game_over():
            best_a, best_score = None, -float('inf')
            sqrt_n = math.sqrt(max(node.n, 1))
            for a, child in node.children.items():
                q = child.w / (child.n + 1e-8)
                u = c_puct * child.p * sqrt_n / (1 + child.n)
                score = q + u
                if score > best_score:
                    best_score, best_a = score, a
            
            move = decode_move(best_a, sim_board)
            if move:
                sim_board.push(move)
            node = node.children[best_a]
            path.append(node)
        
        # Evaluate
        if sim_board.is_game_over():
            result = sim_board.result()
            if result == '1-0':
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == '0-1':
                v = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                v = 0.0
        else:
            _, v = network.predict(encode_board(sim_board), get_legal_mask(sim_board), temp=1.0)
            v = -v  # From opponent's view
        
        # Backprop
        for node in reversed(path):
            node.n += 1
            node.w += v
            v = -v
    
    # Get policy from visits
    policy = np.zeros(NUM_ACTIONS)
    total = sum(c.n for c in root.children.values())
    for a, c in root.children.items():
        policy[a] = c.n / max(total, 1)
    
    best_a = max(root.children.keys(), key=lambda a: root.children[a].n)
    return best_a, policy

# ==============================================================================
# Cell 7: Data Buffer
# ==============================================================================

class Buffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = []
    
    def add(self, state, policy, value):
        self.data.append((state, policy, value))
        if len(self.data) > self.maxlen:
            self.data.pop(0)
    
    def sample(self, n):
        batch = random.sample(self.data, min(n, len(self.data)))
        s, p, v = zip(*batch)
        return np.array(s), np.array(p), np.array(v)
    
    def __len__(self):
        return len(self.data)

buffer = Buffer(config.buffer_size)

# ==============================================================================
# Cell 8: Load Datasets
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("\n📥 Loading datasets...")

try:
    lichess_df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "datasnaek/chess", "games.csv")
    print(f"   ✅ Lichess: {len(lichess_df):,} games")
except:
    lichess_df = None
    print("   ⚠️ Lichess not available")

try:
    evals_df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "ronakbadhe/chess-evaluations", "chessData.csv")
    if len(evals_df) > 500000:
        evals_df = evals_df.sample(500000, random_state=42)
    print(f"   ✅ Evaluations: {len(evals_df):,} positions")
except:
    evals_df = None
    print("   ⚠️ Evaluations not available")

# ==============================================================================
# Cell 9: Phase 1 - Supervised Training (FIXED)
# ==============================================================================

def parse_eval(s) -> Optional[float]:
    """Parse Stockfish eval to [-1, 1] - FIXED VERSION."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    
    # Checkmate
    if '#' in s:
        return 1.0 if '+' in s or s.replace('#', '').lstrip('-').isdigit() and int(s.replace('#', '')) > 0 else -1.0
    
    try:
        cp = float(s)
        # Proper normalization: tanh with good scaling
        return float(np.tanh(cp / 500))  # 500cp = ~0.76, 1000cp = ~0.96
    except:
        return None

def supervised_training(network, lichess_df, evals_df, epochs):
    """Supervised pre-training - FIXED."""
    print("\n📚 PHASE 1: Supervised Training")
    print("=" * 50)
    
    # Collect Lichess data
    l_states, l_actions, l_values = [], [], []
    if lichess_df is not None:
        for _, row in tqdm(lichess_df.iterrows(), total=len(lichess_df), desc="Lichess"):
            try:
                board = chess.Board()
                winner = row['winner']
                for token in str(row['moves']).split():
                    try:
                        move = board.parse_san(token)
                        l_states.append(encode_board(board))
                        l_actions.append(encode_move(move))
                        
                        if winner == 'white':
                            v = 1.0 if board.turn == chess.WHITE else -1.0
                        elif winner == 'black':
                            v = -1.0 if board.turn == chess.WHITE else 1.0
                        else:
                            v = 0.0
                        l_values.append(v)
                        
                        board.push(move)
                    except:
                        break
            except:
                continue
    
    l_states = np.array(l_states) if l_states else np.zeros((0, 8, 8, 8))
    l_actions = np.array(l_actions) if l_actions else np.zeros(0, dtype=np.int64)
    l_values = np.array(l_values) if l_values else np.zeros(0)
    print(f"   Lichess: {len(l_states):,} positions")
    
    # Collect eval data
    e_states, e_values = [], []
    if evals_df is not None:
        for _, row in tqdm(evals_df.iterrows(), total=len(evals_df), desc="Evals"):
            try:
                board = chess.Board(row['FEN'])
                v = parse_eval(row['Evaluation'])
                if v is not None:
                    if board.turn == chess.BLACK:
                        v = -v
                    e_states.append(encode_board(board))
                    e_values.append(v)
            except:
                continue
    
    e_states = np.array(e_states) if e_states else np.zeros((0, 8, 8, 8))
    e_values = np.array(e_values) if e_values else np.zeros(0)
    print(f"   Evals: {len(e_states):,} positions")
    
    # Training
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_supervised, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    best_wr = 0
    best_state = None
    
    for epoch in range(epochs):
        network.train()
        ploss_sum, vloss_sum, batches = 0, 0, 0
        
        # Policy + Value from Lichess
        if len(l_states) > 0:
            idx = np.random.permutation(len(l_states))
            for i in range(0, len(l_states), config.batch_size):
                batch = idx[i:i+config.batch_size]
                
                states = torch.FloatTensor(l_states[batch]).to(device)
                actions = torch.LongTensor(l_actions[batch]).to(device)
                values = torch.FloatTensor(l_values[batch]).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits, pred_v = network(states)
                    ploss = F.cross_entropy(logits, actions, label_smoothing=0.1)
                    vloss = F.mse_loss(pred_v.squeeze(-1), values)
                    loss = ploss + 0.5 * vloss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                ploss_sum += ploss.item()
                vloss_sum += vloss.item()
                batches += 1
        
        # Value only from evals (smaller batch per epoch)
        if len(e_states) > 0:
            e_idx = np.random.permutation(len(e_states))[:50000]
            for i in range(0, len(e_idx), config.batch_size):
                batch = e_idx[i:i+config.batch_size]
                
                states = torch.FloatTensor(e_states[batch]).to(device)
                values = torch.FloatTensor(e_values[batch]).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    _, pred_v = network(states)
                    vloss = F.mse_loss(pred_v.squeeze(-1), values)
                
                optimizer.zero_grad()
                scaler.scale(vloss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                vloss_sum += vloss.item()
                batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 3 == 0:
            wr = evaluate(network, 50)
            avg_p = ploss_sum / max(batches, 1)
            avg_v = vloss_sum / max(batches, 1)
            print(f"   Epoch {epoch+1}/{epochs}: PLoss={avg_p:.4f}, VLoss={avg_v:.4f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
    
    if best_state:
        network.load_state_dict(best_state)
        print(f"   ✅ Restored best (WR={best_wr:.0%})")
    
    torch.save(network.state_dict(), '/kaggle/working/chess_v10_supervised.pt')
    return best_wr

# ==============================================================================
# Cell 10: Phase 2 - Self-Play
# ==============================================================================

def play_game(network):
    """Play one self-play game."""
    board = chess.Board()
    history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < 150:
        # Temperature
        temp = config.temp_high if move_count < config.temp_drop_move else config.temp_low
        
        # Opening book
        if move_count < config.opening_book_moves:
            book_move = get_opening_move(board)
            if book_move:
                board.push(chess.Move.from_uci(book_move))
                move_count += 1
                continue
        
        # MCTS
        state = encode_board(board)
        action, policy = mcts_search(board, network, config.mcts_simulations, config.c_puct)
        
        history.append((state, policy, board.turn))
        
        # Sample with temperature
        if temp > 0.1:
            probs = policy ** (1/temp)
            probs = probs / (probs.sum() + 1e-8)
            valid = np.where(probs > 0)[0]
            if len(valid) > 0:
                action = np.random.choice(valid, p=probs[valid]/probs[valid].sum())
        
        move = decode_move(action, board)
        if move is None:
            move = random.choice(list(board.legal_moves))
        
        board.push(move)
        move_count += 1
    
    # Result
    result = board.result()
    winner = chess.WHITE if result == '1-0' else (chess.BLACK if result == '0-1' else None)
    
    # Assign values
    data = []
    for state, policy, turn in history:
        if winner is None:
            v = 0.0
        elif winner == turn:
            v = 1.0
        else:
            v = -1.0
        data.append((state, policy, v))
    
    return data, result

def selfplay_training(network):
    """Self-play phase."""
    print("\n🤖 PHASE 2: Self-Play")
    print("=" * 50)
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_selfplay, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    best_wr = 0
    
    for iteration in range(config.selfplay_iterations):
        # Generate games
        wins, draws, losses = 0, 0, 0
        
        for _ in tqdm(range(config.games_per_iteration), desc=f"Iter {iteration+1}"):
            data, result = play_game(network)
            for s, p, v in data:
                buffer.add(s, p, v)
            
            if result == '1-0':
                wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        
        # Train
        if len(buffer) > config.batch_size:
            network.train()
            for _ in range(50):
                states, policies, values = buffer.sample(config.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                policies = torch.FloatTensor(policies).to(device)
                values = torch.FloatTensor(values).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits, pred_v = network(states)
                    
                    # Cross entropy with policy targets
                    log_probs = F.log_softmax(logits, dim=-1)
                    ploss = -(policies * log_probs).sum(dim=-1).mean()
                    vloss = F.mse_loss(pred_v.squeeze(-1), values)
                    loss = ploss + vloss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Evaluate
        wr = evaluate(network, 50)
        print(f"   Iter {iteration+1}: W={wins} D={draws} L={losses}, WR={wr:.0%}, Buf={len(buffer):,}")
        
        if wr > best_wr:
            best_wr = wr
            torch.save(network.state_dict(), '/kaggle/working/chess_v10_best.pt')
        
        # Save checkpoint
        if (iteration + 1) % 10 == 0:
            torch.save(network.state_dict(), f'/kaggle/working/chess_v10_iter{iteration+1}.pt')
    
    return best_wr

# ==============================================================================
# Cell 11: Evaluation
# ==============================================================================

def evaluate(network, n_games):
    """Evaluate vs random."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(150):
            if board.is_game_over():
                break
            
            if board.turn == chess.WHITE:
                # Opening book
                if board.fullmove_number <= 5:
                    book = get_opening_move(board)
                    if book:
                        board.push(chess.Move.from_uci(book))
                        continue
                
                # Network (no MCTS for speed)
                state = encode_board(board)
                mask = get_legal_mask(board)
                probs, _ = network.predict(state, mask, temp=0.1)
                move = decode_move(int(np.argmax(probs)), board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        if board.result() == '1-0':
            wins += 1
    
    return wins / n_games

# ==============================================================================
# Cell 12: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

start = time.time()

# Phase 1
super_wr = supervised_training(network, lichess_df, evals_df, config.supervised_epochs)

# Phase 2
final_wr = selfplay_training(network)

# Save final
torch.save(network.state_dict(), '/kaggle/working/chess_v10_final.pt')

print(f"\n⏱️ Total time: {(time.time()-start)/3600:.1f} hours")
print(f"📊 Supervised best: {super_wr:.0%}")
print(f"📊 Final: {final_wr:.0%}")
print(f"\n🎉 Training complete!")
