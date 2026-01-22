"""
================================================================================
⚡ CHESS AI v14 - IMPROVED RL TRAINING
================================================================================
Perbaikan dari v13:

1. CONSISTENT VALUE TARGET:
   - Supervised & RL keduanya menggunakan stockfish_eval
   - Value head belajar satu fungsi yang sama

2. WEAK STOCKFISH OPPONENT (bukan random):
   - RL melawan Stockfish level 3-8
   - Model belajar melawan lawan yang bermain cerdas
   - Model dipaksa belajar capture dan checkmate strategy

3. PROPER ADVANTAGE CALCULATION:
   - GAE (Generalized Advantage Estimation)
   - Tidak normalize per-batch yang menghilangkan sinyal

4. SMALL CAPTURE/CHECK REWARDS:
   - Capture piece: +0.01 to +0.09 (kecil, karena eval sudah include)
   - Give check: +0.005
   - Checkmate: +1.0 (besar, ini yang paling penting)
   - Eval change tetap dominan

5. CURRICULUM LEARNING:
   - RL dimulai dari weak opponent (level 3)
   - Gradually increase difficulty (up to level 8)

6. DUAL DATASET:
   - datasnaek/chess (20k games)
   - shkarupylomaxim/chess-games-dataset-lichess-2017-may (200k+ games dengan eval)

Training time: ~6-8 hours on P100
================================================================================
"""

# ==============================================================================
# Cell 1: Dependencies
# ==============================================================================

try:
    get_ipython().system('pip install -q python-chess tqdm matplotlib pandas kagglehub')
except:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'python-chess', 'tqdm', 'matplotlib', 'pandas', 'kagglehub'], capture_output=True)

import subprocess
import os
import re

# Install Stockfish
if os.path.exists('/kaggle'):
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
elif os.path.exists('/content'):
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], capture_output=True)
    STOCKFISH_PATH = '/usr/games/stockfish'
else:
    STOCKFISH_PATH = 'stockfish'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import chess
import chess.engine
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
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
print("⚡ CHESS AI v14 - IMPROVED RL TRAINING")
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
    input_channels: int = 12
    filters: int = 128
    blocks: int = 6
    
    # Training
    supervised_epochs: int = 15
    rl_iterations: int = 80
    games_per_iter: int = 20
    
    batch_size: int = 256
    lr_supervised: float = 1e-3
    lr_rl: float = 1e-5
    
    # RL rewards (kecil karena eval sudah include capture/check)
    capture_reward_scale: float = 0.1  # multiply piece value by this
    check_reward: float = 0.005
    checkmate_reward: float = 1.0  # Checkmate tetap besar!
    
    # Stockfish
    stockfish_depth: int = 8
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Curriculum
    opponent_start_level: int = 3
    opponent_max_level: int = 8
    
    # Buffer
    buffer_size: int = 100000
    
    # Opening
    opening_book_depth: int = 8

config = Config()
print(f"✅ Config: {config.input_channels} channels, {config.blocks} blocks")
print(f"✅ Rewards: capture={config.capture_reward_scale}x, check={config.check_reward}, mate={config.checkmate_reward}")

# ==============================================================================
# Cell 3: Stockfish Engine
# ==============================================================================

stockfish_eval_engine = None
stockfish_opponent = None

def init_stockfish():
    global stockfish_eval_engine, stockfish_opponent
    try:
        stockfish_eval_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish_eval_engine.configure({"Threads": 1, "Hash": 64})
        
        stockfish_opponent = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish_opponent.configure({"Threads": 1, "Hash": 64, "Skill Level": config.opponent_start_level})
        
        print(f"✅ Stockfish loaded (2 engines)")
        return True
    except Exception as e:
        print(f"⚠️ Stockfish not available: {e}")
        return False

def set_opponent_level(level: int):
    global stockfish_opponent
    if stockfish_opponent:
        stockfish_opponent.configure({"Skill Level": level})

def stockfish_eval(board: chess.Board, depth: int = 8) -> float:
    """Get Stockfish evaluation from WHITE's perspective. Returns [-1, 1]."""
    if stockfish_eval_engine is None:
        return material_eval(board)
    
    try:
        result = stockfish_eval_engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white()
        
        if score.is_mate():
            mate_in = score.mate()
            return 1.0 if mate_in > 0 else -1.0
        else:
            cp = score.score()
            return float(np.tanh(cp / 400))
    except:
        return material_eval(board)

def get_opponent_move(board: chess.Board, time_limit: float = 0.05) -> chess.Move:
    """Get move from opponent Stockfish."""
    if stockfish_opponent is None:
        return random.choice(list(board.legal_moves))
    
    try:
        result = stockfish_opponent.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    except:
        return random.choice(list(board.legal_moves))

def material_eval(board: chess.Board) -> float:
    """Material evaluation from WHITE's perspective."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            v = values[piece.piece_type]
            score += v if piece.color == chess.WHITE else -v
    
    return float(np.tanh(score / 15))

# Piece values for capture reward (scaled by config.capture_reward_scale)
PIECE_VALUES = {
    chess.PAWN: 0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.32,
    chess.ROOK: 0.5,
    chess.QUEEN: 0.9,
    chess.KING: 0
}

has_stockfish = init_stockfish()

# ==============================================================================
# Cell 4: State Encoding (12 channels)
# ==============================================================================

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

NUM_ACTIONS = 4096

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
# Cell 5: Load and Merge Datasets
# ==============================================================================

import kagglehub
from kagglehub import KaggleDatasetAdapter

def clean_moves_with_eval(moves_str: str) -> str:
    """
    Remove eval and clock comments from moves string.
    Example: 'e4 {[%eval 0.18] [%clk 0:10:00]} e5' -> 'e4 e5'
    """
    # Remove {[%eval ...] [%clk ...]} patterns
    cleaned = re.sub(r'\{[^}]*\}', '', moves_str)
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

def load_datasets():
    """Load and merge both datasets."""
    print("\n📥 Loading datasets...")
    
    # Dataset 1: /kaggle/input/chess/games.csv
    print("   Loading Dataset 1: /kaggle/input/chess/games.csv")
    df1_path = '/kaggle/input/chess/games.csv'
    
    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
        print(f"   ✅ Dataset 1: {len(df1):,} games")
    else:
        # Fallback to kagglehub
        print("   ⚠️ Not found, trying kagglehub...")
        df1 = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "datasnaek/chess", "games.csv")
        print(f"   ✅ Dataset 1: {len(df1):,} games")
    
    # Standardize columns for Dataset 1
    df1_clean = pd.DataFrame({
        'moves': df1['moves'].astype(str),
        'result': df1['winner'].map({'white': '1-0', 'black': '0-1', 'draw': '1/2-1/2'}),
        'white_elo': df1['white_rating'] if 'white_rating' in df1.columns else 1500,
        'black_elo': df1['black_rating'] if 'black_rating' in df1.columns else 1500,
        'opening_eco': df1['opening_eco'] if 'opening_eco' in df1.columns else 'A00',
        'source': 'datasnaek'
    })
    
    # Dataset 2: /kaggle/input/chess-games-dataset-lichess-2017-may/games_metadata_profile_2024_01.csv
    print("   Loading Dataset 2: chess-games-dataset-lichess-2017-may...")
    df2_path = '/kaggle/input/chess-games-dataset-lichess-2017-may/games_metadata_profile_2024_01.csv'
    
    df2_clean = pd.DataFrame()
    
    if os.path.exists(df2_path):
        print(f"   ✅ Found: {df2_path}")
        df2 = pd.read_csv(df2_path)
        print(f"   ✅ Dataset 2: {len(df2):,} games")
        
        # Also check for additional CSV files in the same directory
        dataset2_dir = os.path.dirname(df2_path)
        for f in os.listdir(dataset2_dir):
            if f.endswith('.csv') and f != 'games_metadata_profile_2024_01.csv':
                try:
                    extra_df = pd.read_csv(os.path.join(dataset2_dir, f))
                    df2 = pd.concat([df2, extra_df], ignore_index=True)
                    print(f"      + Loaded {f}: {len(extra_df):,} games")
                except Exception as e:
                    print(f"      ⚠️ Error loading {f}: {e}")
        
        print(f"   ✅ Dataset 2 total: {len(df2):,} games")
        
        # Clean moves (remove eval comments like {[%eval 0.18] [%clk 0:10:00]})
        print("   Cleaning moves with eval annotations...")
        if 'Moves' in df2.columns:
            df2['Moves_clean'] = df2['Moves'].apply(clean_moves_with_eval)
            moves_col = 'Moves_clean'
        else:
            moves_col = 'moves' if 'moves' in df2.columns else df2.columns[0]
        
        # Standardize columns
        df2_clean = pd.DataFrame({
            'moves': df2[moves_col].astype(str),
            'result': df2['Result'] if 'Result' in df2.columns else '1/2-1/2',
            'white_elo': df2['WhiteElo'] if 'WhiteElo' in df2.columns else 1500,
            'black_elo': df2['BlackElo'] if 'BlackElo' in df2.columns else 1500,
            'opening_eco': df2['ECO'] if 'ECO' in df2.columns else 'A00',
            'source': 'lichess2017'
        })
    else:
        print(f"   ⚠️ Dataset 2 not found at: {df2_path}")
        print("   💡 TIP: Attach dataset via Kaggle UI: '+ Add Data' -> search 'chess-games-dataset-lichess-2017-may'")
    
    # Merge datasets
    if len(df2_clean) > 0:
        df_merged = pd.concat([df1_clean, df2_clean], ignore_index=True)
    else:
        df_merged = df1_clean
    
    # Filter out invalid games
    df_merged = df_merged[df_merged['moves'].str.len() > 10]
    df_merged = df_merged.dropna(subset=['moves'])
    
    print(f"\n✅ Total merged: {len(df_merged):,} games")
    print(f"   From datasnaek: {len(df_merged[df_merged['source'] == 'datasnaek']):,}")
    print(f"   From lichess2017: {len(df_merged[df_merged['source'] == 'lichess2017']):,}")
    
    return df_merged

# ==============================================================================
# Cell 6: Build Opening Book
# ==============================================================================

def build_opening_book(df: pd.DataFrame, max_depth: int = 8) -> Dict[str, List[str]]:
    """Build opening book from games."""
    book = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building opening book"):
        try:
            board = chess.Board()
            moves_str = str(row['moves']).split()
            
            for i, token in enumerate(moves_str[:max_depth]):
                try:
                    fen_key = ' '.join(board.fen().split()[:4])
                    move = board.parse_san(token)
                    move_uci = move.uci()
                    
                    if move_uci not in book[fen_key]:
                        book[fen_key].append(move_uci)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
    
    filtered = {k: v for k, v in book.items() if len(v) >= 2}
    print(f"   Opening book: {len(filtered)} positions")
    return filtered

def get_book_move(board: chess.Board, opening_book: Dict) -> Optional[str]:
    """Get random move from opening book."""
    fen_key = ' '.join(board.fen().split()[:4])
    if fen_key in opening_book:
        moves = opening_book[fen_key]
        legal = [m for m in moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal:
            return random.choice(legal)
    return None

# ==============================================================================
# Cell 7: Neural Network
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        x = self.tower(self.stem(x))
        policy = self.policy_head(x)
        value = self.value_head(x)
        if mask is not None:
            policy = policy.masked_fill(~mask, -1e9)
        return policy, value
    
    def predict(self, state: np.ndarray, mask: np.ndarray, temperature: float = 0.5):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            
            if temperature < 0.1:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            return action, value.item()

network = ChessNet(in_channels=config.input_channels, 
                   filters=config.filters, 
                   blocks=config.blocks).to(device)
print(f"✅ ChessNet: {sum(p.numel() for p in network.parameters()):,} params")

# ==============================================================================
# Cell 8: GAE Buffer
# ==============================================================================

class GAEBuffer:
    def __init__(self, max_size: int, gamma: float = 0.99, lam: float = 0.95):
        self.max_size = max_size
        self.gamma = gamma
        self.lam = lam
        self.buffer = []
    
    def add_trajectory(self, trajectory: List[dict]):
        n = len(trajectory)
        if n == 0:
            return
        
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = trajectory[t].get('next_value', 0)
            else:
                next_value = trajectory[t + 1]['value']
            
            delta = trajectory[t]['reward'] + self.gamma * next_value - trajectory[t]['value']
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
        
        for t in range(n):
            state = trajectory[t]['state']
            action = trajectory[t]['action']
            advantage = advantages[t]
            value_target = advantages[t] + trajectory[t]['value']
            
            self.buffer.append((state, action, advantage, value_target))
            
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, advantages, value_targets = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(advantages, dtype=np.float32), 
                np.array(value_targets, dtype=np.float32))
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

buffer = GAEBuffer(config.buffer_size, config.gamma, config.gae_lambda)

# ==============================================================================
# Cell 9: Load Data with Caching
# ==============================================================================

import pickle

# Cache directories - prioritize user's uploaded preprocessed data
PREPROCESS_INPUT_DIR = '/kaggle/input/chesslichess'  # User's uploaded cache
CACHE_DIR = '/kaggle/working/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(name: str, data):
    """Save data to cache."""
    cache_path = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 Saved cache: {cache_path}")

def load_cache(name: str):
    """Load data from cache. Checks user's uploaded data first, then working cache."""
    # First check user's uploaded preprocessed data
    preprocess_path = os.path.join(PREPROCESS_INPUT_DIR, f'{name}.pkl')
    if os.path.exists(preprocess_path):
        with open(preprocess_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   📂 Loaded from uploaded data: {preprocess_path}")
        return data
    
    # Then check working cache
    cache_path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   📂 Loaded cache: {cache_path}")
        return data
    return None

def save_numpy_cache(name: str, arrays: dict):
    """Save numpy arrays to cache."""
    cache_path = os.path.join(CACHE_DIR, f'{name}.npz')
    np.savez_compressed(cache_path, **arrays)
    print(f"   💾 Saved cache: {cache_path}")

def load_numpy_cache(name: str) -> dict:
    """Load numpy arrays from cache. Checks user's uploaded data first, then working cache."""
    # First check user's uploaded preprocessed data
    preprocess_path = os.path.join(PREPROCESS_INPUT_DIR, f'{name}.npz')
    if os.path.exists(preprocess_path):
        data = np.load(preprocess_path)
        print(f"   📂 Loaded from uploaded data: {preprocess_path}")
        return dict(data)
    
    # Then check working cache
    cache_path = os.path.join(CACHE_DIR, f'{name}.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"   📂 Loaded cache: {cache_path}")
        return dict(data)
    return None

def preprocess_games_sequential(df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process games sequentially with stockfish_eval.
    This is slower than parallel but required because Stockfish engine
    cannot be shared between processes.
    """
    states, actions, values = [], [], []
    
    total_games = len(df)
    print(f"   Processing {total_games:,} games with stockfish_eval (depth=2)...")
    print(f"   ⚠️ This will take a while on first run, but will be cached for future runs.")
    
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total_games, desc="Processing games")):
        try:
            board = chess.Board()
            
            for token in str(row['moves']).split():
                try:
                    move = board.parse_san(token)
                    
                    state = encode_board(board)
                    action = encode_move(move)
                    # Use stockfish_eval for consistency with RL!
                    value = stockfish_eval(board, depth=2)
                    
                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    board.push(move)
                except:
                    break
        except:
            continue
        
        # Progress checkpoint every 10k games
        if (idx + 1) % 10000 == 0:
            print(f"      Processed {idx+1:,}/{total_games:,} games, {len(states):,} positions")
    
    if len(states) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(states), np.array(actions), np.array(values, dtype=np.float32)

# Load datasets
df = load_datasets()

# Build/load opening book (check uploaded data first)
print("\n📖 Loading opening book...")
opening_book_cache = load_cache('opening_book')
if opening_book_cache is not None:
    opening_book = opening_book_cache
    print(f"   Opening book: {len(opening_book)} positions")
else:
    print("   No cached opening book found, building from dataset...")
    opening_book = build_opening_book(df, max_depth=config.opening_book_depth)
    save_cache('opening_book', opening_book)

# ==============================================================================
# Cell 10: Supervised Learning Phase with Caching
# ==============================================================================

def supervised_phase(network, df, opening_book, epochs: int = 15):
    """Supervised learning from game data with caching."""
    print("\n📚 PHASE 1: Supervised Learning")
    print("=" * 50)
    
    # Try to load from cache first
    cache_data = load_numpy_cache('supervised_data')
    
    if cache_data is not None:
        states = cache_data['states']
        actions = cache_data['actions']
        values = cache_data['values']
        print(f"   Loaded {len(states):,} positions from cache")
    else:
        print("   No cache found, processing games...")
        
        # Use sequential processing with stockfish_eval (cached after first run)
        states, actions, values = preprocess_games_sequential(df)
        
        if len(states) == 0:
            print("   ⚠️ No positions extracted!")
            return 0, {'ploss': [], 'vloss': [], 'wr': []}
        
        print(f"   Positions: {len(states):,}")
        print(f"   Value range: [{values.min():.2f}, {values.max():.2f}], mean={values.mean():.2f}")
        
        # Save to cache
        save_numpy_cache('supervised_data', {
            'states': states,
            'actions': actions,
            'values': values
        })
    
    # Augmentation
    print("   Augmenting data (horizontal flip)...")
    aug_states = np.flip(states, axis=3).copy()
    
    # Vectorized action augmentation (much faster!)
    from_sq = actions // 64
    to_sq = actions % 64
    from_r, from_f = from_sq // 8, from_sq % 8
    to_r, to_f = to_sq // 8, to_sq % 8
    new_from = from_r * 8 + (7 - from_f)
    new_to = to_r * 8 + (7 - to_f)
    aug_actions = new_from * 64 + new_to
    
    states = np.concatenate([states, aug_states])
    actions = np.concatenate([actions, aug_actions])
    values = np.concatenate([values, values])
    
    print(f"   After augmentation: {len(states):,} positions")
    
    # Training
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_supervised, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    best_wr, best_state = 0, None
    history = {'ploss': [], 'vloss': [], 'wr': []}
    
    n = len(states)
    
    for epoch in range(epochs):
        network.train()
        idx = np.random.permutation(n)
        
        ploss_sum, vloss_sum, batches = 0, 0, 0
        
        for i in range(0, n, config.batch_size):
            batch_idx = idx[i:i+config.batch_size]
            
            s = torch.FloatTensor(states[batch_idx]).to(device)
            a = torch.LongTensor(actions[batch_idx]).to(device)
            v = torch.FloatTensor(values[batch_idx]).to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, pred_v = network(s)
                
                ploss = F.cross_entropy(logits, a, label_smoothing=0.1)
                vloss = F.mse_loss(pred_v.squeeze(-1), v)
                loss = ploss + vloss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ploss_sum += ploss.item()
            vloss_sum += vloss.item()
            batches += 1
        
        scheduler.step()
        
        history['ploss'].append(ploss_sum / batches)
        history['vloss'].append(vloss_sum / batches)
        
        if (epoch + 1) % 3 == 0:
            wr = evaluate(network, opening_book, 50)
            history['wr'].append(wr)
            
            print(f"   Epoch {epoch+1}/{epochs}: PLoss={ploss_sum/batches:.4f}, "
                  f"VLoss={vloss_sum/batches:.4f}, WR={wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
    
    if best_state:
        network.load_state_dict(best_state)
        print(f"   ✅ Restored best model (WR={best_wr:.0%})")
    
    torch.save(network.state_dict(), '/kaggle/working/chess_v14_supervised.pt')
    
    return best_wr, history

# ==============================================================================
# Cell 11: RL Phase
# ==============================================================================

def compute_capture_reward(board: chess.Board, move: chess.Move) -> float:
    """Compute SMALL reward for capturing pieces (scaled by config)."""
    if board.is_capture(move):
        captured_sq = move.to_square
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN] * config.capture_reward_scale
        captured_piece = board.piece_at(captured_sq)
        if captured_piece:
            return PIECE_VALUES[captured_piece.piece_type] * config.capture_reward_scale
    return 0.0

def play_rl_game(network, opening_book, epsilon: float, opponent_level: int):
    """Play game: Model (White) vs Stockfish (Black)."""
    set_opponent_level(opponent_level)
    
    board = chess.Board()
    trajectory = []
    move_count = 0
    
    while not board.is_game_over() and move_count < 150:
        
        if board.turn == chess.WHITE:
            # Opening book
            if move_count < config.opening_book_depth:
                book_move = get_book_move(board, opening_book)
                if book_move:
                    board.push(chess.Move.from_uci(book_move))
                    move_count += 1
                    continue
            
            state = encode_board(board)
            mask = get_legal_mask(board)
            
            current_value = stockfish_eval(board)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                move = random.choice(list(board.legal_moves))
                action = encode_move(move)
            else:
                action, _ = network.predict(state, mask, temperature=0.3)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
                    action = encode_move(move)
            
            # Small capture reward (scaled down!)
            capture_reward = compute_capture_reward(board, move)
            
            # Small check reward
            check_reward = config.check_reward if board.gives_check(move) else 0.0
            
            board.push(move)
            
            # Big checkmate reward
            if board.is_checkmate():
                checkmate_reward = config.checkmate_reward
            else:
                checkmate_reward = 0.0
            
            new_value = stockfish_eval(board)
            
            # Total reward = eval change (dominant) + small bonuses
            eval_reward = new_value - current_value
            total_reward = eval_reward + capture_reward + check_reward + checkmate_reward
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': total_reward,
                'value': current_value,
                'next_value': new_value
            })
        
        else:
            # Stockfish opponent
            move = get_opponent_move(board)
            board.push(move)
        
        move_count += 1
    
    # Game result
    result = board.result()
    if len(trajectory) > 0:
        if result == '1-0':
            trajectory[-1]['reward'] += 0.5
            trajectory[-1]['next_value'] = 1.0
        elif result == '0-1':
            trajectory[-1]['reward'] -= 0.5
            trajectory[-1]['next_value'] = -1.0
        else:
            trajectory[-1]['next_value'] = 0.0
    
    buffer.add_trajectory(trajectory)
    
    return result

def rl_phase(network, opening_book, iterations: int = 80):
    """RL training with curriculum learning."""
    print("\n🤖 PHASE 2: Curriculum RL (vs Stockfish)")
    print("=" * 50)
    
    buffer.clear()
    print("   Buffer cleared")
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_rl, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    initial_wr = evaluate(network, opening_book, 50)
    best_wr = initial_wr
    print(f"   Starting WR: {initial_wr:.0%}")
    
    history = [initial_wr]
    no_improve = 0
    current_level = config.opponent_start_level
    
    for iteration in range(iterations):
        progress = iteration / iterations
        epsilon = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress
        
        # Curriculum
        level_progress = min(1.0, (iteration + 1) / (iterations * 0.7))
        current_level = int(config.opponent_start_level + 
                           (config.opponent_max_level - config.opponent_start_level) * level_progress)
        
        # Play games
        wins, draws, losses = 0, 0, 0
        for _ in range(config.games_per_iter):
            result = play_rl_game(network, opening_book, epsilon, current_level)
            if result == '1-0':
                wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        
        # Train
        if len(buffer) >= config.batch_size:
            network.train()
            
            for _ in range(5):
                states, actions, advantages, value_targets = buffer.sample(config.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                advantages = torch.FloatTensor(advantages).to(device)
                value_targets = torch.FloatTensor(value_targets).to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits, pred_v = network(states)
                    
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    # FIX #1: Use raw GAE advantages (already properly computed)
                    # Only light scaling for numerical stability, no per-batch normalization
                    scaled_advantages = advantages / (advantages.abs().max() + 1e-8)
                    
                    policy_loss = -(selected_log_probs * scaled_advantages.detach()).mean()
                    value_loss = F.mse_loss(pred_v.squeeze(-1), value_targets)
                    
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
        
        # Evaluate
        if (iteration + 1) % 10 == 0:
            # FIX #2: Evaluate vs both Random AND Stockfish at current level
            wr_random = evaluate(network, opening_book, 30)
            sf_w, sf_d, sf_l = evaluate_vs_stockfish(network, opening_book, 10, current_level)
            sf_score = (sf_w + sf_d * 0.5) / 10
            
            # Combined metric: 60% random, 40% stockfish
            combined_wr = 0.6 * wr_random + 0.4 * sf_score
            history.append(combined_wr)
            
            print(f"   Iter {iteration+1}/{iterations}: W={wins} D={draws} L={losses} "
                  f"(vs SF{current_level}), ε={epsilon:.2f}")
            print(f"      Eval: Random={wr_random:.0%}, SF{current_level}={sf_w}W/{sf_d}D/{sf_l}L, "
                  f"Combined={combined_wr:.0%}")
            
            if combined_wr > best_wr:
                best_wr = combined_wr
                no_improve = 0
                torch.save(network.state_dict(), '/kaggle/working/chess_v14_best.pt')
            else:
                no_improve += 1
            
            if combined_wr < initial_wr * 0.5 and iteration > 30:
                print(f"   ⚠️ WR dropped below 50% of initial, stopping")
                break
            
            # FIX #3: Increase patience for early stopping (8 evals = 80 iterations)
            if no_improve >= 8:
                print(f"   ⏹️ No improvement for 8 evals, stopping")
                break
    
    if os.path.exists('/kaggle/working/chess_v14_best.pt'):
        network.load_state_dict(torch.load('/kaggle/working/chess_v14_best.pt'))
        print(f"   ✅ Restored best model (WR={best_wr:.0%})")
    
    return best_wr, history

# ==============================================================================
# Cell 12: Evaluation
# ==============================================================================

def evaluate(network, opening_book, n_games: int = 50):
    """Evaluate model against random opponent."""
    network.eval()
    wins = 0
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 150:
            if board.turn == chess.WHITE:
                if move_count < config.opening_book_depth:
                    book_move = get_book_move(board, opening_book)
                    if book_move:
                        board.push(chess.Move.from_uci(book_move))
                        move_count += 1
                        continue
                
                state = encode_board(board)
                mask = get_legal_mask(board)
                action, _ = network.predict(state, mask, temperature=0.1)
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

def evaluate_vs_stockfish(network, opening_book, n_games: int = 20, sf_level: int = 5):
    """Evaluate model against Stockfish."""
    network.eval()
    wins, draws, losses = 0, 0, 0
    set_opponent_level(sf_level)
    
    for _ in range(n_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 150:
            if board.turn == chess.WHITE:
                if move_count < config.opening_book_depth:
                    book_move = get_book_move(board, opening_book)
                    if book_move:
                        board.push(chess.Move.from_uci(book_move))
                        move_count += 1
                        continue
                
                state = encode_board(board)
                mask = get_legal_mask(board)
                action, _ = network.predict(state, mask, temperature=0.1)
                move = decode_move(action, board)
                if move is None:
                    move = random.choice(list(board.legal_moves))
            else:
                move = get_opponent_move(board)
            
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
    
    return wins, draws, losses

# ==============================================================================
# Cell 13: Run Training
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

start_time = time.time()

# Phase 1: Supervised
supervised_wr, sup_history = supervised_phase(network, df, opening_book, 
                                               epochs=config.supervised_epochs)

# Phase 2: RL
rl_best_wr, rl_history = rl_phase(network, opening_book, 
                                   iterations=config.rl_iterations)

# Final evaluation
print("\n📊 Final Evaluation...")
final_wr_random = evaluate(network, opening_book, 100)
print(f"   vs Random (100 games): {final_wr_random:.0%}")

print("\n📊 Evaluation vs Stockfish...")
for level in [3, 5, 8]:
    w, d, l = evaluate_vs_stockfish(network, opening_book, 20, level)
    print(f"   vs Stockfish Lv{level}: W={w} D={d} L={l} (WR={(w+d*0.5)/20:.0%})")

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/3600:.1f} hours")

# Save final model
torch.save(network.state_dict(), '/kaggle/working/chess_v14_final.pt')
print("💾 Models saved!")

# Cleanup
if stockfish_eval_engine:
    stockfish_eval_engine.quit()
if stockfish_opponent:
    stockfish_opponent.quit()

# ==============================================================================
# Cell 14: Plot Results
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(sup_history['ploss'], 'b-')
axes[0].set_title('Policy Loss (Supervised)')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(sup_history['vloss'], 'g-')
axes[1].set_title('Value Loss (Supervised)')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

x_sup = list(range(3, len(sup_history['wr'])*3+1, 3))

axes[2].plot(x_sup, sup_history['wr'], 'b-o', label='Supervised', markersize=4)
axes[2].axhline(supervised_wr, color='blue', linestyle='--', alpha=0.5)

if len(rl_history) > 1:
    rl_x = [0] + list(range(10, (len(rl_history))*10, 10))
    axes[2].plot([max(x_sup)] + [max(x_sup) + x for x in rl_x[1:]], 
                 rl_history, 'r-o', label='RL (Curriculum)', markersize=4)

axes[2].axhline(0.8, color='gray', linestyle=':', alpha=0.5)
axes[2].set_title(f'Win Rate (Final: {final_wr_random:.0%})')
axes[2].set_xlabel('Training Progress')
axes[2].set_ylim(0, 1)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_v14.png', dpi=150)
plt.show()

print(f"\n🎉 TRAINING COMPLETE!")
print(f"   Supervised best: {supervised_wr:.0%}")
print(f"   RL best: {rl_best_wr:.0%}")
print(f"   Final vs Random: {final_wr_random:.0%}")
print(f"\n📌 v14 improvements:")
print(f"   ✓ Dual dataset ({len(df):,} games)")
print(f"   ✓ Consistent value targets (stockfish_eval)")
print(f"   ✓ GAE advantages")
print(f"   ✓ Stockfish opponent (curriculum)")
print(f"   ✓ Small capture/check rewards (eval dominant)")
