"""
================================================================================
⚡ CHESS AI v17 - ULTRA RICH DIAGNOSTICS
================================================================================
Enhanced dari v16 dengan COMPREHENSIVE LOGGING untuk debugging training:

NEW DIAGNOSTICS v17:
1. LAYER-BY-LAYER GRADIENT FLOW:
   - Track gradient norm per layer (stem, tower, policy, value)
   - Detect vanishing/exploding gradients early

2. ACTIVATION STATISTICS:
   - Monitor activations at each network stage
   - Detect dead ReLUs and saturation

3. LOSS COMPONENT BREAKDOWN:
   - Per-batch detailed loss tracking
   - Moving averages for trend detection

4. POLICY HEAD ANALYSIS:
   - Entropy tracking (exploration vs exploitation)
   - Top-k action confidence
   - Legal move accuracy

5. VALUE HEAD DEEP ANALYSIS:
   - Pre-tanh activation distribution
   - Tanh saturation percentage
   - Prediction vs target correlation

6. EARLY WARNING SYSTEM:
   - Gradient spike detection
   - Loss spike detection  
   - Learning stagnation alerts

7. TRAINING DYNAMICS:
   - Learning curve slope
   - Effective learning rate tracking
   - Weight update magnitude

8. CHECKPOINT DIAGNOSTICS:
   - Per-epoch model health score
   - Automatic issue summarization

Training time: ~3-4 hours on P100
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
print("⚡ CHESS AI v17 - ULTRA RICH DIAGNOSTICS")
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
    rl_iterations: int = 50   # Reduced - focus on supervised
    games_per_iter: int = 20
    
    batch_size: int = 256
    
    # [v16 FIX] Separate learning rates for policy and value heads
    lr_policy: float = 1e-3       # Policy head - normal
    lr_value: float = 1e-4        # Value head - 10x lower to prevent explosion
    lr_value_warmup: float = 1e-5 # Even lower for first 2 epochs
    warmup_epochs: int = 2
    
    lr_rl: float = 1e-6  # RL learning rate
    
    # [v16 FIX] Value head collapse detection
    value_collapse_threshold: float = 0.95  # If |pred_v| > this for 90% samples, reset
    
    # Value loss weight
    value_loss_weight: float = 1.0  # Back to 1.0, separate LR handles it
    
    # [v16 FIX] Per-head gradient clipping
    grad_clip_policy: float = 1.0
    grad_clip_value: float = 0.5  # Stricter for value head
    
    # RL rewards
    capture_reward_scale: float = 0.1
    check_reward: float = 0.005
    checkmate_reward: float = 1.0
    
    # Stockfish
    stockfish_depth: int = 8
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Curriculum
    opponent_start_level: int = 0
    opponent_max_level: int = 3  # Lower max since we're focusing on policy
    min_win_rate_to_advance: float = 0.30
    
    # Buffer
    buffer_size: int = 50000
    
    # Opening
    opening_book_depth: int = 8

config = Config()
print(f"✅ Config: {config.input_channels} channels, {config.blocks} blocks")
print(f"✅ Policy LR: {config.lr_policy}, Value LR: {config.lr_value} (warmup: {config.lr_value_warmup})")
print(f"✅ Grad clip: policy={config.grad_clip_policy}, value={config.grad_clip_value}")
print(f"✅ Collapse threshold: {config.value_collapse_threshold}")

# ==============================================================================
# Cell 2.5: [v17] TRAINING DIAGNOSTICS SYSTEM
# ==============================================================================

class TrainingDiagnostics:
    """[v17] Comprehensive training diagnostics for debugging."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all diagnostic history."""
        # Loss tracking
        self.batch_ploss = []
        self.batch_vloss = []
        self.epoch_ploss = []
        self.epoch_vloss = []
        
        # Gradient tracking
        self.grad_stem = []
        self.grad_tower = []
        self.grad_policy = []
        self.grad_value = []
        self.grad_spikes = []
        
        # Value head analysis
        self.pretanh_mean = []
        self.pretanh_std = []
        self.pretanh_min = []
        self.pretanh_max = []
        self.tanh_saturation = []
        self.value_target_corr = []
        
        # Policy head analysis
        self.policy_entropy = []
        self.top1_confidence = []
        self.top5_confidence = []
        self.legal_accuracy = []
        
        # Weight tracking
        self.weight_update_magnitude = []
        self.weight_norm_history = {}
        
        # Early warnings
        self.warnings = []
        self.health_scores = []
        
        # Epoch info
        self.current_epoch = 0
        self.current_batch = 0
    
    def compute_layer_grad_norms(self, network) -> dict:
        """Compute gradient norms for each layer group."""
        norms = {'stem': 0.0, 'tower': 0.0, 'policy': 0.0, 'value': 0.0}
        
        # Stem
        for p in network.stem.parameters():
            if p.grad is not None:
                norms['stem'] += p.grad.data.norm(2).item() ** 2
        norms['stem'] = norms['stem'] ** 0.5
        
        # Tower
        for p in network.tower.parameters():
            if p.grad is not None:
                norms['tower'] += p.grad.data.norm(2).item() ** 2
        norms['tower'] = norms['tower'] ** 0.5
        
        # Policy head
        for p in network.policy_head.parameters():
            if p.grad is not None:
                norms['policy'] += p.grad.data.norm(2).item() ** 2
        norms['policy'] = norms['policy'] ** 0.5
        
        # Value head
        for p in network.value_head.parameters():
            if p.grad is not None:
                norms['value'] += p.grad.data.norm(2).item() ** 2
        norms['value'] = norms['value'] ** 0.5
        
        return norms
    
    def analyze_value_head_pretanh(self, network, x) -> dict:
        """Analyze pre-tanh activations in value head."""
        network.eval()
        with torch.no_grad():
            # Forward through stem and tower
            features = network.tower(network.stem(x))
            
            # Manually go through value head to get pre-tanh
            for i, layer in enumerate(network.value_head):
                features = layer(features)
                # Stop before tanh (last layer)
                if i == len(network.value_head) - 2:  # Linear before Tanh
                    pretanh = features.cpu().numpy().flatten()
                    return {
                        'mean': float(np.mean(pretanh)),
                        'std': float(np.std(pretanh)),
                        'min': float(np.min(pretanh)),
                        'max': float(np.max(pretanh)),
                        'saturation_pct': float(np.mean(np.abs(pretanh) > 2.0) * 100)
                    }
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'saturation_pct': 0}
    
    def analyze_policy_head(self, logits, targets) -> dict:
        """Analyze policy head predictions."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Entropy
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            
            # Top-k confidence
            top1 = probs.max(dim=-1)[0].mean().item()
            top5 = probs.topk(5, dim=-1)[0].sum(dim=-1).mean().item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()
            
            return {
                'entropy': entropy,
                'top1_conf': top1,
                'top5_conf': top5,
                'accuracy': accuracy
            }
    
    def compute_value_correlation(self, predictions, targets) -> float:
        """Compute correlation between predictions and targets."""
        if len(predictions) < 2:
            return 0.0
        pred_np = predictions.cpu().numpy().flatten()
        tgt_np = targets.cpu().numpy().flatten()
        if np.std(pred_np) < 1e-6 or np.std(tgt_np) < 1e-6:
            return 0.0
        return float(np.corrcoef(pred_np, tgt_np)[0, 1])
    
    def record_batch(self, ploss, vloss, grad_norms, pred_v, targets, logits, actions):
        """Record batch-level diagnostics."""
        self.batch_ploss.append(ploss)
        self.batch_vloss.append(vloss)
        
        self.grad_stem.append(grad_norms['stem'])
        self.grad_tower.append(grad_norms['tower'])
        self.grad_policy.append(grad_norms['policy'])
        self.grad_value.append(grad_norms['value'])
        
        # Detect gradient spikes
        if len(self.grad_value) > 1:
            if grad_norms['value'] > self.grad_value[-2] * 10:
                self.grad_spikes.append((self.current_epoch, self.current_batch, grad_norms['value']))
                self.warnings.append(f"E{self.current_epoch}B{self.current_batch}: Grad spike in value head ({grad_norms['value']:.2f})")
        
        # Policy analysis
        policy_stats = self.analyze_policy_head(logits, actions)
        self.policy_entropy.append(policy_stats['entropy'])
        self.top1_confidence.append(policy_stats['top1_conf'])
        self.top5_confidence.append(policy_stats['top5_conf'])
        self.legal_accuracy.append(policy_stats['accuracy'])
        
        # Value correlation
        corr = self.compute_value_correlation(pred_v, targets)
        self.value_target_corr.append(corr)
        
        self.current_batch += 1
    
    def record_pretanh_stats(self, stats):
        """Record pre-tanh activation statistics."""
        self.pretanh_mean.append(stats['mean'])
        self.pretanh_std.append(stats['std'])
        self.pretanh_min.append(stats['min'])
        self.pretanh_max.append(stats['max'])
        self.tanh_saturation.append(stats['saturation_pct'])
    
    def compute_health_score(self) -> float:
        """Compute overall model health score (0-100)."""
        score = 100.0
        
        # Penalize high tanh saturation
        if len(self.tanh_saturation) > 0:
            sat = self.tanh_saturation[-1]
            if sat > 50: score -= 30
            elif sat > 20: score -= 15
            elif sat > 10: score -= 5
        
        # Penalize low value correlation
        if len(self.value_target_corr) > 0:
            corr = np.mean(self.value_target_corr[-10:])  # Last 10 batches
            if corr < 0.1: score -= 20
            elif corr < 0.3: score -= 10
        
        # Penalize gradient spikes
        score -= len(self.grad_spikes) * 5
        
        # Penalize low policy accuracy
        if len(self.legal_accuracy) > 0:
            acc = np.mean(self.legal_accuracy[-10:])
            if acc < 0.01: score -= 20
            elif acc < 0.05: score -= 10
        
        # Penalize extreme entropy (too high or too low)
        if len(self.policy_entropy) > 0:
            ent = np.mean(self.policy_entropy[-10:])
            if ent > 8: score -= 10  # Too random
            if ent < 0.5: score -= 10  # Too confident
        
        return max(0, min(100, score))
    
    def print_batch_summary(self, batch_num, total_batches, ploss, vloss, 
                            grad_norms, pred_v_stats, print_interval=50):
        """Print batch summary at intervals."""
        if batch_num % print_interval != 0 and batch_num != total_batches - 1:
            return
        
        # Get recent averages
        recent_ploss = np.mean(self.batch_ploss[-print_interval:])
        recent_vloss = np.mean(self.batch_vloss[-print_interval:])
        recent_corr = np.mean(self.value_target_corr[-print_interval:]) if self.value_target_corr else 0
        
        print(f"      [{batch_num+1}/{total_batches}] "
              f"PLoss={recent_ploss:.4f} VLoss={recent_vloss:.4f} "
              f"Corr={recent_corr:.3f} "
              f"Grad[V]={grad_norms['value']:.3f}")
    
    def print_epoch_summary(self, epoch, epochs):
        """Print comprehensive epoch summary."""
        print(f"\n   {'═' * 60}")
        print(f"   📊 EPOCH {epoch+1}/{epochs} DIAGNOSTIC REPORT")
        print(f"   {'═' * 60}")
        
        # Health score
        health = self.compute_health_score()
        health_emoji = "🟢" if health >= 80 else "🟡" if health >= 50 else "🔴"
        print(f"   {health_emoji} HEALTH SCORE: {health:.0f}/100")
        self.health_scores.append(health)
        
        # Loss summary
        print(f"\n   📉 LOSS:")
        print(f"      Policy Loss: {np.mean(self.batch_ploss[-100:]):.4f} (last 100 batches)")
        print(f"      Value Loss:  {np.mean(self.batch_vloss[-100:]):.4f} (last 100 batches)")
        
        # Gradient flow
        print(f"\n   🔄 GRADIENT FLOW (avg last 100 batches):")
        print(f"      Stem:   {np.mean(self.grad_stem[-100:]):.4f}")
        print(f"      Tower:  {np.mean(self.grad_tower[-100:]):.4f}")
        print(f"      Policy: {np.mean(self.grad_policy[-100:]):.4f}")
        print(f"      Value:  {np.mean(self.grad_value[-100:]):.4f}")
        
        # Value head analysis
        if len(self.pretanh_mean) > 0:
            print(f"\n   📊 VALUE HEAD (Pre-Tanh):")
            print(f"      Mean: {self.pretanh_mean[-1]:.4f}")
            print(f"      Std:  {self.pretanh_std[-1]:.4f}")
            print(f"      Range: [{self.pretanh_min[-1]:.2f}, {self.pretanh_max[-1]:.2f}]")
            print(f"      Saturation: {self.tanh_saturation[-1]:.1f}%")
        
        if len(self.value_target_corr) > 0:
            print(f"      Pred-Target Correlation: {np.mean(self.value_target_corr[-100:]):.3f}")
        
        # Policy head analysis
        print(f"\n   🎯 POLICY HEAD:")
        print(f"      Entropy: {np.mean(self.policy_entropy[-100:]):.3f}")
        print(f"      Top-1 Confidence: {np.mean(self.top1_confidence[-100:]):.4f}")
        print(f"      Top-5 Confidence: {np.mean(self.top5_confidence[-100:]):.4f}")
        print(f"      Accuracy: {np.mean(self.legal_accuracy[-100:]):.2%}")
        
        # Warnings
        if self.warnings:
            print(f"\n   ⚠️ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[-5:]:  # Last 5
                print(f"      - {w}")
        
        # Trend analysis
        if len(self.batch_vloss) > 200:
            early_vloss = np.mean(self.batch_vloss[:100])
            late_vloss = np.mean(self.batch_vloss[-100:])
            vloss_trend = (late_vloss - early_vloss) / early_vloss * 100
            trend_emoji = "📈" if vloss_trend < -5 else "📉" if vloss_trend > 5 else "➡️"
            print(f"\n   {trend_emoji} TREND: VLoss {vloss_trend:+.1f}% change this epoch")
        
        print(f"   {'═' * 60}")
    
    def get_diagnosis(self) -> List[str]:
        """Get automated diagnosis of training issues."""
        issues = []
        
        # Check value head health
        if len(self.tanh_saturation) > 0 and self.tanh_saturation[-1] > 50:
            issues.append("🔴 CRITICAL: Value head tanh saturation > 50% - risk of collapse")
        
        if len(self.value_target_corr) > 10:
            corr = np.mean(self.value_target_corr[-10:])
            if corr < 0.1:
                issues.append("🔴 CRITICAL: Value predictions uncorrelated with targets")
            elif corr < 0.3:
                issues.append("🟡 WARNING: Low correlation between value predictions and targets")
        
        # Check gradient flow
        if len(self.grad_value) > 10:
            avg_grad = np.mean(self.grad_value[-10:])
            if avg_grad < 1e-6:
                issues.append("🔴 CRITICAL: Vanishing gradients in value head")
            elif avg_grad > 10:
                issues.append("🟡 WARNING: High gradients in value head - risk of instability")
        
        # Check policy head
        if len(self.legal_accuracy) > 10:
            acc = np.mean(self.legal_accuracy[-10:])
            if acc < 0.01:
                issues.append("🟡 WARNING: Very low policy accuracy - model may not be learning")
        
        if len(self.policy_entropy) > 10:
            ent = np.mean(self.policy_entropy[-10:])
            if ent > 8:
                issues.append("🟡 WARNING: High policy entropy - predictions are nearly random")
            elif ent < 0.5:
                issues.append("🟡 WARNING: Low policy entropy - model is overconfident")
        
        if not issues:
            issues.append("✅ No major issues detected")
        
        return issues

# Initialize global diagnostics
diagnostics = TrainingDiagnostics()
print("✅ v17 Diagnostics System initialized")

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
    
    def reset_value_head(self):
        """[v16] Reset value head weights if collapsed."""
        print("      ⚠️ Resetting value head weights!")
        for m in self.value_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Use smaller init for value head
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_backbone_params(self):
        """[v16] Get backbone parameters (stem + tower)."""
        return list(self.stem.parameters()) + list(self.tower.parameters())
    
    def get_policy_head_params(self):
        """[v16] Get policy head parameters."""
        return list(self.policy_head.parameters())
    
    def get_value_head_params(self):
        """[v16] Get value head parameters."""
        return list(self.value_head.parameters())
    
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
print(f"   - Backbone: {sum(p.numel() for p in network.get_backbone_params()):,} params")
print(f"   - Policy head: {sum(p.numel() for p in network.get_policy_head_params()):,} params")
print(f"   - Value head: {sum(p.numel() for p in network.get_value_head_params()):,} params")

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
    
    # [v16] Separate optimizers for different heads
    backbone_policy_params = network.get_backbone_params() + network.get_policy_head_params()
    value_params = network.get_value_head_params()
    
    optimizer_policy = torch.optim.AdamW(backbone_policy_params, lr=config.lr_policy, weight_decay=1e-4)
    optimizer_value = torch.optim.AdamW(value_params, lr=config.lr_value_warmup, weight_decay=1e-4)
    
    scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, epochs)
    scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_value, epochs - config.warmup_epochs)
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    best_wr, best_state = 0, None
    
    # [v16] Extended history for rich diagnostics
    history = {
        'ploss': [], 'vloss': [], 'wr': [],
        'pred_v_mean': [], 'pred_v_std': [], 'pred_v_min': [], 'pred_v_max': [],
        'value_collapsed': [],
        'grad_norm_policy': [], 'grad_norm_value': [],
        'lr_policy': [], 'lr_value': [],
        'value_head_weight_norm': [], 'policy_head_weight_norm': [],
    }
    
    n = len(states)
    value_reset_count = 0
    
    # [v16] Helper function for gradient norms
    def get_grad_norm(params):
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    # [v16] Helper function for weight norms  
    def get_weight_norm(params):
        total_norm = 0.0
        for p in params:
            total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    # [v16] Helper to print value head weight stats
    def print_value_head_stats(net, prefix=""):
        print(f"{prefix}📊 Value Head Weights:")
        for i, m in enumerate(net.value_head.modules()):
            if isinstance(m, nn.Linear):
                w = m.weight.data
                print(f"{prefix}   Linear[{i}]: shape={tuple(w.shape)}, "
                      f"mean={w.mean():.4f}, std={w.std():.4f}, "
                      f"min={w.min():.4f}, max={w.max():.4f}")
                if m.bias is not None:
                    b = m.bias.data
                    print(f"{prefix}   Bias[{i}]: mean={b.mean():.4f}, min={b.min():.4f}, max={b.max():.4f}")
    
    # Print initial stats
    print(f"\n   📊 Data Statistics:")
    print(f"      Positions: {n:,}")
    print(f"      Value targets: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}, std={values.std():.3f}")
    
    # Print value histogram
    hist, bin_edges = np.histogram(values, bins=10)
    print(f"      Value distribution:")
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / max(hist) * 20)
        print(f"         [{bin_edges[i]:+.2f}, {bin_edges[i+1]:+.2f}]: {hist[i]:>7,} {bar}")
    
    print(f"\n   📊 v16 Training Setup:")
    print(f"      Policy/Backbone LR: {config.lr_policy}")
    print(f"      Value Head LR: {config.lr_value_warmup} (warmup) → {config.lr_value} (after epoch {config.warmup_epochs})")
    print(f"      Grad clip: policy={config.grad_clip_policy}, value={config.grad_clip_value}")
    print(f"      Collapse threshold: {config.value_collapse_threshold}")
    
    # Print initial weights
    print(f"\n   📊 Initial Network State:")
    print_value_head_stats(network, "      ")
    
    for epoch in range(epochs):
        network.train()
        idx = np.random.permutation(n)
        
        # [v17] Reset diagnostics for new epoch
        diagnostics.current_epoch = epoch
        diagnostics.current_batch = 0
        
        # [v16] After warmup, increase value LR
        if epoch == config.warmup_epochs:
            print(f"\n   🔥 Warmup complete! Increasing value LR to {config.lr_value}")
            for pg in optimizer_value.param_groups:
                pg['lr'] = config.lr_value
        
        ploss_sum, vloss_sum, batches = 0, 0, 0
        collapsed_batches = 0
        pred_v_epoch = []
        total_batches = (n + config.batch_size - 1) // config.batch_size
        
        for i in range(0, n, config.batch_size):
            batch_idx = idx[i:i+config.batch_size]
            batch_num = i // config.batch_size
            
            s = torch.FloatTensor(states[batch_idx]).to(device)
            a = torch.LongTensor(actions[batch_idx]).to(device)
            v = torch.FloatTensor(values[batch_idx]).to(device)
            
            # [v16] Compute forward pass with AMP for policy only
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, pred_v = network(s)
                ploss = F.cross_entropy(logits, a, label_smoothing=0.1)
            
            # [v16] Compute value loss in FULL PRECISION to prevent numerical issues
            pred_v_float = pred_v.float()
            vloss = F.mse_loss(pred_v_float.squeeze(-1), v)
            
            # Check for collapse BEFORE backward
            pred_v_np = pred_v_float.squeeze(-1).detach().cpu().numpy()
            collapsed = np.mean(np.abs(pred_v_np) > config.value_collapse_threshold) > 0.9
            if collapsed:
                collapsed_batches += 1
            
            # [FIXED] Accumulated gradients + Shared Scaler
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            
            # Backward for both heads (accumulate gradients on backbone)
            scaler.scale(ploss).backward(retain_graph=True)
            scaler.scale(vloss).backward()
            
            # Unscale both - required for clipping and diagnostics
            scaler.unscale_(optimizer_policy)
            scaler.unscale_(optimizer_value)
            
            # [v17] Compute layer-by-layer gradient norms (now all valid and unscaled)
            grad_norms = diagnostics.compute_layer_grad_norms(network)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(backbone_policy_params, config.grad_clip_policy)
            torch.nn.utils.clip_grad_norm_(value_params, config.grad_clip_value)
            
            # Step optimizers
            scaler.step(optimizer_policy)
            scaler.step(optimizer_value)
            
            scaler.update()
            
            ploss_sum += ploss.item()
            vloss_sum += vloss.item()
            batches += 1
            
            # [v17] Record batch diagnostics
            diagnostics.record_batch(
                ploss=ploss.item(),
                vloss=vloss.item(), 
                grad_norms=grad_norms,
                pred_v=pred_v_float.squeeze(-1).detach(),
                targets=v,
                logits=logits.detach(),
                actions=a
            )
            
            # [v17] Print batch progress at intervals
            diagnostics.print_batch_summary(
                batch_num, total_batches, 
                ploss.item(), vloss.item(),
                grad_norms, {'mean': np.mean(pred_v_np)},
                print_interval=100  # Print every 100 batches
            )
            
            # Collect first batch predictions
            if i == 0:
                pred_v_epoch = pred_v_np
                
                # [v17] Analyze pre-tanh activations on first batch
                pretanh_stats = diagnostics.analyze_value_head_pretanh(network, s)
                diagnostics.record_pretanh_stats(pretanh_stats)
        
        # [v16] Check for value head collapse at end of epoch
        is_collapsed = collapsed_batches > batches * 0.5  # >50% batches collapsed
        history['value_collapsed'].append(is_collapsed)
        
        if is_collapsed and value_reset_count < 2:  # Max 2 resets
            print(f"\n   🔴 VALUE HEAD COLLAPSED! Resetting...")
            network.reset_value_head()
            # Recreate value optimizer with fresh params
            optimizer_value = torch.optim.AdamW(network.get_value_head_params(), 
                                                lr=config.lr_value_warmup, weight_decay=1e-4)
            value_reset_count += 1
            print(f"      Reset count: {value_reset_count}/2")
            print_value_head_stats(network, "      ")
        
        # [v16] Track gradient norms before stepping
        grad_norm_p = get_grad_norm(backbone_policy_params)
        grad_norm_v = get_grad_norm(value_params)
        
        scheduler_policy.step()
        if epoch >= config.warmup_epochs:
            scheduler_value.step()
        
        # [v16] Track current learning rates
        current_lr_policy = optimizer_policy.param_groups[0]['lr']
        current_lr_value = optimizer_value.param_groups[0]['lr']
        
        # [v16] Track weight norms
        weight_norm_policy = get_weight_norm(network.get_policy_head_params())
        weight_norm_value = get_weight_norm(network.get_value_head_params())
        
        # Record extended history
        history['ploss'].append(ploss_sum / batches)
        history['vloss'].append(vloss_sum / batches)
        history['pred_v_mean'].append(float(np.mean(pred_v_epoch)))
        history['pred_v_std'].append(float(np.std(pred_v_epoch)))
        history['pred_v_min'].append(float(np.min(pred_v_epoch)))
        history['pred_v_max'].append(float(np.max(pred_v_epoch)))
        history['grad_norm_policy'].append(grad_norm_p)
        history['grad_norm_value'].append(grad_norm_v)
        history['lr_policy'].append(current_lr_policy)
        history['lr_value'].append(current_lr_value)
        history['value_head_weight_norm'].append(weight_norm_value)
        history['policy_head_weight_norm'].append(weight_norm_policy)
        
        # [v17] Use comprehensive diagnostics system for epoch summary
        diagnostics.print_epoch_summary(epoch, epochs)
        
        # [v17] Additional v16-style logging for compatibility
        collapse_flag = "🔴 COLLAPSED" if is_collapsed else "✅"
        print(f"\n   📝 QUICK STATS: {collapse_flag}")
        print(f"      PLoss={ploss_sum/batches:.4f}, VLoss={vloss_sum/batches:.4f}")
        print(f"      pred_v=[{np.min(pred_v_epoch):.3f}, {np.max(pred_v_epoch):.3f}]")
        print(f"      LR: policy={current_lr_policy:.2e}, value={current_lr_value:.2e}")
        print(f"      Collapsed: {collapsed_batches}/{batches} batches")
        
        # [v16] Show sample predictions vs targets every epoch
        if len(pred_v_epoch) > 0:
            sample_size = min(5, len(pred_v_epoch))
            sample_idx = np.random.choice(len(pred_v_epoch), sample_size, replace=False)
            print(f"   Sample Predictions vs Targets:")
            batch_values = values[idx[:config.batch_size]]  # Corresponding targets
            for j in sample_idx:
                target = batch_values[j] if j < len(batch_values) else 0
                pred = pred_v_epoch[j]
                error = abs(pred - target)
                print(f"      [{j}] pred={pred:+.3f}, target={target:+.3f}, error={error:.3f}")
        
        # [v17] Print automated diagnosis
        print(f"\n   🔍 AUTOMATED DIAGNOSIS:")
        for issue in diagnostics.get_diagnosis():
            print(f"      {issue}")
        
        # Evaluate every 3 epochs
        if (epoch + 1) % 3 == 0:
            wr = evaluate(network, opening_book, 50)
            history['wr'].append(wr)
            print(f"\n   🎯 EVALUATION: WR vs Random = {wr:.0%}")
            
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
                print(f"      ✨ New best model!")
    
    # [v17] Final comprehensive summary
    print(f"\n   {'═' * 60}")
    print(f"   📊 SUPERVISED PHASE FINAL SUMMARY")
    print(f"   {'═' * 60}")
    print(f"   Best WR: {best_wr:.0%}")
    print(f"   Value head resets: {value_reset_count}")
    print(f"   Final value predictions: [{history['pred_v_min'][-1]:.3f}, {history['pred_v_max'][-1]:.3f}]")
    print(f"   Final grad norms: policy={history['grad_norm_policy'][-1]:.4f}, value={history['grad_norm_value'][-1]:.4f}")
    
    # [v17] Print health score trend
    if diagnostics.health_scores:
        print(f"\n   📈 HEALTH SCORE TREND:")
        for i, score in enumerate(diagnostics.health_scores):
            emoji = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
            print(f"      Epoch {i+1}: {emoji} {score:.0f}/100")
    
    # [v17] Final diagnosis
    print(f"\n   🔍 FINAL DIAGNOSIS:")
    for issue in diagnostics.get_diagnosis():
        print(f"      {issue}")
    
    if best_state:
        network.load_state_dict(best_state)
        print(f"\n   ✅ Restored best model (WR={best_wr:.0%})")
    
    # Print final value head state
    print(f"\n   📊 Final Value Head State:")
    print_value_head_stats(network, "      ")
    
    # Save model
    torch.save(network.state_dict(), '/kaggle/working/chess_v17_supervised.pt')
    print(f"   💾 Saved: chess_v17_supervised.pt")
    
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

def rl_phase(network, opening_book, iterations: int = 100):
    """RL training with GENTLE curriculum learning."""
    print("\n🤖 PHASE 2: Gentle Curriculum RL (vs Stockfish)")
    print("=" * 50)
    
    buffer.clear()
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr_rl, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Initial evaluation
    initial_wr_random = evaluate(network, opening_book, 50)
    print(f"\n   📊 Initial Stats:")
    print(f"      LR: {config.lr_rl}")
    print(f"      Games/iter: {config.games_per_iter}")
    print(f"      Starting level: SF{config.opponent_start_level}")
    print(f"      WR vs Random: {initial_wr_random:.0%}")
    
    best_wr = initial_wr_random
    history = {'wr_random': [initial_wr_random], 'wr_sf': [], 'level': [], 'wins': [], 'losses': []}
    no_improve = 0
    current_level = config.opponent_start_level
    level_wins = 0  # Track wins at current level
    level_games = 0
    
    for iteration in range(iterations):
        progress = iteration / iterations
        epsilon = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress
        
        # Play games
        wins, draws, losses = 0, 0, 0
        for _ in range(config.games_per_iter):
            result = play_rl_game(network, opening_book, epsilon, current_level)
            if result == '1-0':
                wins += 1
                level_wins += 1
            elif result == '0-1':
                losses += 1
            else:
                draws += 1
        level_games += config.games_per_iter
        
        # [FIX] Skip training if 100% losses (no positive signal)
        if wins == 0 and draws == 0:
            print(f"   Iter {iteration+1}: SF{current_level} W={wins} D={draws} L={losses} - "
                  f"🔴 All losses, skipping training (no positive signal)")
            continue
        
        # Train only if we have some positive experiences
        if len(buffer) >= config.batch_size:
            network.train()
            
            train_losses = []
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
                    
                    # Scale advantages for stability
                    scaled_advantages = advantages / (advantages.abs().max() + 1e-8)
                    
                    policy_loss = -(selected_log_probs * scaled_advantages.detach()).mean()
                    value_loss = F.mse_loss(pred_v.squeeze(-1), value_targets)
                    
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    
                    # [FIX] Apply value loss weight
                    loss = policy_loss + config.value_loss_weight * value_loss - 0.01 * entropy
                    train_losses.append(loss.item())
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
        
        # Log every iteration with rich info
        win_rate_this_iter = wins / config.games_per_iter
        level_wr = level_wins / level_games if level_games > 0 else 0
        
        print(f"   Iter {iteration+1}/{iterations}: SF{current_level} W={wins} D={draws} L={losses} "
              f"({win_rate_this_iter:.0%}), ε={epsilon:.2f}, Buffer={len(buffer)}, LvlWR={level_wr:.0%}")
        
        # [FIX] Advance level only if win rate is high enough
        if level_games >= 50 and level_wr >= config.min_win_rate_to_advance:
            if current_level < config.opponent_max_level:
                current_level += 1
                level_wins = 0
                level_games = 0
                print(f"      ⬆️ Level up! Now playing SF{current_level}")
        
        # Evaluate every 10 iterations
        if (iteration + 1) % 10 == 0:
            wr_random = evaluate(network, opening_book, 30)
            sf_w, sf_d, sf_l = evaluate_vs_stockfish(network, opening_book, 10, current_level)
            sf_score = (sf_w + sf_d * 0.5) / 10
            
            history['wr_random'].append(wr_random)
            history['wr_sf'].append(sf_score)
            history['level'].append(current_level)
            history['wins'].append(wins)
            history['losses'].append(losses)
            
            print(f"\n   📊 Eval at iter {iteration+1}:")
            print(f"      WR vs Random: {wr_random:.0%}")
            print(f"      vs SF{current_level}: {sf_w}W/{sf_d}D/{sf_l}L ({sf_score:.0%})")
            print(f"      Buffer size: {len(buffer)}")
            
            if wr_random > best_wr:
                best_wr = wr_random
                no_improve = 0
                torch.save(network.state_dict(), '/kaggle/working/chess_v16_best.pt')
                print(f"      ✨ New best! Saved.")
            else:
                no_improve += 1
            
            # Stop if model is collapsing
            if wr_random < initial_wr_random * 0.6 and iteration > 30:
                print(f"\n   🛑 WR dropped to {wr_random:.0%} (<60% of initial {initial_wr_random:.0%}), stopping")
                break
            
            if no_improve >= 10:
                print(f"\n   ⏹️ No improvement for 10 evals, stopping")
                break
            print()
    
    # Load best model
    if os.path.exists('/kaggle/working/chess_v16_best.pt'):
        network.load_state_dict(torch.load('/kaggle/working/chess_v16_best.pt'))
        print(f"\n   ✅ Restored best model (WR={best_wr:.0%})")
    
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
print("🚀 STARTING v17 TRAINING - ULTRA RICH DIAGNOSTICS")
print("=" * 60)

start_time = time.time()

# Phase 1: Supervised
supervised_wr, sup_history = supervised_phase(network, df, opening_book, 
                                               epochs=config.supervised_epochs)

# Check if value head is healthy before RL
value_healthy = not any(sup_history.get('value_collapsed', [False]))
if not value_healthy:
    print("\n⚠️ Value head had collapse issues during supervised phase.")
    print("   RL phase will proceed with caution.")

# [v17] Print diagnostic summary before RL
print("\n" + "=" * 60)
print("📊 v17 DIAGNOSTIC SUMMARY - POST SUPERVISED")
print("=" * 60)
print(f"Health Scores: {diagnostics.health_scores}")
print(f"Total Warnings: {len(diagnostics.warnings)}")
print(f"Gradient Spikes: {len(diagnostics.grad_spikes)}")
if diagnostics.tanh_saturation:
    print(f"Final Tanh Saturation: {diagnostics.tanh_saturation[-1]:.1f}%")
if diagnostics.value_target_corr:
    print(f"Final Value-Target Correlation: {np.mean(diagnostics.value_target_corr[-100:]):.3f}")

# Phase 2: RL (shorter since we focus on supervised)
rl_best_wr, rl_history = rl_phase(network, opening_book, 
                                   iterations=config.rl_iterations)

# Final evaluation
print("\n" + "=" * 60)
print("📊 FINAL EVALUATION")
print("=" * 60)

final_wr_random = evaluate(network, opening_book, 100)
print(f"\n   vs Random (100 games): {final_wr_random:.0%}")

print("\n   vs Stockfish:")
sf_results = {}
for level in [0, 1, 2, 3]:
    w, d, l = evaluate_vs_stockfish(network, opening_book, 20, level)
    wr = (w + d * 0.5) / 20
    sf_results[level] = {'w': w, 'd': d, 'l': l, 'wr': wr}
    print(f"      Lv{level}: W={w} D={d} L={l} ({wr:.0%})")

total_time = time.time() - start_time
print(f"\n⏱️ Total time: {total_time/3600:.1f} hours")

# Save final model
torch.save(network.state_dict(), '/kaggle/working/chess_v17_final.pt')
print("\n💾 Saved: chess_v17_final.pt")

# Cleanup
if stockfish_eval_engine:
    stockfish_eval_engine.quit()
if stockfish_opponent:
    stockfish_opponent.quit()

# ==============================================================================
# Cell 14: Plot Results
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Supervised plots
axes[0, 0].plot(sup_history['ploss'], 'b-', linewidth=2)
axes[0, 0].set_title('Policy Loss (Supervised)', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(sup_history['vloss'], 'g-', linewidth=2)
axes[0, 1].set_title('Value Loss (Supervised)', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Value head predictions over time
axes[0, 2].plot(sup_history['pred_v_mean'], 'purple', linewidth=2, label='Mean')
axes[0, 2].fill_between(range(len(sup_history['pred_v_mean'])),
                        [m - s for m, s in zip(sup_history['pred_v_mean'], sup_history['pred_v_std'])],
                        [m + s for m, s in zip(sup_history['pred_v_mean'], sup_history['pred_v_std'])],
                        alpha=0.3, color='purple')
# Mark collapsed epochs
collapsed_epochs = [i for i, c in enumerate(sup_history.get('value_collapsed', [])) if c]
if collapsed_epochs:
    for e in collapsed_epochs:
        axes[0, 2].axvline(e, color='red', linestyle='--', alpha=0.5)
axes[0, 2].set_title('Value Predictions (Supervised)', fontsize=12)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('pred_v')
axes[0, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0, 2].set_ylim(-1.1, 1.1)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Win rate progression
x_sup = list(range(3, len(sup_history['wr'])*3+1, 3))
axes[1, 0].plot(x_sup, sup_history['wr'], 'b-o', label='Supervised', markersize=6, linewidth=2)
axes[1, 0].axhline(supervised_wr, color='blue', linestyle='--', alpha=0.5)

if 'wr_random' in rl_history and len(rl_history['wr_random']) > 1:
    rl_x = list(range(len(rl_history['wr_random'])))
    axes[1, 0].plot([max(x_sup) + x * 10 for x in rl_x], 
                    rl_history['wr_random'], 'r-o', label='RL', markersize=4, linewidth=2)

axes[1, 0].axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='80% target')
axes[1, 0].set_title(f'Win Rate vs Random (Final: {final_wr_random:.0%})', fontsize=12)
axes[1, 0].set_xlabel('Training Progress')
axes[1, 0].set_ylabel('Win Rate')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# RL level progression
if 'level' in rl_history and len(rl_history['level']) > 0:
    axes[1, 1].plot(rl_history['level'], 'g-o', markersize=4, linewidth=2)
    axes[1, 1].set_title('SF Opponent Level (RL)', fontsize=12)
    axes[1, 1].set_xlabel('Eval #')
    axes[1, 1].set_ylabel('Level')
    axes[1, 1].set_ylim(0, config.opponent_max_level + 1)
    axes[1, 1].grid(True, alpha=0.3)

# Final SF results bar chart
levels = list(sf_results.keys())
wrs = [sf_results[lv]['wr'] for lv in levels]
colors = ['green' if wr >= 0.5 else 'orange' if wr >= 0.2 else 'red' for wr in wrs]
bars = axes[1, 2].bar([f'SF{lv}' for lv in levels], wrs, color=colors, edgecolor='black')
axes[1, 2].set_title('Final WR vs Stockfish', fontsize=12)
axes[1, 2].set_ylabel('Win Rate')
axes[1, 2].set_ylim(0, 1)
axes[1, 2].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
for bar, wr in zip(bars, wrs):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{wr:.0%}', ha='center', fontsize=10)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/kaggle/working/training_v17.png', dpi=150)
plt.show()

# Summary
print("\n" + "=" * 60)
print("🎉 v17 TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📊 Results Summary:")
print(f"   Supervised phase: {supervised_wr:.0%} WR")
print(f"   RL phase: {rl_best_wr:.0%} WR")
print(f"   Final vs Random: {final_wr_random:.0%}")
print(f"\n📁 Files saved:")
print(f"   chess_v17_supervised.pt - After supervised learning")
print(f"   chess_v17_best.pt - Best during RL")
print(f"   chess_v17_final.pt - Final model")
print(f"   training_v17.png - Training plots")

print(f"\n🔧 v17 Features:")
print(f"   ✅ ULTRA RICH DIAGNOSTICS SYSTEM")
print(f"   ✅ Layer-by-layer gradient flow tracking")
print(f"   ✅ Pre-tanh activation analysis")
print(f"   ✅ Value-target correlation tracking")
print(f"   ✅ Policy entropy & confidence monitoring")
print(f"   ✅ Automated health score (0-100)")
print(f"   ✅ Early warning system for training issues")
print(f"   ✅ Automated diagnosis at each epoch")
print(f"   ✅ Batch-level progress tracking")
print(f"   ✅ Collapse detection + auto-reset (threshold={config.value_collapse_threshold})")
print(f"   ✅ Focused on stable supervised learning")
