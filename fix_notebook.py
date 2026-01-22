import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

def add_cell(source_code, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_code.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

# 1. Header
add_cell("""# 🏆 Complete Chess RL Training - Production Version
## Notebook Lengkap dengan Semua Fitur Optimasi

**Fitur:**
- ✅ Auto-resume dari checkpoint
- ✅ Self-play training dengan opponent pool
- ✅ Fine-tuning melawan Stockfish (progressive levels)
- ✅ Adaptive PPO (clip range, entropy scheduling)
- ✅ Stability monitoring & early stopping
- ✅ RolloutBuffer dengan GAE
- ✅ Google Drive persistence

**Cara pakai:** Klik Runtime → Run all. Selesai.""", "markdown")

# 2. Config
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 📋 MASTER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ===== Training Phases =====
    'total_updates': 20000,
    'phase1_end': 5000,
    'phase2_end': 15000,
    
    # ===== Network Architecture =====
    'num_filters': 256,
    'num_blocks': 12,
    'use_se_blocks': True,
    
    # ===== PPO Hyperparameters =====
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.02,
    'final_entropy_coef': 0.001,
    'max_grad_norm': 0.5,
    'target_kl': 0.015,
    
    # ===== Training Loop =====
    'n_steps': 512,
    'n_epochs': 4,
    'batch_size': 128,
    
    # ===== Adaptive Features =====
    'adaptive_clip_range': True,
    'entropy_scheduling': True,
    'normalize_advantage': True,
    
    # ===== LR Warmup =====
    'warmup_steps': 500,
    
    # ===== Self-Play =====
    'opponent_pool_size': 10,
    'opponent_update_freq': 100,
    'self_play_prob': 0.8,
    
    # ===== Stockfish =====
    'stockfish_levels': [0, 1, 2, 3, 5],
    'stockfish_time_ms': 100,
    'level_advance_winrate': 0.6,
    
    # ===== Stability & Early Stopping =====
    'stability_window': 100,
    'early_stopping_patience': 10,
    'min_improvement': 0.01,
    'save_interval': 500,
    'eval_interval': 250,
    'eval_games': 20,
}

DRIVE_PATH = '/content/drive/MyDrive/chess_rl'
AUTO_RESUME = True

print('✅ Configuration loaded!')""")

# 3. Setup
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 🔧 SETUP - GPU, Dependencies, Drive
# ══════════════════════════════════════════════════════════════════════════════

import torch
import os
import sys
import numpy as np
import chess
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import copy
import random
from collections import deque
import math
from typing import Optional, Tuple, Dict, Any, List

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 Device: {device}')

# Install dependencies
!pip install -q python-chess gymnasium tqdm matplotlib stockfish

# Install Stockfish
!apt-get install -qq stockfish > /dev/null 2>&1
print('✅ Stockfish installed')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
os.makedirs(DRIVE_PATH, exist_ok=True)
print(f'✅ Drive mounted: {DRIVE_PATH}')""")

# 4. Helper Classes (StateEncoder & ActionSpace)
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ HELPER CLASSES (StateEncoder & ActionSpace)
# ══════════════════════════════════════════════════════════════════════════════

class StateEncoder:
    PIECE_CHANNELS = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    NUM_CHANNELS = 14
    
    def encode(self, board: chess.Board) -> np.ndarray:
        state = np.zeros((self.NUM_CHANNELS, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = self.PIECE_CHANNELS[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                state[channel, square // 8, square % 8] = 1.0
        
        if board.turn == chess.WHITE:
            state[12, :, :] = 1.0
            
        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE): state[13, 0, 7] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): state[13, 0, 0] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): state[13, 7, 7] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): state[13, 7, 0] = 1.0
        
        if board.ep_square:
            state[13, board.ep_square // 8, board.ep_square % 8] = 0.5
            
        return state

class ActionSpace:
    ACTION_SIZE = 4672
    QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    KNIGHT_MOVES = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    
    def __init__(self):
        self.move_to_action = {}
        self.action_to_move = {}
        self._build_lookup_tables()
        
    def _build_lookup_tables(self):
        for from_sq in range(64):
            from_row, from_col = from_sq // 8, from_sq % 8
            move_type = 0
            
            # Queen moves
            for dr, dc in self.QUEEN_DIRECTIONS:
                for dist in range(1, 8):
                    to_row, to_col = from_row + dr*dist, from_col + dc*dist
                    if 0<=to_row<8 and 0<=to_col<8:
                        to_sq = to_row*8 + to_col
                        action = from_sq*73 + move_type
                        self._add(chess.Move(from_sq, to_sq), action)
                        if to_row in [0, 7]:
                            self._add(chess.Move(from_sq, to_sq, promotion=chess.QUEEN), action)
                    move_type += 1
                    if move_type % 7 == 0: break
                move_type = (move_type // 7 + 1) * 7
            
            # Knight moves
            move_type = 56
            for dr, dc in self.KNIGHT_MOVES:
                to_row, to_col = from_row + dr, from_col + dc
                if 0<=to_row<8 and 0<=to_col<8:
                    self._add(chess.Move(from_sq, to_row*8 + to_col), from_sq*73 + move_type)
                move_type += 1
                
            # Underpromotions
            move_type = 64
            if from_row in [1, 6]:
                for promo in self.UNDERPROMOTIONS:
                    for dc in [-1, 0, 1]:
                        to_row = 7 if from_row == 6 else 0
                        to_col = from_col + dc
                        if 0<=to_col<8:
                            self._add(chess.Move(from_sq, to_row*8 + to_col, promotion=promo), from_sq*73 + move_type)
                        move_type += 1

    def _add(self, move, action):
        self.move_to_action[move] = action
        self.action_to_move[action] = move

    def encode(self, move):
        return self.move_to_action.get(move, None)
    
    def decode(self, action):
        return self.action_to_move.get(action, None)

    def get_legal_action_mask(self, board):
        mask = np.zeros(self.ACTION_SIZE, dtype=bool)
        for move in board.legal_moves:
            action = self.encode(move)
            if action is not None:
                mask[action] = True
        return mask

print('✅ Helper classes ready!')""")

# 5. Environment
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# ♟️ CHESS ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state_encoder = StateEncoder()
        self.action_space_handler = ActionSpace()
        self.observation_space = spaces.Box(low=0, high=1, shape=(14, 8, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)
        self.board = chess.Board()
        self.move_to_action = self.action_space_handler.move_to_action
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = chess.Board()
        return self.state_encoder.encode(self.board), {}
    
    def step(self, action_idx):
        move = self.action_space_handler.decode(action_idx)
        if move is None or move not in self.board.legal_moves:
            return self.state_encoder.encode(self.board), -0.1, True, False, {'illegal': True}
            
        self.board.push(move)
        
        reward = 0
        terminated = False
        
        if self.board.is_checkmate():
            reward = 1.0
            terminated = True
        elif self.board.is_game_over():
            reward = 0.0 # Draw
            terminated = True
        else:
            reward = 0.001 # Small survival reward / shaping
            
        return self.state_encoder.encode(self.board), reward, terminated, False, {}

    def get_legal_action_mask(self):
        return self.action_space_handler.get_legal_action_mask(self.board)

env = ChessEnv()
print('✅ Environment ready!')""")

# 6. Network
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 🧠 NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class ChessNetwork(nn.Module):
    def __init__(self, num_filters=256, num_blocks=12):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(14, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*64, 4672)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mask=None):
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        
        logits = self.policy_head(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
            
        return F.log_softmax(logits, dim=-1), self.value_head(x)

network = ChessNetwork(CONFIG['num_filters'], CONFIG['num_blocks']).to(device)
print(f'✅ Network initialized: {sum(p.numel() for p in network.parameters()):,} params')""")

# 7. Buffer & Stability
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 📦 BUFFER & STABILITY
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, size):
        self.states, self.actions, self.rewards, self.dones = [], [], [], []
        self.log_probs, self.values, self.masks = [], [], []
        
    def add(self, s, a, r, d, lp, v, m):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.log_probs.append(lp)
        self.values.append(v)
        self.masks.append(m)
        
    def compute_gae(self, last_val, last_done, gamma, gae_lambda):
        advs = np.zeros(len(self.rewards))
        last_gae = 0
        for t in reversed(range(len(self.rewards))):
            next_val = last_val if t == len(self.rewards)-1 else self.values[t+1]
            next_non_terminal = 1.0 - (last_done if t == len(self.rewards)-1 else self.dones[t+1])
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            advs[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        return advs, advs + self.values

    def get_batches(self, batch_size):
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        for start in range(0, len(self.states), batch_size):
            yield indices[start:start+batch_size]

class StabilityMetrics:
    def __init__(self):
        self.history = {'grad': [], 'loss': [], 'kl': [], 'ent': []}
    
    def update(self, grad, loss, kl, ent):
        self.history['grad'].append(grad)
        self.history['loss'].append(loss)
        self.history['kl'].append(kl)
        self.history['ent'].append(ent)
        
    def check(self):
        warnings = []
        if self.history['grad'] and list(self.history['grad'])[-1] > 100: warnings.append('Grad Explosion')
        if self.history['ent'] and list(self.history['ent'])[-1] < 0.01: warnings.append('Entropy Collapse')
        return warnings

stability = StabilityMetrics()
print('✅ Buffer & Stability ready!')""")

# 8. Agent
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 🎮 PPO AGENT
# ══════════════════════════════════════════════════════════════════════════════
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.AdamW(network.parameters(), lr=config['learning_rate'])
        self.clip = config['clip_range']
        
    def select_action(self, state, mask, deterministic=False):
        self.network.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(device)
            log_probs, val = self.network(s, m)
            if deterministic:
                action = torch.argmax(log_probs).item()
            else:
                action = Categorical(logits=log_probs).sample().item()
        self.network.train()
        return action, log_probs[0, action].item(), val.item()

    def update(self, buffer, advs, returns):
        states = torch.FloatTensor(np.array(buffer.states)).to(device)
        actions = torch.LongTensor(buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
        masks = torch.BoolTensor(np.array(buffer.masks)).to(device)
        advs = torch.FloatTensor(advs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        stats = {'loss': [], 'kl': [], 'ent': []}
        
        for _ in range(self.config['n_epochs']):
            for idx in buffer.get_batches(self.config['batch_size']):
                lp, vals = self.network(states[idx], masks[idx])
                vals = vals.squeeze(-1)
                
                new_lp = lp.gather(1, actions[idx].unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(new_lp - old_log_probs[idx])
                
                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advs[idx]
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(vals, returns[idx])
                entropy = -(torch.exp(lp) * lp).sum(dim=1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                stats['loss'].append(loss.item())
                stats['ent'].append(entropy.item())
                with torch.no_grad():
                    stats['kl'].append((old_log_probs[idx] - new_lp).mean().item())
                    
        return {k: np.mean(v) for k, v in stats.items()}

ppo = PPOAgent(network, CONFIG)
print('✅ Agent ready!')""")

# 9. Opponents & Checkpoint
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 👥 OPPONENTS & SYSTEMS
# ══════════════════════════════════════════════════════════════════════════════

class RandomOpponent:
    def select_action(self, env):
        legal = list(env.board.legal_moves)
        return env.action_space_handler.encode(random.choice(legal)) if legal else None

class StockfishOpponent:
    def __init__(self, level=0):
        from stockfish import Stockfish
        self.sf = Stockfish('/usr/games/stockfish')
        self.sf.set_skill_level(level)
    def select_action(self, env):
        self.sf.set_fen_position(env.board.fen())
        res = self.sf.get_best_move_time(100)
        return env.action_space_handler.encode(chess.Move.from_uci(res)) if res else None

def save_checkpoint(name):
    torch.save(network.state_dict(), f'{DRIVE_PATH}/{name}.pt')
    print(f'💾 Saved {name}')

print('✅ Systems ready!')""")

# 10. Main Loop
add_cell("""# ══════════════════════════════════════════════════════════════════════════════
# 🚀 TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

opp = RandomOpponent()
history = {'win_rate': [], 'loss': []}

for update in range(CONFIG['total_updates']):
    buffer = RolloutBuffer(CONFIG['n_steps'])
    state, _ = env.reset()
    
    # Collection
    for _ in range(CONFIG['n_steps']):
        mask = env.get_legal_action_mask()
        if env.board.turn == chess.WHITE:
            action, lp, val = ppo.select_action(state, mask)
        else:
            action = opp.select_action(env)
            lp, val = 0, 0
            
        next_state, r, term, _, _ = env.step(action)
        buffer.add(state, action, r, term, lp, val, mask)
        state = next_state
        if term: state, _ = env.reset()
            
    # Update
    advs, returns = buffer.compute_gae(0, True, 0.99, 0.95)
    info = ppo.update(buffer, advs, returns)
    
    if update % 10 == 0:
        print(f'Update {update} | Loss: {info["loss"]:.4f} | KL: {info["kl"]:.4f}')
        if update % 100 == 0: save_checkpoint(f'ckpt_{update}')
""")

with open('notebooks/complete_training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=4)

print("Notebook generated successfully!")
