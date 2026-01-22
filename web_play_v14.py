#!/usr/bin/env python3
"""
=============================================================================
Web Interface for Chess AI v14 - WITH MCTS FOR TACTICAL PLAY
=============================================================================
Uses supervised model with MCTS at inference time for:
- Finding mate-in-1, mate-in-2
- Capturing hanging pieces
- Better tactical decisions

Based on v13 with MCTS integration.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
import os
import math
from flask import Flask, render_template_string, jsonify, request
from typing import Optional, Dict, List, Tuple


# =============================================================================
# Configuration
# =============================================================================

MCTS_SIMULATIONS = 30  # Simulations per move (adjustable via UI)
MCTS_CPUCT = 1.5       # Exploration constant
MODEL_PATH = r'checkpoints\chess_v21_best.pt'


# =============================================================================
# State Encoder (12 Channels)
# =============================================================================

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

def get_legal_actions(board: chess.Board) -> List[int]:
    return [encode_move(m) for m in board.legal_moves]


# =============================================================================
# Opening Book
# =============================================================================

OPENING_BOOK = {
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq': ['d7d5', 'g8f6', 'e7e6'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq': ['g1f3', 'f1c4', 'b1c3'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq': ['b8c6', 'g8f6'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq': ['f1b5', 'f1c4', 'd2d4'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq': ['f8c5', 'g8f6'],
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq': ['c2c4', 'g1f3'],
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq': ['e7e6', 'c7c6'],
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq': ['g1f3', 'b1c3', 'c2c3'],
}

def get_opening_book_move(board: chess.Board) -> Optional[str]:
    fen_parts = board.fen().split()
    fen_key = ' '.join(fen_parts[:3])
    
    if fen_key in OPENING_BOOK:
        moves = OPENING_BOOK[fen_key]
        legal = [m for m in moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal:
            return random.choice(legal)
    return None


# =============================================================================
# Neural Network
# =============================================================================

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
        """Get policy probabilities and value."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            return probs, value.item()


# =============================================================================
# MCTS Implementation
# =============================================================================

class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, board: chess.Board, parent=None, action=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, cpuct: float) -> float:
        prior_score = cpuct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        value_score = -self.value
        return value_score + prior_score
    
    def select_child(self, cpuct: float) -> 'MCTSNode':
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(self.visit_count, cpuct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray):
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
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search for tactical play."""
    
    def __init__(self, network: ChessNet, simulations: int = 30, cpuct: float = 1.5):
        self.network = network
        self.simulations = simulations
        self.cpuct = cpuct
    
    def search(self, board: chess.Board) -> Tuple[int, float, dict]:
        """
        Run MCTS and return best move.
        Returns: (best_action, root_value, debug_info)
        """
        root = MCTSNode(board)
        
        # Get initial policy from network
        state = encode_board(board)
        mask = get_legal_mask(board)
        policy_probs, _ = self.network.predict(state, mask)
        
        # Expand root
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(self.simulations):
            node = root
            
            # SELECT
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child(self.cpuct)
            
            # EVALUATE
            if node.board.is_game_over():
                result = node.board.result()
                if result == '1-0':
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == '0-1':
                    value = -1.0 if node.board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0
            else:
                state = encode_board(node.board)
                mask = get_legal_mask(node.board)
                policy_probs, value = self.network.predict(state, mask)
                node.expand(policy_probs)
            
            # BACKPROPAGATE
            node.backpropagate(value)
        
        # Select best move by visit count
        best_action = None
        best_visits = -1
        
        debug_info = {'moves': []}
        
        for action, child in root.children.items():
            move = decode_move(action, board)
            move_info = {
                'move': move.uci() if move else '???',
                'visits': int(child.visit_count),
                'value': float(-child.value),  # Negate for this player's perspective
                'prior': float(child.prior)
            }
            debug_info['moves'].append(move_info)
            
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        
        # Sort moves by visits for display
        debug_info['moves'].sort(key=lambda x: x['visits'], reverse=True)
        debug_info['total_simulations'] = self.simulations
        
        return best_action, float(root.value), debug_info


# =============================================================================
# HTML Template with MCTS controls
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>♟️ Chess AI v14 - MCTS</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container { display: flex; gap: 25px; flex-wrap: wrap; justify-content: center; }
        .board-container { flex: 0 0 auto; }
        #board { width: 480px; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }
        .panel {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            color: white;
            min-width: 300px;
            max-width: 360px;
        }
        h1 { font-size: 1.4em; margin-bottom: 12px; }
        .badge { background: #e94560; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
        .badge-mcts { background: #0f3460; border: 1px solid #e94560; }
        .mode-btns { display: flex; gap: 5px; flex-wrap: wrap; margin: 12px 0; }
        .mode-btn {
            padding: 8px 12px;
            background: rgba(255,255,255,0.1);
            border: 2px solid transparent;
            color: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .mode-btn.active { border-color: #e94560; background: rgba(233,69,96,0.3); }
        .status { background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin: 12px 0; }
        .status-row { display: flex; justify-content: space-between; padding: 5px 0; }
        .status-label { color: #aaa; }
        .btns { display: flex; flex-direction: column; gap: 8px; margin: 12px 0; }
        button {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .btn-go { background: linear-gradient(135deg, #e94560, #c23a4f); color: white; }
        .btn-stop { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .btn-reset { background: rgba(255,255,255,0.2); color: white; }
        .slider-group { margin: 8px 0; }
        .slider-group label { display: flex; align-items: center; gap: 8px; font-size: 0.9em; }
        .slider-group input { flex: 1; }
        .moves { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; max-height: 100px; overflow-y: auto; }
        .moves h3 { color: #aaa; font-size: 0.85em; margin-bottom: 8px; }
        #moveList { font-family: monospace; font-size: 0.8em; }
        .result { display: none; padding: 12px; border-radius: 8px; text-align: center; margin: 10px 0; font-weight: bold; }
        .result.show { display: block; }
        .result.white { background: rgba(255,255,255,0.3); }
        .result.black { background: rgba(0,0,0,0.5); }
        .result.draw { background: rgba(234,179,8,0.3); }
        .thinking { display: none; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(233,69,96,0.3); border-radius: 6px; margin: 8px 0; }
        .thinking.show { display: flex; }
        .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .value-bar { height: 8px; background: #333; border-radius: 4px; margin-top: 8px; overflow: hidden; }
        .value-fill { height: 100%; transition: width 0.3s; }
        .mcts-info { background: rgba(233,69,96,0.1); border: 1px solid rgba(233,69,96,0.3); border-radius: 8px; padding: 10px; margin: 10px 0; font-size: 0.8em; }
        .mcts-info h4 { color: #e94560; margin-bottom: 5px; }
        .mcts-moves { max-height: 80px; overflow-y: auto; }
        .mcts-move { display: flex; justify-content: space-between; padding: 2px 0; }
        .mcts-move-name { font-family: monospace; }
        .mcts-move-stats { color: #aaa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="board-container">
            <div id="board"></div>
        </div>
        <div class="panel">
            <h1>♟️ Chess AI v14 <span class="badge badge-mcts">MCTS</span></h1>
            
            <div class="mode-btns">
                <button class="mode-btn active" onclick="setMode('human')">👤 vs AI</button>
                <button class="mode-btn" onclick="setMode('ai')">🤖 vs 🤖</button>
            </div>
            
            <div class="thinking" id="thinking">
                <div class="spinner"></div>
                <span>MCTS searching...</span>
            </div>
            
            <div class="result" id="result"></div>
            
            <div class="status">
                <div class="status-row">
                    <span class="status-label">Mode</span>
                    <span id="modeText">Human vs AI</span>
                </div>
                <div class="status-row">
                    <span class="status-label">Turn</span>
                    <span id="turnText">White</span>
                </div>
                <div class="status-row">
                    <span class="status-label">Move #</span>
                    <span id="moveNum">1</span>
                </div>
                <div class="status-row">
                    <span class="status-label">Eval</span>
                    <span id="evalText">0.00</span>
                </div>
                <div class="value-bar">
                    <div class="value-fill" id="valueFill" style="width: 50%; background: linear-gradient(90deg, #333, #e94560);"></div>
                </div>
            </div>
            
            <div class="slider-group">
                <label>
                    <span>🔍 MCTS:</span>
                    <input type="range" min="10" max="100" value="30" id="mctsSlider">
                    <span id="mctsVal">30 sims</span>
                </label>
            </div>
            
            <div class="slider-group" id="speedDiv" style="display:none;">
                <label>
                    <span>Speed:</span>
                    <input type="range" min="500" max="3000" value="1500" id="speedSlider">
                    <span id="speedVal">1500ms</span>
                </label>
            </div>
            
            <div class="btns">
                <button class="btn-go" onclick="startGame()">▶️ Start</button>
                <button class="btn-stop" onclick="stopGame()">⏹️ Stop</button>
                <button class="btn-reset" onclick="resetGame()">🔄 Reset</button>
            </div>
            
            <div class="mcts-info" id="mctsInfo" style="display:none;">
                <h4>🎯 MCTS Analysis</h4>
                <div class="mcts-moves" id="mctsMoves"></div>
            </div>
            
            <div class="moves">
                <h3>📜 Moves</h3>
                <div id="moveList"></div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    
    <script>
        let board, game = new Chess();
        let mode = 'human', running = false, moves = [];
        
        function setMode(m) {
            mode = m;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('modeText').textContent = m === 'human' ? 'Human vs AI' : 'AI vs AI';
            document.getElementById('speedDiv').style.display = m === 'human' ? 'none' : 'block';
            resetGame();
        }
        
        function onDragStart(source, piece) {
            if (mode !== 'human' || game.game_over()) return false;
            if (piece.search(/^b/) !== -1 || game.turn() !== 'w') return false;
        }
        
        function onDrop(source, target) {
            let move = game.move({from: source, to: target, promotion: 'q'});
            if (!move) return 'snapback';
            moves.push(move.san);
            updateUI();
            sendMove(source + target);
        }
        
        function onSnapEnd() { board.position(game.fen()); }
        
        function sendMove(uci) {
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({move: uci})
            }).then(r => r.json()).then(d => {
                if (d.success && !d.game_over) getAIMove('white');
                else if (d.game_over) showResult(d.result);
            });
        }
        
        function getAIMove(player) {
            if (game.game_over()) { showResult(game.in_checkmate() ? (game.turn() === 'w' ? '0-1' : '1-0') : '1/2-1/2'); return; }
            if (!running && mode !== 'human') return;
            
            document.getElementById('thinking').classList.add('show');
            let mctsSims = parseInt(document.getElementById('mctsSlider').value);
            
            fetch('/api/ai_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mcts_simulations: mctsSims, player: player})
            }).then(r => r.json()).then(d => {
                document.getElementById('thinking').classList.remove('show');
                if (d.success) {
                    let move = game.move({from: d.move.slice(0,2), to: d.move.slice(2,4), promotion: d.move.length > 4 ? d.move[4] : 'q'});
                    if (move) { moves.push(move.san); board.position(game.fen()); updateUI(); }
                    
                    // Update eval display
                    if (d.value !== undefined) {
                        document.getElementById('evalText').textContent = d.value.toFixed(2);
                        let pct = (d.value + 1) / 2 * 100;
                        document.getElementById('valueFill').style.width = pct + '%';
                    }
                    
                    // Show MCTS info
                    if (d.mcts_info) {
                        showMCTSInfo(d.mcts_info);
                    }
                    
                    if (d.game_over) { showResult(d.result); running = false; }
                    else if (mode === 'ai' && running) {
                        let speed = parseInt(document.getElementById('speedSlider').value);
                        setTimeout(() => getAIMove(game.turn() === 'w' ? 'white' : 'black'), speed);
                    }
                }
            });
        }
        
        function showMCTSInfo(info) {
            let el = document.getElementById('mctsInfo');
            el.style.display = 'block';
            
            let html = '';
            let topMoves = info.moves.slice(0, 5);
            for (let m of topMoves) {
                let valueStr = (m.value >= 0 ? '+' : '') + m.value.toFixed(2);
                html += `<div class="mcts-move">
                    <span class="mcts-move-name">${m.move}</span>
                    <span class="mcts-move-stats">${m.visits} visits (${valueStr})</span>
                </div>`;
            }
            document.getElementById('mctsMoves').innerHTML = html;
        }
        
        function startGame() {
            running = true;
            if (mode === 'ai') getAIMove('white');
        }
        
        function stopGame() { running = false; }
        
        function resetGame() {
            running = false;
            game.reset();
            moves = [];
            board.position('start');
            document.getElementById('result').classList.remove('show');
            document.getElementById('evalText').textContent = '0.00';
            document.getElementById('valueFill').style.width = '50%';
            document.getElementById('mctsInfo').style.display = 'none';
            updateUI();
            fetch('/api/reset', {method: 'POST'});
        }
        
        function updateUI() {
            document.getElementById('turnText').textContent = game.turn() === 'w' ? 'White' : 'Black';
            document.getElementById('moveNum').textContent = Math.ceil(moves.length / 2) + 1;
            let html = '';
            for (let i = 0; i < moves.length; i += 2) {
                html += (Math.floor(i/2)+1) + '. ' + moves[i];
                if (moves[i+1]) html += ' ' + moves[i+1];
                html += ' ';
            }
            document.getElementById('moveList').textContent = html;
        }
        
        function showResult(r) {
            running = false;
            let el = document.getElementById('result');
            el.classList.add('show');
            if (r === '1-0') { el.textContent = '⚪ White Wins!'; el.className = 'result show white'; }
            else if (r === '0-1') { el.textContent = '⚫ Black Wins!'; el.className = 'result show black'; }
            else { el.textContent = '🤝 Draw'; el.className = 'result show draw'; }
        }
        
        document.getElementById('speedSlider').oninput = function() { document.getElementById('speedVal').textContent = this.value + 'ms'; };
        document.getElementById('mctsSlider').oninput = function() { document.getElementById('mctsVal').textContent = this.value + ' sims'; };
        
        board = Chessboard('board', {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        });
        $(window).resize(() => board.resize());
    </script>
</body>
</html>
'''


# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create network
network = ChessNet(in_channels=12, filters=128, blocks=6).to(device)

# Load model
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        network.load_state_dict(state_dict)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("   Using random weights")
else:
    print(f"⚠️ Model not found: {MODEL_PATH}")
    print("   Using random weights")

network.eval()

# Game state
game_state = {'board': chess.Board(), 'move_count': 0}


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/reset', methods=['POST'])
def reset():
    game_state['board'] = chess.Board()
    game_state['move_count'] = 0
    return jsonify({'success': True})


@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    move_uci = data.get('move', '')
    
    try:
        for legal in game_state['board'].legal_moves:
            if legal.uci()[:4] == move_uci[:4]:
                game_state['board'].push(legal)
                game_state['move_count'] += 1
                return jsonify({
                    'success': True,
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
        return jsonify({'success': False, 'error': 'Illegal move'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    if game_state['board'].is_game_over():
        return jsonify({
            'success': True,
            'game_over': True,
            'result': game_state['board'].result()
        })
    
    try:
        data = request.json or {}
        mcts_simulations = data.get('mcts_simulations', MCTS_SIMULATIONS)
        
        # Check opening book first
        if game_state['move_count'] < 8:
            book_move = get_opening_book_move(game_state['board'])
            if book_move:
                move = chess.Move.from_uci(book_move)
                game_state['board'].push(move)
                game_state['move_count'] += 1
                return jsonify({
                    'success': True,
                    'move': move.uci(),
                    'source': 'book',
                    'value': 0.0,
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
        
        # Use MCTS for move selection
        mcts = MCTS(network, simulations=mcts_simulations, cpuct=MCTS_CPUCT)
        best_action, root_value, debug_info = mcts.search(game_state['board'])
        
        move = decode_move(best_action, game_state['board'])
        if move is None:
            # Fallback to random legal move
            move = random.choice(list(game_state['board'].legal_moves))
        
        game_state['board'].push(move)
        game_state['move_count'] += 1
        
        return jsonify({
            'success': True,
            'move': move.uci(),
            'source': 'mcts',
            'value': root_value,
            'mcts_info': debug_info,
            'fen': game_state['board'].fen(),
            'game_over': game_state['board'].is_game_over(),
            'result': game_state['board'].result() if game_state['board'].is_game_over() else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("♟️  Chess AI v14 - WITH MCTS")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"MCTS Default: {MCTS_SIMULATIONS} simulations")
    print("\nOpen browser: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
