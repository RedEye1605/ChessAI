#!/usr/bin/env python3
"""
=============================================================================
Web Interface untuk Bermain Melawan Model Kaggle
=============================================================================
Flask web server untuk bermain catur melawan model yang di-training di Kaggle.

Penggunaan:
    python web_play.py --checkpoint checkpoints/chess_model_best.pt
    
Lalu buka browser: http://localhost:5000

Author: Chess RL Project
=============================================================================
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from typing import Optional


# =============================================================================
# Chess Environment & Network (Same as play_kaggle_model.py)
# =============================================================================

class ChessEnv:
    """Chess Environment dengan 18-channel state encoding."""
    
    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    def __init__(self, max_moves=200):
        self.board = chess.Board()
        self.max_moves = max_moves
        self.move_count = 0
        self._init_move_encoding()
    
    def _init_move_encoding(self):
        self.action_to_move = {}
        self.move_to_action = {}
        
        directions = []
        for d in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            for dist in range(1, 8):
                directions.append((d[0]*dist, d[1]*dist))
        
        for d in [(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)]:
            directions.append(d)
        
        action = 0
        for sq in range(64):
            from_rank, from_file = sq // 8, sq % 8
            
            for dx, dy in directions:
                to_rank = from_rank + dy
                to_file = from_file + dx
                
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move = chess.Move(sq, to_sq)
                    self.action_to_move[action] = move
                    self.move_to_action[move.uci()] = action
                action += 1
            
            if from_rank == 6:
                for dx in [-1, 0, 1]:
                    for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                        to_file = from_file + dx
                        if 0 <= to_file < 8:
                            to_sq = 7 * 8 + to_file
                            move = chess.Move(sq, to_sq, promotion=promo)
                            self.action_to_move[action] = move
                            self.move_to_action[move.uci()] = action
                        action += 1
    
    def encode_state(self) -> np.ndarray:
        state = np.zeros((18, 8, 8), dtype=np.float32)
        
        piece_to_channel = {
            (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
            (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
            (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
            (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11
        }
        
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                rank, file = sq // 8, sq % 8
                ch = piece_to_channel[(piece.piece_type, piece.color)]
                state[ch, rank, file] = 1.0
        
        state[12, :, :] = 1.0 if self.board.turn else 0.0
        state[13, 0, :] = float(self.board.has_kingside_castling_rights(True))
        state[14, 0, :] = float(self.board.has_queenside_castling_rights(True))
        state[15, 0, :] = float(self.board.has_kingside_castling_rights(False))
        state[16, 0, :] = float(self.board.has_queenside_castling_rights(False))
        
        if self.board.ep_square:
            ep_rank, ep_file = self.board.ep_square // 8, self.board.ep_square % 8
            state[17, ep_rank, ep_file] = 1.0
        
        return state
    
    def get_legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(4672, dtype=bool)
        for move in self.board.legal_moves:
            uci = move.uci()
            if uci in self.move_to_action:
                mask[self.move_to_action[uci]] = True
            elif len(uci) == 5 and uci[4] == 'q':
                base_uci = uci[:4]
                if base_uci in self.move_to_action:
                    mask[self.move_to_action[base_uci]] = True
        return mask


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class ChessNetwork(nn.Module):
    def __init__(self, input_channels=18, num_filters=256, num_blocks=12, action_size=4672):
        super().__init__()
        
        self.action_size = action_size
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, use_se=(i % 2 == 0))
            for i in range(num_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(num_filters, 80, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(80)
        self.policy_fc = nn.Linear(80 * 64, action_size)
        
        self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x, legal_mask=None):
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        if legal_mask is not None:
            policy_logits = policy_logits.float()
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_probs, value


# =============================================================================
# HTML Template (Inline for simplicity)
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>♟️ Chess RL - Play Against AI</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            display: flex;
            gap: 30px;
            max-width: 1200px;
            width: 100%;
        }
        
        .board-container {
            flex: 0 0 auto;
        }
        
        #board {
            width: 560px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .info-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            color: white;
            min-width: 300px;
        }
        
        h1 {
            font-size: 1.8em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .status-item:last-child { border-bottom: none; }
        
        .status-label { color: #aaa; }
        .status-value { font-weight: bold; }
        
        .buttons {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        button {
            padding: 12px 20px;
            font-size: 1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .move-history {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .move-history h3 {
            margin-bottom: 10px;
            color: #aaa;
        }
        
        #moves {
            font-family: monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .thinking {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .thinking.active { display: flex; }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .game-result {
            display: none;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .game-result.show { display: block; }
        .game-result.win { background: rgba(34, 197, 94, 0.3); }
        .game-result.lose { background: rgba(239, 68, 68, 0.3); }
        .game-result.draw { background: rgba(234, 179, 8, 0.3); }
        
        .eval-bar {
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .eval-fill {
            height: 100%;
            width: 50%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="board-container">
            <div id="board"></div>
        </div>
        
        <div class="info-panel">
            <h1>♟️ Chess RL</h1>
            
            <div class="thinking" id="thinking">
                <div class="spinner"></div>
                <span>AI sedang berpikir...</span>
            </div>
            
            <div class="game-result" id="gameResult"></div>
            
            <div class="status">
                <div class="status-item">
                    <span class="status-label">Giliran</span>
                    <span class="status-value" id="turn">White</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Move #</span>
                    <span class="status-value" id="moveNum">1</span>
                </div>
                <div class="status-item">
                    <span class="status-label">AI Eval</span>
                    <span class="status-value" id="eval">0.00</span>
                </div>
                <div class="eval-bar">
                    <div class="eval-fill" id="evalBar"></div>
                </div>
            </div>
            
            <div class="buttons">
                <button class="btn-primary" onclick="newGame('white')">🎮 New Game (You: White)</button>
                <button class="btn-primary" onclick="newGame('black')">🎮 New Game (You: Black)</button>
                <button class="btn-secondary" onclick="undoMove()">↩️ Undo Move</button>
                <button class="btn-danger" onclick="resetGame()">🔄 Reset</button>
            </div>
            
            <div class="move-history">
                <h3>📜 Move History</h3>
                <div id="moves"></div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    
    <script>
        let board = null;
        let game = new Chess();
        let playerColor = 'white';
        let moveHistory = [];
        
        function onDragStart(source, piece, position, orientation) {
            if (game.game_over()) return false;
            
            // Only pick up own pieces
            if (playerColor === 'white' && piece.search(/^b/) !== -1) return false;
            if (playerColor === 'black' && piece.search(/^w/) !== -1) return false;
            
            // Only move on player's turn
            if ((game.turn() === 'w' && playerColor !== 'white') ||
                (game.turn() === 'b' && playerColor !== 'black')) return false;
        }
        
        function onDrop(source, target) {
            // Try move
            let move = game.move({
                from: source,
                to: target,
                promotion: 'q'
            });
            
            if (move === null) return 'snapback';
            
            moveHistory.push(move.san);
            updateStatus();
            updateMoveHistory();
            
            // Send to server
            sendMove(source + target);
        }
        
        function onSnapEnd() {
            board.position(game.fen());
        }
        
        function sendMove(moveUci) {
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({move: moveUci})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success && !data.game_over) {
                    // Get AI move
                    getAIMove();
                } else if (data.game_over) {
                    showResult(data.result);
                }
            });
        }
        
        function getAIMove() {
            document.getElementById('thinking').classList.add('active');
            
            fetch('/api/ai_move', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                document.getElementById('thinking').classList.remove('active');
                
                if (data.success) {
                    // Apply AI move
                    let move = game.move({
                        from: data.move.substring(0, 2),
                        to: data.move.substring(2, 4),
                        promotion: data.move.length > 4 ? data.move[4] : 'q'
                    });
                    
                    if (move) {
                        moveHistory.push(move.san);
                        board.position(game.fen());
                        updateStatus();
                        updateMoveHistory();
                        
                        // Update eval
                        let evalValue = data.value || 0;
                        document.getElementById('eval').textContent = evalValue.toFixed(2);
                        let evalPercent = (evalValue + 1) / 2 * 100;
                        document.getElementById('evalBar').style.width = evalPercent + '%';
                    }
                    
                    if (data.game_over) {
                        showResult(data.result);
                    }
                }
            });
        }
        
        function updateStatus() {
            let turn = game.turn() === 'w' ? 'White' : 'Black';
            document.getElementById('turn').textContent = turn;
            document.getElementById('moveNum').textContent = Math.ceil(moveHistory.length / 2) + 1;
        }
        
        function updateMoveHistory() {
            let html = '';
            for (let i = 0; i < moveHistory.length; i += 2) {
                let moveNum = Math.floor(i / 2) + 1;
                html += moveNum + '. ' + moveHistory[i];
                if (moveHistory[i + 1]) {
                    html += ' ' + moveHistory[i + 1];
                }
                html += ' ';
            }
            document.getElementById('moves').textContent = html;
        }
        
        function showResult(result) {
            let resultEl = document.getElementById('gameResult');
            resultEl.classList.add('show');
            
            if (result === '1-0') {
                if (playerColor === 'white') {
                    resultEl.textContent = '🎉 YOU WIN!';
                    resultEl.className = 'game-result show win';
                } else {
                    resultEl.textContent = '😢 AI Wins';
                    resultEl.className = 'game-result show lose';
                }
            } else if (result === '0-1') {
                if (playerColor === 'black') {
                    resultEl.textContent = '🎉 YOU WIN!';
                    resultEl.className = 'game-result show win';
                } else {
                    resultEl.textContent = '😢 AI Wins';
                    resultEl.className = 'game-result show lose';
                }
            } else {
                resultEl.textContent = '🤝 Draw';
                resultEl.className = 'game-result show draw';
            }
        }
        
        function newGame(color) {
            playerColor = color;
            game.reset();
            moveHistory = [];
            
            board.orientation(color);
            board.position('start');
            
            document.getElementById('gameResult').classList.remove('show');
            document.getElementById('eval').textContent = '0.00';
            document.getElementById('evalBar').style.width = '50%';
            
            updateStatus();
            updateMoveHistory();
            
            fetch('/api/reset', {method: 'POST'});
            
            // If playing black, AI moves first
            if (color === 'black') {
                setTimeout(getAIMove, 500);
            }
        }
        
        function undoMove() {
            if (moveHistory.length >= 2) {
                game.undo();
                game.undo();
                moveHistory.pop();
                moveHistory.pop();
                board.position(game.fen());
                updateStatus();
                updateMoveHistory();
                
                fetch('/api/undo', {method: 'POST'});
            }
        }
        
        function resetGame() {
            newGame(playerColor);
        }
        
        // Initialize board
        let config = {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };
        
        board = Chessboard('board', config);
        
        $(window).resize(function() {
            board.resize();
        });
    </script>
</body>
</html>
'''


# =============================================================================
# Flask App
# =============================================================================

def create_app(network, env, device):
    app = Flask(__name__)
    
    game_state = {
        'board': chess.Board(),
        'history': []
    }
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'ok',
            'fen': game_state['board'].fen(),
            'turn': 'white' if game_state['board'].turn else 'black'
        })
    
    @app.route('/api/reset', methods=['POST'])
    def reset():
        game_state['board'] = chess.Board()
        game_state['history'] = []
        env.board = chess.Board()
        return jsonify({'success': True})
    
    @app.route('/api/move', methods=['POST'])
    def make_move():
        data = request.json
        move_uci = data.get('move', '')
        
        try:
            # Handle promotions
            if len(move_uci) == 4:
                move = chess.Move.from_uci(move_uci)
                # Check if this is a promotion
                for legal in game_state['board'].legal_moves:
                    if legal.uci()[:4] == move_uci:
                        move = legal
                        break
            else:
                move = chess.Move.from_uci(move_uci)
            
            if move in game_state['board'].legal_moves:
                game_state['board'].push(move)
                game_state['history'].append(move_uci)
                env.board = game_state['board'].copy()
                
                return jsonify({
                    'success': True,
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
            
            # Try to find matching promotion
            for legal in game_state['board'].legal_moves:
                if legal.uci()[:4] == move_uci[:4]:
                    game_state['board'].push(legal)
                    game_state['history'].append(legal.uci())
                    env.board = game_state['board'].copy()
                    
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
            return jsonify({'success': False, 'error': 'Game over'})
        
        try:
            # Sync env board
            env.board = game_state['board'].copy()
            
            # Get AI move
            state = env.encode_state()
            legal_mask = env.get_legal_action_mask()
            
            network.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(legal_mask).unsqueeze(0).to(device)
                
                log_probs, value = network(state_t, mask_t)
                action = log_probs.argmax(dim=-1).item()
            
            # Decode action to move
            base_move = env.action_to_move.get(action)
            if base_move is None:
                return jsonify({'success': False, 'error': 'Invalid action'})
            
            # Find matching legal move
            actual_move = None
            for legal in game_state['board'].legal_moves:
                if legal.uci()[:4] == base_move.uci()[:4]:
                    actual_move = legal
                    break
            
            if actual_move is None:
                return jsonify({'success': False, 'error': 'No legal move found'})
            
            game_state['board'].push(actual_move)
            game_state['history'].append(actual_move.uci())
            
            return jsonify({
                'success': True,
                'move': actual_move.uci(),
                'fen': game_state['board'].fen(),
                'value': value.item(),
                'game_over': game_state['board'].is_game_over(),
                'result': game_state['board'].result() if game_state['board'].is_game_over() else None
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/undo', methods=['POST'])
    def undo():
        if len(game_state['history']) >= 2:
            game_state['board'].pop()
            game_state['board'].pop()
            game_state['history'] = game_state['history'][:-2]
            env.board = game_state['board'].copy()
        return jsonify({'success': True})
    
    return app


def load_model(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
        return None
    
    print(f"📂 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    network = ChessNetwork(
        input_channels=config.get('input_channels', 18),
        num_filters=config.get('num_filters', 256),
        num_blocks=config.get('num_blocks', 12),
        action_size=4672
    )
    
    network.load_state_dict(checkpoint['network_state_dict'])
    network = network.to(device)
    network.eval()
    
    best_win_rate = checkpoint.get('best_win_rate', 0)
    print(f"✅ Model loaded!")
    print(f"   Best win rate: {best_win_rate:.1%}")
    print(f"   Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    return network


def main():
    parser = argparse.ArgumentParser(description='Web interface for Chess RL')
    parser.add_argument('--checkpoint', '-c', default='checkpoints/chess_model_best.pt')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Device: {device}")
    
    # Load model
    network = load_model(args.checkpoint, device)
    if network is None:
        return
    
    # Create env and app
    env = ChessEnv()
    app = create_app(network, env, device)
    
    print(f"\n🎮 Chess RL Web Interface")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"\n   Buka URL di atas di browser untuk bermain!\n")
    print("=" * 50)
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
