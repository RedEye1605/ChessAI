#!/usr/bin/env python3
"""
=============================================================================
Web Interface for Chess AI v9 - With Opening Book & Temperature
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
from flask import Flask, render_template_string, jsonify, request
from typing import Optional, Dict


# =============================================================================
# Opening Book
# =============================================================================

OPENING_BOOK = {
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -': ['d7d5', 'g8f6', 'e7e6'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['g1f3', 'f1c4', 'b1c3'],
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -': ['b8c6', 'g8f6'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -': ['f1b5', 'f1c4', 'd2d4'],
    'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -': ['f8c5', 'g8f6'],
    'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': ['c2c4', 'g1f3'],
    'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -': ['e7e6', 'c7c6'],
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['g1f3', 'b1c3', 'c2c3'],
}

def get_opening_book_move(board: chess.Board) -> Optional[str]:
    fen_key = ' '.join(board.fen().split()[:4])
    if fen_key in OPENING_BOOK:
        moves = OPENING_BOOK[fen_key]
        legal = [m for m in moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal:
            return random.choice(legal)
    return None


# =============================================================================
# State Encoder
# =============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    state = np.zeros((8, 8, 8), dtype=np.float32)
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = piece_to_channel[piece.piece_type]
            rank, file = sq // 8, sq % 8
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


# =============================================================================
# Neural Network
# =============================================================================

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
    def __init__(self, in_ch=8, filters=128, blocks=6, actions=NUM_ACTIONS):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, actions)
        )
        # Value head (optional - only if model has it)
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
        self.has_value_head = True
    
    def forward(self, x, mask=None):
        x = self.res_blocks(self.input_conv(x))
        p = self.policy_head(x)
        if mask is not None:
            p = p.masked_fill(~mask, -1e4)
        if self.has_value_head:
            v = self.value_head(x)
            return p, v
        return p
    
    def predict(self, state, mask, temperature=0.5):
        """Predict with temperature for variety."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            result = self(x, m)
            if isinstance(result, tuple):
                logits, value = result
                value = value.item()
            else:
                logits = result
                value = 0.0
            
            if temperature <= 0.05:
                probs = torch.zeros_like(logits)
                probs[0, logits.argmax()] = 1.0
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
            
            return probs.squeeze(0).cpu().numpy(), value


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>♟️ Chess AI v9</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
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
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            color: white;
            min-width: 280px;
            max-width: 340px;
        }
        h1 { font-size: 1.4em; margin-bottom: 12px; }
        .badge { background: #6366f1; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
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
        .mode-btn.active { border-color: #6366f1; background: rgba(99,102,241,0.3); }
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
        .btn-go { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .btn-stop { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .btn-reset { background: rgba(255,255,255,0.2); color: white; }
        .speed { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
        .speed input { flex: 1; }
        .temp { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
        .temp input { flex: 1; }
        .moves { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; max-height: 100px; overflow-y: auto; }
        .moves h3 { color: #aaa; font-size: 0.85em; margin-bottom: 8px; }
        #moveList { font-family: monospace; font-size: 0.8em; }
        .result { display: none; padding: 12px; border-radius: 8px; text-align: center; margin: 10px 0; font-weight: bold; }
        .result.show { display: block; }
        .result.white { background: rgba(255,255,255,0.3); }
        .result.black { background: rgba(0,0,0,0.5); }
        .result.draw { background: rgba(234,179,8,0.3); }
        .thinking { display: none; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(99,102,241,0.3); border-radius: 6px; margin: 8px 0; }
        .thinking.show { display: flex; }
        .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="board-container">
            <div id="board"></div>
        </div>
        <div class="panel">
            <h1>♟️ Chess AI v9 <span class="badge">+ Opening Book</span></h1>
            
            <div class="mode-btns">
                <button class="mode-btn active" onclick="setMode('human')">👤 vs AI</button>
                <button class="mode-btn" onclick="setMode('ai')">🤖 vs 🤖</button>
            </div>
            
            <div class="thinking" id="thinking">
                <div class="spinner"></div>
                <span>AI berpikir...</span>
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
            </div>
            
            <div class="speed" id="speedDiv" style="display:none;">
                <span>Speed:</span>
                <input type="range" min="200" max="2000" value="800" id="speedSlider">
                <span id="speedVal">800ms</span>
            </div>
            
            <div class="temp">
                <span>AI Variety:</span>
                <input type="range" min="1" max="100" value="30" id="tempSlider">
                <span id="tempVal">0.30</span>
            </div>
            
            <div class="btns">
                <button class="btn-go" onclick="startGame()">▶️ Start</button>
                <button class="btn-stop" onclick="stopGame()">⏹️ Stop</button>
                <button class="btn-reset" onclick="resetGame()">🔄 Reset</button>
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
            let temp = parseInt(document.getElementById('tempSlider').value) / 100;
            
            fetch('/api/ai_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({temperature: temp, player: player})
            }).then(r => r.json()).then(d => {
                document.getElementById('thinking').classList.remove('show');
                if (d.success) {
                    let move = game.move({from: d.move.slice(0,2), to: d.move.slice(2,4), promotion: d.move.length > 4 ? d.move[4] : 'q'});
                    if (move) { moves.push(move.san); board.position(game.fen()); updateUI(); }
                    if (d.game_over) { showResult(d.result); running = false; }
                    else if (mode === 'ai' && running) {
                        let speed = parseInt(document.getElementById('speedSlider').value);
                        setTimeout(() => getAIMove(game.turn() === 'w' ? 'white' : 'black'), speed);
                    }
                }
            });
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
        document.getElementById('tempSlider').oninput = function() { document.getElementById('tempVal').textContent = (this.value/100).toFixed(2); };
        
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

# Load model - try v9 first, then v7
MODEL_PATHS = [
    r'C:\Users\Rhendy Saragih\Downloads\chess\checkpoints\chess_v9_final.pt',
    r'C:\Users\Rhendy Saragih\Downloads\chess\checkpoints\chess_v9_supervised.pt',
    r'C:\Users\Rhendy Saragih\Downloads\chess\checkpoints\chess_lichess_v7.pt',
]

network = ChessNet(in_ch=8, filters=128, blocks=6).to(device)

for path in MODEL_PATHS:
    import os
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        try:
            state_dict = torch.load(path, map_location=device)
            # Handle both policy-only and policy+value models
            model_dict = network.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered)
            network.load_state_dict(model_dict, strict=False)
            print(f"✅ Loaded {len(filtered)} layers")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
else:
    print("⚠️ No model found, using random weights")

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
        temperature = data.get('temperature', 0.3)
        
        # Check opening book first
        if game_state['move_count'] < 10:
            book_move = get_opening_book_move(game_state['board'])
            if book_move:
                move = chess.Move.from_uci(book_move)
                game_state['board'].push(move)
                game_state['move_count'] += 1
                return jsonify({
                    'success': True,
                    'move': move.uci(),
                    'source': 'book',
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
        
        # Use neural network
        state = encode_board(game_state['board'])
        mask = get_legal_mask(game_state['board'])
        probs, value = network.predict(state, mask, temperature=temperature)
        
        # Sample from distribution
        if temperature > 0.05:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = int(np.argmax(probs))
        
        move = decode_move(action, game_state['board'])
        if move is None:
            move = random.choice(list(game_state['board'].legal_moves))
        
        game_state['board'].push(move)
        game_state['move_count'] += 1
        
        return jsonify({
            'success': True,
            'move': move.uci(),
            'source': 'network',
            'value': value,
            'fen': game_state['board'].fen(),
            'game_over': game_state['board'].is_game_over(),
            'result': game_state['board'].result() if game_state['board'].is_game_over() else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("♟️  Chess AI v9 - With Opening Book")
    print("=" * 50)
    print("\nOpen browser: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
