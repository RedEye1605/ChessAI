
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Optional

# =============================================================================
# Configuration & Constants
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

NUM_ACTIONS = 64 * 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Helper Functions (Encoding/Decoding)
# =============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
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
# Opening Book
# =============================================================================

OPENING_BOOK = {
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPPP1PPP/RNBQKBNR b KQkq': ['d7d5', 'g8f6', 'e7e6'],
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
# Neural Network Architecture
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
            policy = policy.masked_fill(~mask, -1e4) # Updated to -1e4
        
        return policy, value
    
    def predict(self, state: np.ndarray, mask: np.ndarray, temperature: float = 0.5):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(next(self.parameters()).device)
            
            logits, value = self(x, m)
            
            if temperature <= 0.05:
                probs = torch.zeros_like(logits)
                probs[0, logits.argmax()] = 1.0
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
            
            return probs.squeeze(0).cpu().numpy(), value.item()

# =============================================================================
# Model Initialization
# =============================================================================

MODEL_PATH = "model.pt" # Simplified path for deployment
network = ChessNet(in_channels=12, filters=128, blocks=6).to(DEVICE)

# Note: The model file must be placed in the same directory for deployment
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Load with map_location to ensure CPU usage if CUDA not available on HF
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}. Prediction will use random weights.")

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/')
def home():
    return "Chess AI Backend is Running. Use /api/ai_move to get moves."

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    try:
        data = request.json or {}
        
        # Input Validation
        fen = data.get('fen')
        if not fen:
            return jsonify({'success': False, 'error': 'FEN string required'}), 400
            
        temperature = data.get('temperature', 0.3)
        
        # Initialize board from FEN
        board = chess.Board(fen)
        
        if board.is_game_over():
             return jsonify({
                'success': True,
                'game_over': True,
                'result': board.result()
            })
            
        # Check Opening Book (First 10 moves approx)
        if board.fullmove_number <= 7: 
            book_move = get_opening_book_move(board)
            if book_move:
                move = chess.Move.from_uci(book_move)
                board.push(move)
                return jsonify({
                    'success': True,
                    'move': move.uci(),
                    'source': 'book',
                    'value': 0.0,
                    'fen': board.fen(),
                    'game_over': board.is_game_over(),
                    'result': board.result() if board.is_game_over() else None
                })
        
        # Neural Network Prediction
        state = encode_board(board)
        mask = get_legal_mask(board)
        
        probs, value = network.predict(state, mask, temperature=temperature)
        
        if temperature > 0.05:
            # Add small noise for variety if strictly following probability
            probs = probs / probs.sum()
            action = np.random.choice(len(probs), p=probs)
        else:
            action = int(np.argmax(probs))
        
        move = decode_move(action, board)
        
        # Fallback if neural network predicts illegal move (should be handled by mask, but safety net)
        if move is None:
             move = random.choice(list(board.legal_moves))
        
        # Apply move to get new FEN
        board.push(move)
        
        return jsonify({
            'success': True,
            'move': move.uci(),
            'source': 'network',
            'value': value,
            'fen': board.fen(),
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Use port 7860 for Hugging Face Spaces compatibility implicitly
    app.run(host='0.0.0.0', port=7860)
