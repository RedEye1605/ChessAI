#!/usr/bin/env python3
"""
=============================================================================
Web Interface for Chess AI v27 - Premium Design
=============================================================================
Beautiful, modern UI with glassmorphism and smooth animations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
import os
from flask import Flask, render_template_string, jsonify, request
from typing import Optional


# =============================================================================
# State Encoder (12 Channels)
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
            policy = policy.masked_fill(~mask, -1e4)
        
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
# HTML Template - PREMIUM DESIGN
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>♟️ Chess AI v27</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.8);
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.1);
            --glow: rgba(99, 102, 241, 0.4);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(30, 41, 59, 1) 0%, var(--bg-dark) 100%);
            min-height: 100vh;
            color: var(--text);
            overflow-x: hidden;
        }
        
        /* Animated background particles */
        .bg-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
        }
        
        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--primary);
            border-radius: 50%;
            opacity: 0.3;
            animation: float 20s infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            10% { opacity: 0.3; }
            90% { opacity: 0.3; }
            100% { transform: translateY(-100vh) rotate(720deg); opacity: 0; }
        }
        
        .main-container {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 20px;
            gap: 30px;
        }
        
        /* Header */
        .header {
            text-align: center;
            animation: slideDown 0.6s ease-out;
        }
        
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .logo {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, var(--primary) 50%, var(--success) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 400;
        }
        
        .version-badge {
            display: inline-block;
            background: linear-gradient(135deg, var(--primary), var(--success));
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 10px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 var(--glow); }
            50% { box-shadow: 0 0 20px 5px var(--glow); }
        }
        
        /* Game Area */
        .game-area {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
            animation: fadeIn 0.8s ease-out 0.2s backwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Board Container */
        .board-wrapper {
            position: relative;
        }
        
        .board-glow {
            position: absolute;
            top: -20px;
            left: -20px;
            right: -20px;
            bottom: -20px;
            background: linear-gradient(135deg, var(--primary), var(--success));
            border-radius: 20px;
            opacity: 0.2;
            filter: blur(30px);
            z-index: -1;
        }
        
        #board {
            width: 480px !important;
            height: 480px !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                0 0 0 1px var(--border);
        }
        
        /* Panel */
        .panel {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            width: 320px;
            border: 1px solid var(--border);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }
        
        .panel-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, var(--primary), var(--success));
            border-radius: 2px;
        }
        
        /* Mode Buttons */
        .mode-btns {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
        }
        
        .mode-btn {
            flex: 1;
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid transparent;
            color: var(--text-muted);
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s ease;
            font-family: inherit;
        }
        
        .mode-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text);
        }
        
        .mode-btn.active {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.15);
            color: var(--text);
        }
        
        /* Status Cards */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .status-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 14px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .status-card:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-2px);
        }
        
        .status-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        
        .status-value {
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        .status-value.turn-white { color: #fff; }
        .status-value.turn-black { color: var(--text-muted); }
        
        /* Eval Bar */
        .eval-container {
            margin-bottom: 20px;
        }
        
        .eval-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .eval-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .eval-value {
            font-size: 1rem;
            font-weight: 600;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        
        .eval-bar {
            height: 10px;
            background: linear-gradient(90deg, #1e293b 0%, #1e293b 100%);
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }
        
        .eval-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease, background 0.5s ease;
            background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
        }
        
        /* Thinking Indicator - Fixed height to prevent layout shift */
        .thinking {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 10px;
            margin-bottom: 20px;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.2s ease;
            height: 52px;
        }
        
        .thinking.show { 
            visibility: visible; 
            opacity: 1; 
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(99, 102, 241, 0.3);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .thinking-text {
            font-size: 0.85rem;
            color: var(--primary);
        }
        
        /* Result Banner */
        .result {
            display: none;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 20px;
            animation: popIn 0.4s ease-out;
        }
        
        @keyframes popIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .result.show { display: block; }
        .result.white { background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.3); }
        .result.black { background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); }
        .result.draw { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); }
        
        /* Sliders */
        .slider-group {
            margin-bottom: 16px;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.8rem;
        }
        
        .slider-label span:first-child {
            color: var(--text-muted);
        }
        
        .slider-label span:last-child {
            font-weight: 600;
            color: var(--primary);
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, var(--primary), var(--success));
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 10px var(--glow);
            transition: transform 0.2s;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }
        
        /* Buttons */
        .btn-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .btn {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            font-family: inherit;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--danger), #dc2626);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Move History */
        .moves-container {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 14px;
            max-height: 120px;
            overflow-y: auto;
        }
        
        .moves-header {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        
        #moveList {
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 0.8rem;
            line-height: 1.6;
            color: var(--text-muted);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
        
        /* Responsive */
        @media (max-width: 900px) {
            .game-area { flex-direction: column; align-items: center; }
            #board { width: 360px !important; height: 360px !important; }
            .panel { width: 360px; }
        }
        
        @media (max-width: 400px) {
            #board { width: 320px !important; height: 320px !important; }
            .panel { width: 320px; padding: 16px; }
        }
    </style>
</head>
<body>
    <!-- Background Particles -->
    <div class="bg-particles">
        <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
        <div class="particle" style="left: 20%; animation-delay: 2s;"></div>
        <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
        <div class="particle" style="left: 40%; animation-delay: 6s;"></div>
        <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
        <div class="particle" style="left: 60%; animation-delay: 10s;"></div>
        <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
        <div class="particle" style="left: 80%; animation-delay: 14s;"></div>
        <div class="particle" style="left: 90%; animation-delay: 16s;"></div>
    </div>
    
    <div class="main-container">
        <!-- Header -->
        <div class="header">
            <div class="logo">♟️ Chess AI</div>
            <div class="subtitle">Neural Network powered by Stockfish-Guided Learning</div>
            <div class="version-badge">v27 • SF-Trained</div>
        </div>
        
        <!-- Game Area -->
        <div class="game-area">
            <!-- Board -->
            <div class="board-wrapper">
                <div class="board-glow"></div>
                <div id="board"></div>
            </div>
            
            <!-- Control Panel -->
            <div class="panel">
                <div class="panel-title">Game Controls</div>
                
                <!-- Mode Buttons -->
                <div class="mode-btns">
                    <button class="mode-btn active" onclick="setMode('human')">👤 Play vs AI</button>
                    <button class="mode-btn" onclick="setMode('ai')">🤖 AI vs AI</button>
                </div>
                
                <!-- Thinking Indicator -->
                <div class="thinking" id="thinking">
                    <div class="spinner"></div>
                    <span class="thinking-text">AI is thinking...</span>
                </div>
                
                <!-- Result Banner -->
                <div class="result" id="result"></div>
                
                <!-- Status Grid -->
                <div class="status-grid">
                    <div class="status-card">
                        <div class="status-label">Turn</div>
                        <div class="status-value turn-white" id="turnText">White</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">Move</div>
                        <div class="status-value" id="moveNum">1</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">Mode</div>
                        <div class="status-value" id="modeText" style="font-size: 0.9rem;">Human</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">Status</div>
                        <div class="status-value" id="statusText" style="font-size: 0.9rem; color: var(--success);">Ready</div>
                    </div>
                </div>
                
                <!-- Eval Bar -->
                <div class="eval-container">
                    <div class="eval-header">
                        <span class="eval-label">AI Evaluation</span>
                        <span class="eval-value" id="evalText">0.00</span>
                    </div>
                    <div class="eval-bar">
                        <div class="eval-fill" id="valueFill" style="width: 50%;"></div>
                    </div>
                </div>
                
                <!-- Speed Slider (AI vs AI) -->
                <div class="slider-group" id="speedDiv" style="display:none;">
                    <div class="slider-label">
                        <span>AI Speed</span>
                        <span id="speedVal">800ms</span>
                    </div>
                    <input type="range" min="200" max="2000" value="800" id="speedSlider">
                </div>
                
                <!-- Temperature Slider -->
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Move Variety</span>
                        <span id="tempVal">0.30</span>
                    </div>
                    <input type="range" min="1" max="100" value="30" id="tempSlider">
                </div>
                
                <!-- Buttons -->
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="startGame()">▶ Start</button>
                    <button class="btn btn-danger" onclick="stopGame()">⏹ Stop</button>
                    <button class="btn btn-secondary" onclick="resetGame()">↻ Reset</button>
                </div>
                
                <!-- Move History -->
                <div class="moves-container">
                    <div class="moves-header">📜 Move History</div>
                    <div id="moveList">1. e4 e5 2. Nf3 Nc6 ...</div>
                </div>
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
            document.getElementById('modeText').textContent = m === 'human' ? 'Human' : 'AI vs AI';
            document.getElementById('speedDiv').style.display = m === 'human' ? 'none' : 'block';
            resetGame();
        }
        
        function onDragStart(source, piece) {
            if (mode !== 'human' || game.game_over()) return false;
            // Allow dragging only own pieces on own turn
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
                return false;
            }
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
                if (d.success && !d.game_over) getAIMove('black');
                else if (d.game_over) showResult(d.result);
            });
        }
        
        function getAIMove(player) {
            if (game.game_over()) { 
                showResult(game.in_checkmate() ? (game.turn() === 'w' ? '0-1' : '1-0') : '1/2-1/2'); 
                return; 
            }
            if (!running && mode !== 'human') return;
            
            document.getElementById('thinking').classList.add('show');
            document.getElementById('statusText').textContent = 'Thinking...';
            document.getElementById('statusText').style.color = 'var(--primary)';
            
            let temp = parseInt(document.getElementById('tempSlider').value) / 100;
            
            fetch('/api/ai_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({temperature: temp, player: player})
            }).then(r => r.json()).then(d => {
                document.getElementById('thinking').classList.remove('show');
                document.getElementById('statusText').textContent = 'Ready';
                document.getElementById('statusText').style.color = 'var(--success)';
                
                if (d.success) {
                    let move = game.move({from: d.move.slice(0,2), to: d.move.slice(2,4), promotion: d.move.length > 4 ? d.move[4] : 'q'});
                    if (move) { 
                        moves.push(move.san); 
                        board.position(game.fen(), true); 
                        updateUI(); 
                    }
                    
                    if (d.value !== undefined) {
                        document.getElementById('evalText').textContent = d.value.toFixed(2);
                        let pct = (d.value + 1) / 2 * 100;
                        document.getElementById('valueFill').style.width = pct + '%';
                    }
                    
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
            document.getElementById('statusText').textContent = 'Playing';
            document.getElementById('statusText').style.color = 'var(--success)';
            if (mode === 'ai') getAIMove('white');
        }
        
        function stopGame() { 
            running = false; 
            document.getElementById('statusText').textContent = 'Paused';
            document.getElementById('statusText').style.color = 'var(--warning)';
        }
        
        function resetGame() {
            running = false;
            game.reset();
            moves = [];
            board.position('start', false);
            document.getElementById('result').classList.remove('show');
            document.getElementById('result').className = 'result';
            document.getElementById('evalText').textContent = '0.00';
            document.getElementById('valueFill').style.width = '50%';
            document.getElementById('statusText').textContent = 'Ready';
            document.getElementById('statusText').style.color = 'var(--success)';
            document.getElementById('moveList').textContent = '';
            updateUI();
            fetch('/api/reset', {method: 'POST'});
        }
        
        function updateUI() {
            let turnEl = document.getElementById('turnText');
            turnEl.textContent = game.turn() === 'w' ? 'White' : 'Black';
            turnEl.className = game.turn() === 'w' ? 'status-value turn-white' : 'status-value turn-black';
            document.getElementById('moveNum').textContent = Math.ceil(moves.length / 2) || 1;
            
            let html = '';
            for (let i = 0; i < moves.length; i += 2) {
                html += (Math.floor(i/2)+1) + '. ' + moves[i];
                if (moves[i+1]) html += ' ' + moves[i+1];
                html += '  ';
            }
            document.getElementById('moveList').textContent = html || 'Game not started';
        }
        
        function showResult(r) {
            running = false;
            let el = document.getElementById('result');
            el.classList.add('show');
            document.getElementById('statusText').textContent = 'Game Over';
            document.getElementById('statusText').style.color = 'var(--text-muted)';
            
            if (r === '1-0') { 
                el.textContent = '👑 White Wins!'; 
                el.className = 'result show white'; 
            }
            else if (r === '0-1') { 
                el.textContent = '👑 Black Wins!'; 
                el.className = 'result show black'; 
            }
            else { 
                el.textContent = '🤝 Draw!'; 
                el.className = 'result show draw'; 
            }
        }
        
        document.getElementById('speedSlider').oninput = function() { 
            document.getElementById('speedVal').textContent = this.value + 'ms'; 
        };
        document.getElementById('tempSlider').oninput = function() { 
            document.getElementById('tempVal').textContent = (this.value/100).toFixed(2); 
        };
        
        board = Chessboard('board', {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
            moveSpeed: 'slow',
            snapbackSpeed: 'slow',
            snapSpeed: 50,
            appearSpeed: 'slow',
            trashSpeed: 'slow'
        });
        
        $(window).resize(() => board.resize());
        updateUI();
    </script>
</body>
</html>
'''



# =============================================================================
# Web Interface (Premium UI/UX v2)
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess AI v27 - Neural Nebula</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- CSS Dependencies -->
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    
    <style>
        :root {
            /* Palette: Deep Space Nebula */
            --bg-deep: #030712;
            --bg-surface: #0f172a;
            --primary: #818cf8;       /* Indigo 400 */
            --primary-glow: rgba(129, 140, 248, 0.5);
            --secondary: #2dd4bf;     /* Teal 400 */
            --accent: #f472b6;        /* Pink 400 */
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            
            /* Status Colors */
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            
            /* Glassmorphism */
            --glass-bg: rgba(15, 23, 42, 0.65);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-highlight: rgba(255, 255, 255, 0.03);
            --blur-strength: 24px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-deep);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(79, 70, 229, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(45, 212, 191, 0.15), transparent 25%);
        }

        /* SVG Icons */
        .icon { width: 20px; height: 20px; fill: currentColor; }
        .icon-lg { width: 24px; height: 24px; }
        .icon-sm { width: 16px; height: 16px; }

        /* Background Particles */
        .bg-particles {
            position: fixed;
            top: 0; left: 0; w: 100%; h: 100%;
            pointer-events: none;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, transparent 100%);
            border-radius: 50%;
            width: 2px; height: 2px;
            bottom: -10px;
            animation: floatUp 20s linear infinite;
            opacity: 0.3;
        }
        
        @keyframes floatUp { 
            0% { transform: translateY(0) scale(1); opacity: 0; }
            10% { opacity: 0.5; }
            90% { opacity: 0.2; }
            100% { transform: translateY(-110vh) scale(0.5); opacity: 0; }
        }

        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--glass-border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 40px; height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px var(--primary-glow);
        }
        
        .brand-title h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #fff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-title span {
            font-size: 0.8rem;
            color: var(--primary);
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        
        .model-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--glass-highlight);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Game Area */
        .game-layout {
            display: flex;
            justify-content: center;
            gap: 40px;
            align-items: flex-start;
            flex-wrap: wrap;
        }

        /* Board */
        .board-container {
            position: relative;
            padding: 10px;
            background: rgba(255,255,255,0.02);
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
        }

        #board {
            width: 480px !important;
            height: 480px !important;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1);
        }

        /* Sidebar Control Panel */
        .control-panel {
            width: 360px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(var(--blur-strength));
            -webkit-backdrop-filter: blur(var(--blur-strength));
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px -5px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            box-shadow: 0 15px 35px -5px rgba(0,0,0,0.4);
            border-color: rgba(255,255,255,0.15);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        /* Mode Selection */
        .mode-toggle {
            display: grid;
            grid-template-columns: 1fr 1fr;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 20px;
        }

        .mode-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 10px;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .mode-btn.active {
            background: var(--primary);
            color: #fff;
            box-shadow: 0 2px 10px rgba(129, 140, 248, 0.4);
        }

        .mode-btn:hover:not(.active) {
            color: var(--text-main);
            background: rgba(255,255,255,0.05);
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .stat-box {
            background: rgba(255,255,255,0.03);
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .turn-indicator {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        /* Eval Bar */
        .eval-wrapper {
            margin-bottom: 24px;
        }
        
        .eval-info {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .eval-track {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .eval-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
            width: 50%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .eval-fill::after {
            content: '';
            position: absolute;
            right: 0; top: 0; bottom: 0;
            width: 2px;
            background: #fff;
            box-shadow: 0 0 10px #fff;
        }

        /* Action Buttons */
        .actions {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-top: auto;
        }

        .btn {
            border: none;
            padding: 12px;
            border-radius: 10px;
            cursor: pointer;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #6366f1);
            color: white;
            grid-column: span 1;
        }

        .btn-danger {
            background: rgba(248, 113, 113, 0.1);
            color: var(--danger);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            border: 1px solid var(--glass-border);
        }

        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .btn-primary:hover { box-shadow: 0 4px 15px var(--primary-glow); }
        .btn-danger:hover { background: rgba(248, 113, 113, 0.2); }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.1); }

        /* Sliders */
        .slider-control { margin-bottom: 16px; }
        
        .slider-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        input[type="range"] {
            width: 100%;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            appearance: none;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 16px; height: 16px;
            border-radius: 50%;
            background: var(--text-main);
            border: 2px solid var(--primary);
            cursor: pointer;
            transition: transform 0.1s;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }

        /* Thinking Indicator */
        .thinking-overlay {
            position: absolute;
            top: 20px; right: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--primary);
            padding: 10px 20px;
            border-radius: 30px;
            backdrop-filter: blur(10px);
            opacity: 0;
            transform: translateY(-10px);
            transition: all 0.3s ease;
            pointer-events: none;
            z-index: 10;
        }
        
        .thinking-overlay.active { opacity: 1; transform: translateY(0); }
        
        .pulse-dot {
            width: 8px; height: 8px;
            background: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--primary);
            animation: pulse 1s infinite alternate;
        }
        
        @keyframes pulse { from { opacity: 0.4; transform: scale(0.8); } to { opacity: 1; transform: scale(1.2); } }

        /* Move List */
        .move-history {
            max-height: 150px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.8;
            color: var(--text-muted);
            padding-right: 8px;
        }
        
        .move-row {
            display: flex;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding: 4px 0;
        }
        
        .move-num { width: 40px; color: var(--text-muted); opacity: 0.5; }
        .move-white { width: 80px; color: var(--text-main); }
        .move-black { width: 80px; color: var(--text-main); }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

        /* Responsive */
        @media (max-width: 900px) {
            .game-layout { flex-direction: column; align-items: center; }
            .control-panel { width: 100%; max-width: 480px; }
        }
    </style>
</head>
<body>
    <div class="bg-particles">
        <!-- SVG generated particles via JS or simple divs -->
        <div class="particle" style="left: 10%; width: 2px;"></div>
        <div class="particle" style="left: 30%; width: 3px; animation-delay: 2s;"></div>
        <div class="particle" style="left: 70%; width: 2px; animation-delay: 5s;"></div>
        <div class="particle" style="left: 90%; width: 4px; animation-delay: 3s;"></div>
    </div>

    <div class="main-container">
        <header class="header">
            <div class="brand">
                <div class="brand-icon">
                    <svg class="icon icon-lg" viewBox="0 0 24 24" style="fill:white;">
                        <path d="M12 2c5.523 0 10 4.477 10 10s-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2zm0 2a8 8 0 100 16 8 8 0 000-16z"/>
                        <circle cx="12" cy="8" r="3"/>
                        <path d="M7 16h10v2H7z"/>
                    </svg>
                </div>
                <div class="brand-title">
                    <h1>Chess AI</h1>
                    <span>Neural Network</span>
                </div>
            </div>
            <div class="model-badge">
                <svg class="icon icon-sm" viewBox="0 0 24 24">
                    <path d="M9 3v1H4v2h5v1h6V6h5V4h-5V3H9zm6 18v-1h5v-2h-5v-1H9v1H4v2h5v1h6z"/>
                    <path d="M5 8h14v8H5z" opacity=".3"/>
                </svg>
                v27 • Nebula
            </div>
        </header>

        <main class="game-layout">
            <!-- Board Section -->
            <div class="board-container">
                <div id="board"></div>
                
                <!-- Thinking Overlay inside board area -->
                <div class="thinking-overlay" id="thinkingAlert">
                    <div class="pulse-dot"></div>
                    <span style="font-weight: 500; font-size: 0.9rem;">AI is computing...</span>
                </div>
            </div>

            <!-- Controls Section -->
            <div class="control-panel">
                
                <!-- Status Card -->
                <div class="glass-card">
                    <div class="section-header">
                        <svg class="icon" viewBox="0 0 24 24" style="margin-right: 8px;">
                            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
                        </svg>
                        Game Status
                    </div>
                    
                    <div class="mode-toggle">
                        <button class="mode-btn active" onclick="setMode('human')" id="btn-human">
                            <svg class="icon icon-sm" viewBox="0 0 24 24">
                                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                            </svg>
                            Human
                        </button>
                        <button class="mode-btn" onclick="setMode('ai')" id="btn-ai">
                            <svg class="icon icon-sm" viewBox="0 0 24 24">
                                <path d="M20 9V7c0-1.1-.9-2-2-2h-3c0-1.66-1.34-3-3-3S9 3.34 9 5H6c-1.1 0-2 .9-2 2v2c-1.66 0-3 1.34-3 3s1.34 3 3 3v2c0 1.1.9 2 2 2h3c0 1.66 1.34 3 3 3s3-1.34 3-3h3c1.1 0 2-.9 2-2v-2c1.66 0 3-1.34 3-3s-1.34-3-3-3zm-5.5-2c.83 0 1.5.67 1.5 1.5S15.33 10 14.5 10 13 9.33 13 8.5 13.67 7 14.5 7zM9.5 7c.83 0 1.5.67 1.5 1.5S10.33 10 9.5 10 8 9.33 8 8.5 8.67 7 9.5 7zM12 18c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/>
                            </svg>
                            AI vs AI
                        </button>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Turn</div>
                            <div class="stat-value" id="turnIndicator">
                                <span class="turn-indicator" style="background: #fff;"></span>
                                <span id="turnText">White</span>
                            </div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">State</div>
                            <div class="stat-value" id="statusText" style="color: var(--success); font-size: 0.95rem;">Ready</div>
                        </div>
                    </div>

                    <!-- Eval Bar -->
                    <div class="eval-wrapper">
                        <div class="eval-info">
                            <span>Evaluation</span>
                            <span id="evalValue">+0.00</span>
                        </div>
                        <div class="eval-track">
                            <div class="eval-fill" id="evalBar"></div>
                        </div>
                    </div>
                    
                    <div id="resultBanner" style="display:none; text-align:center; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; margin-bottom: 20px;">
                        <strong id="resultText">Checkmate!</strong>
                    </div>
                </div>

                <!-- Settings & History -->
                <div class="glass-card">
                    <div id="aiSettings">
                        <div class="section-header">
                            <svg class="icon" viewBox="0 0 24 24" style="margin-right: 8px;">
                                <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L6.14 8.79c-.11.21-.06.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.58 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.11-.22.06-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
                            </svg>
                            Config
                        </div>

                        <div class="slider-control">
                            <div class="slider-header">
                                <span>Temperature (Creativity)</span>
                                <span id="tempVal">0.3</span>
                            </div>
                            <input type="range" id="tempSlider" min="0" max="100" value="30">
                        </div>

                        <div class="slider-control" id="speedControl" style="display:none;">
                            <div class="slider-header">
                                <span>AI Move Speed</span>
                                <span id="speedVal">800ms</span>
                            </div>
                            <input type="range" id="speedSlider" min="200" max="2000" value="800" step="100">
                        </div>
                    </div>

                    <div class="section-header" style="margin-top: 20px;">
                        <svg class="icon" viewBox="0 0 24 24" style="margin-right: 8px;">
                            <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                        </svg>
                        Move History
                    </div>
                    <div class="move-history" id="moveConfig">
                        <div id="moveList"></div>
                    </div>

                    <div class="actions" style="margin-top: 20px;">
                        <button class="btn btn-primary" onclick="startGame()">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                            Start
                        </button>
                        <button class="btn btn-danger" onclick="stopGame()">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>
                            Stop
                        </button>
                        <button class="btn btn-secondary" onclick="resetGame()">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
                            Reset
                        </button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>

    <script>
        // Game State
        let board, game = new Chess();
        let mode = 'human';
        let isRunning = false;
        let moves = [];
        
        // Configuration
        const config = {
            animation: {
                moveSpeed: 250,     // Faster sliding (was 'slow' ~600ms)
                snapbackSpeed: 300, // Faster snapback
                snapSpeed: 100, 
                appearSpeed: 400    // Faster appearance
            }
        };

        // --- Board Initialization ---
        function onDragStart(source, piece) {
            if (mode !== 'human' || game.game_over()) return false;
            // Only pick up own pieces
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
                return false;
            }
        }

        function onDrop(source, target) {
            // see if the move is legal
            let move = game.move({
                from: source,
                to: target,
                promotion: 'q' 
            });

            // illegal move
            if (move === null) return 'snapback';

            // Valid move
            moves.push(move.san);
            updateStatus();
            
            // Notify backend
            sendMoveToBackend(source + target);
        }

        function onSnapEnd() {
            // Important for instant updates after potential promotion or messy drops
            // However, rely on internal board state for visual consistency
            board.position(game.fen());
        }

        board = Chessboard('board', {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
            moveSpeed: config.animation.moveSpeed,
            snapbackSpeed: config.animation.snapbackSpeed,
            snapSpeed: config.animation.snapSpeed,
            appearSpeed: config.animation.appearSpeed,
        });

        $(window).resize(() => board.resize());

        // --- Interaction Logic ---
        function setMode(newMode) {
            mode = newMode;
            // Visual Updates
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + newMode).classList.add('active');
            
            document.getElementById('speedControl').style.display = (newMode === 'ai') ? 'block' : 'none';
            resetGame();
        }

        function startGame() {
            if (game.game_over()) return;
            isRunning = true;
            updateStatusText('Game Started', 'var(--success)');
            
            if (mode === 'ai') {
                requestAIMove('white');
            }
        }

        function stopGame() {
            isRunning = false;
            updateStatusText('Paused', 'var(--warning)');
        }

        function resetGame() {
            isRunning = false;
            game.reset();
            moves = [];
            board.position('start', true); // Animate back to start
            
            // UI Reset
            document.getElementById('resultBanner').style.display = 'none';
            document.getElementById('moveList').innerHTML = '';
            document.getElementById('evalValue').innerText = '+0.00';
            document.getElementById('evalBar').style.width = '50%';
            updateStatus();
            updateStatusText('Ready', 'var(--success)');
            
            // Backend Reset
            fetch('/api/reset', { method: 'POST' });
        }

        // --- Networking ---
        function sendMoveToBackend(uciMove) {
            checkThinking(true);
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ move: uciMove })
            })
            .then(res => res.json())
            .then(data => {
                checkThinking(false);
                if (data.success && !data.game_over) {
                    if (mode === 'human') {
                        // Trigger AI response
                        setTimeout(() => requestAIMove('black'), 200);
                    }
                } else if (data.game_over) {
                    handleGameOver(data.result);
                }
            });
        }

        function requestAIMove(playerColor) {
            if (game.game_over() || (!isRunning && mode === 'ai')) return;
            
            checkThinking(true);
            const temp = parseInt(document.getElementById('tempSlider').value) / 100;
            
            fetch('/api/ai_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ temperature: temp, player: playerColor })
            })
            .then(res => res.json())
            .then(data => {
                checkThinking(false);
                
                if (data.success) {
                    // Update internal JS game state first
                    // UCI format e.g. "e2e4" -> from: "e2", to: "e4"
                    const fromSq = data.move.substring(0, 2);
                    const toSq = data.move.substring(2, 4);
                    const promotion = data.move.length > 4 ? data.move.substring(4) : undefined;
                    
                    const moveObj = game.move({ from: fromSq, to: toSq, promotion: promotion || 'q' });
                    
                    if (moveObj) {
                        moves.push(moveObj.san);
                        
                        // Handle Castling Animations Explicitly
                        let moveCmd = fromSq + '-' + toSq;
                        if (moveObj.flags.includes('k')) { // Kingside
                            moveCmd = (game.turn() === 'b' ? 'e1-g1' : 'e8-g8') + '-' + (game.turn() === 'b' ? 'h1-f1' : 'h8-f8'); // Note: Turn is swapped because move already pushed to 'game' but we need visual move for the color that JUST moved
                        } else if (moveObj.flags.includes('q')) { // Queenside
                            moveCmd = (game.turn() === 'b' ? 'e1-c1' : 'e8-c8') + '-' + (game.turn() === 'b' ? 'a1-d1' : 'a8-d8');
                        }
                        
                        // Use basic move check for standard moves if not constructed manually above
                         if (!moveCmd.includes('-')) moveCmd = fromSq + '-' + toSq;
                         
                        // Correct logic for castling detection based on squares since flags might be tricky with just 'game' state
                        // Actually, simplified: just trust board.move for the main piece, and if it's castling, we add the rook move.
                        // Better approach: Check the SAN or the move object details.
                        
                        let visualMoves = [fromSq + '-' + toSq];
                        if (moveObj.flags.includes('k') || moveObj.flags.includes('q')) {
                            // Logic: calculate rook squares.
                            // White Kingside: e1-g1, h1-f1
                            if (fromSq === 'e1' && toSq === 'g1') visualMoves.push('h1-f1');
                            else if (fromSq === 'e1' && toSq === 'c1') visualMoves.push('a1-d1');
                            else if (fromSq === 'e8' && toSq === 'g8') visualMoves.push('h8-f8');
                            else if (fromSq === 'e8' && toSq === 'c8') visualMoves.push('a8-d8');
                        }

                        board.move(...visualMoves);
                        
                        // CRITICAL FIX: Sync board state after animation to handle En Passant / Promotion / Castling final states guaranteed
                        // Wait for animation (moveSpeed) + small buffer
                        setTimeout(() => {
                            board.position(game.fen(), false); // Snap to exact state (removes captured pieces, fixes rooks if missed)
                        }, config.animation.moveSpeed + 50);
                        
                        updateStatus();
                        updateEval(data.value);
                    }

                    if (data.game_over) {
                        handleGameOver(data.result);
                    } else if (mode === 'ai' && isRunning) {
                        const delay = parseInt(document.getElementById('speedSlider').value);
                        setTimeout(() => {
                            requestAIMove(game.turn() === 'w' ? 'white' : 'black');
                        }, delay);
                    }
                }
            });
        }

        // --- UI Updates ---
        function updateStatus() {
            const turnText = document.getElementById('turnText');
            const turnIndicator = document.querySelector('.turn-indicator');
            
            if (game.turn() === 'w') {
                turnText.innerText = 'White';
                turnIndicator.style.background = '#fff';
                turnIndicator.style.boxShadow = '0 0 10px rgba(255,255,255,0.8)';
            } else {
                turnText.innerText = 'Black';
                turnIndicator.style.background = '#64748b'; // Slate 500
                turnIndicator.style.boxShadow = 'none';
            }
            
            // Move History
            renderHistory();
        }

        function renderHistory() {
            const list = document.getElementById('moveList');
            let html = '';
            for (let i = 0; i < moves.length; i += 2) {
                const num = (i / 2) + 1;
                const wMove = moves[i];
                const bMove = moves[i+1] || '';
                
                html += `
                <div class="move-row">
                    <span class="move-num">${num}.</span>
                    <span class="move-white">${wMove}</span>
                    <span class="move-black">${bMove}</span>
                </div>`;
            }
            list.innerHTML = html;
            list.scrollTop = list.scrollHeight;
        }

        function updateEval(val) {
            if (val === undefined) return;
            
            const raw = parseFloat(val);
            const displayVal = (raw > 0 ? '+' : '') + raw.toFixed(2);
            document.getElementById('evalValue').innerText = displayVal;
            
            // Normalize -1 to 1 range -> 0% to 100%
            // visual clamp standard: -3 to +3 mostly
            let clampled = Math.max(-3, Math.min(3, raw));
            // map -3 -> 0, 3 -> 100
            let pct = ((clampled + 3) / 6) * 100;
            document.getElementById('evalBar').style.width = `${pct}%`;
        }

        function updateStatusText(text, color) {
            const el = document.getElementById('statusText');
            el.innerText = text;
            el.style.color = color;
        }

        function checkThinking(isThinking) {
            const el = document.getElementById('thinkingAlert');
            if (isThinking) el.classList.add('active');
            else el.classList.remove('active');
        }

        function handleGameOver(result) {
            isRunning = false;
            updateStatusText('Game Over', 'var(--text-muted)');
            
            const banner = document.getElementById('resultBanner');
            const txt = document.getElementById('resultText');
            
            banner.style.display = 'block';
            if (result === '1-0') {
                txt.innerText = 'Result: 1-0 (White Wins)';
                txt.style.color = '#fff';
            } else if (result === '0-1') {
                txt.innerText = 'Result: 0-1 (Black Wins)';
                txt.style.color = 'var(--text-muted)';
            } else {
                txt.innerText = 'Result: 1/2-1/2 (Draw)';
                txt.style.color = 'var(--warning)';
            }
        }
        
        // Input Listeners
        document.getElementById('tempSlider').oninput = function() {
            document.getElementById('tempVal').innerText = (this.value / 100).toFixed(2);
        };
        document.getElementById('speedSlider').oninput = function() {
            document.getElementById('speedVal').innerText = this.value + 'ms';
        };
    </script>
</body>
</html>
'''


# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = os.path.join('checkpoints', 'chess_v27_final.pt')

network = ChessNet(in_channels=12, filters=128, blocks=6).to(device)

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
        
        # Use opening book for first few moves
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
                    'value': 0.0,
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
        
        state = encode_board(game_state['board'])
        mask = get_legal_mask(game_state['board'])
        probs, value = network.predict(state, mask, temperature=temperature)
        
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
    print("\n" + "=" * 60)
    print("♟️  Chess AI v27 - Premium Web Interface")
    print("=" * 60)
    print("\n🌐 Open browser: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
