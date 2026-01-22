"""
Web Application untuk Visualisasi Chess AI
===========================================
Web interface untuk melihat AI bermain catur secara interaktif.
"""

import os
import json
import chess
import torch
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from typing import Optional, Dict, Any
from pathlib import Path

# Get template and static directories
TEMPLATE_DIR = Path(__file__).parent / 'templates'
STATIC_DIR = Path(__file__).parent / 'static'


def create_app(
    agent=None,
    env=None,
    device=None,
    debug: bool = False
) -> Flask:
    """
    Create Flask app untuk visualisasi.
    
    Args:
        agent: Trained chess agent
        env: Chess environment
        device: Torch device
        debug: Debug mode
        
    Returns:
        Flask application
    """
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR)
    )
    app.config['SECRET_KEY'] = 'chess-rl-secret'
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Game state
    game_state = {
        'board': chess.Board(),
        'agent': agent,
        'env': env,
        'device': device,
        'game_history': [],
        'ai_thinking': False
    }
    
    @app.route('/')
    def index():
        """Main page."""
        return render_template('index.html')
    
    @app.route('/api/status')
    def status():
        """Get server status."""
        return jsonify({
            'status': 'ok',
            'agent_loaded': game_state['agent'] is not None,
            'current_fen': game_state['board'].fen(),
            'turn': 'white' if game_state['board'].turn else 'black'
        })
    
    @app.route('/api/reset', methods=['POST'])
    def reset_game():
        """Reset game."""
        game_state['board'] = chess.Board()
        game_state['game_history'] = []
        
        return jsonify({
            'success': True,
            'fen': game_state['board'].fen()
        })
    
    @app.route('/api/move', methods=['POST'])
    def make_move():
        """Make a move."""
        data = request.json
        move_uci = data.get('move')
        
        try:
            move = chess.Move.from_uci(move_uci)
            
            if move in game_state['board'].legal_moves:
                game_state['board'].push(move)
                game_state['game_history'].append(move_uci)
                
                return jsonify({
                    'success': True,
                    'fen': game_state['board'].fen(),
                    'game_over': game_state['board'].is_game_over(),
                    'result': game_state['board'].result() if game_state['board'].is_game_over() else None
                })
            else:
                return jsonify({'success': False, 'error': 'Illegal move'})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ai_move', methods=['POST'])
    def get_ai_move():
        """Get AI's move."""
        if game_state['agent'] is None:
            return jsonify({'success': False, 'error': 'No agent loaded'})
        
        if game_state['board'].is_game_over():
            return jsonify({'success': False, 'error': 'Game is over'})
        
        try:
            game_state['ai_thinking'] = True
            
            # Encode state
            state = encode_board(game_state['board'])
            legal_mask = get_legal_mask(game_state['board'], game_state['env'])
            
            # Get AI move
            game_state['agent'].eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if game_state['device']:
                    state_tensor = state_tensor.to(game_state['device'])
                
                mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0)
                if game_state['device']:
                    mask_tensor = mask_tensor.to(game_state['device'])
                
                log_probs, value = game_state['agent'](state_tensor, mask_tensor)
                action = log_probs.argmax(dim=-1).item()
                value = value.item()
            
            # Decode action to move
            move = game_state['env'].action_space_handler.decode_action(action)
            
            # Make move
            game_state['board'].push(move)
            game_state['game_history'].append(move.uci())
            
            game_state['ai_thinking'] = False
            
            return jsonify({
                'success': True,
                'move': move.uci(),
                'fen': game_state['board'].fen(),
                'value': value,
                'game_over': game_state['board'].is_game_over(),
                'result': game_state['board'].result() if game_state['board'].is_game_over() else None
            })
        
        except Exception as e:
            game_state['ai_thinking'] = False
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/legal_moves')
    def get_legal_moves():
        """Get legal moves for current position."""
        moves = [move.uci() for move in game_state['board'].legal_moves]
        return jsonify({'moves': moves})
    
    @app.route('/api/game_info')
    def get_game_info():
        """Get current game info."""
        info = {
            'fen': game_state['board'].fen(),
            'turn': 'white' if game_state['board'].turn else 'black',
            'move_number': game_state['board'].fullmove_number,
            'is_check': game_state['board'].is_check(),
            'is_checkmate': game_state['board'].is_checkmate(),
            'is_stalemate': game_state['board'].is_stalemate(),
            'is_game_over': game_state['board'].is_game_over(),
            'result': game_state['board'].result() if game_state['board'].is_game_over() else None,
            'history': game_state['game_history']
        }
        return jsonify(info)
    
    @app.route('/api/ai_vs_ai', methods=['POST'])
    def ai_vs_ai_game():
        """Run AI vs AI game."""
        if game_state['agent'] is None:
            return jsonify({'success': False, 'error': 'No agent loaded'})
        
        game_state['board'] = chess.Board()
        game_state['game_history'] = []
        moves = []
        
        max_moves = 200
        
        while not game_state['board'].is_game_over() and len(moves) < max_moves:
            # Get AI move
            state = encode_board(game_state['board'])
            legal_mask = get_legal_mask(game_state['board'], game_state['env'])
            
            game_state['agent'].eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if game_state['device']:
                    state_tensor = state_tensor.to(game_state['device'])
                
                mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0)
                if game_state['device']:
                    mask_tensor = mask_tensor.to(game_state['device'])
                
                log_probs, _ = game_state['agent'](state_tensor, mask_tensor)
                action = log_probs.argmax(dim=-1).item()
            
            move = game_state['env'].action_space_handler.decode_action(action)
            moves.append({
                'move': move.uci(),
                'fen': game_state['board'].fen()
            })
            
            game_state['board'].push(move)
        
        return jsonify({
            'success': True,
            'moves': moves,
            'final_fen': game_state['board'].fen(),
            'result': game_state['board'].result(),
            'num_moves': len(moves)
        })
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        emit('connected', {'status': 'Connected to Chess RL Server'})
    
    @socketio.on('request_ai_move')
    def handle_ai_move_request():
        """Handle AI move request via WebSocket."""
        if game_state['agent'] is None:
            emit('error', {'message': 'No agent loaded'})
            return
        
        emit('ai_thinking', {'status': True})
        
        try:
            state = encode_board(game_state['board'])
            legal_mask = get_legal_mask(game_state['board'], game_state['env'])
            
            game_state['agent'].eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if game_state['device']:
                    state_tensor = state_tensor.to(game_state['device'])
                
                mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0)
                if game_state['device']:
                    mask_tensor = mask_tensor.to(game_state['device'])
                
                log_probs, value = game_state['agent'](state_tensor, mask_tensor)
                action = log_probs.argmax(dim=-1).item()
            
            move = game_state['env'].action_space_handler.decode_action(action)
            game_state['board'].push(move)
            
            emit('ai_move', {
                'move': move.uci(),
                'fen': game_state['board'].fen(),
                'value': value.item(),
                'game_over': game_state['board'].is_game_over()
            })
        
        except Exception as e:
            emit('error', {'message': str(e)})
        
        finally:
            emit('ai_thinking', {'status': False})
    
    return app


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board ke tensor representation."""
    state = np.zeros((14, 8, 8), dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            channel = piece_map[piece.piece_type]
            if not piece.color:
                channel += 6
            state[channel, row, col] = 1.0
    
    if board.turn == chess.WHITE:
        state[12, :, :] = 1.0
    
    return state


def get_legal_mask(board: chess.Board, env) -> np.ndarray:
    """Get legal action mask."""
    if env is not None:
        return env.action_space_handler.get_legal_action_mask(board)
    
    # Fallback
    mask = np.zeros(4672, dtype=bool)
    for move in board.legal_moves:
        # Simple encoding
        from_sq = move.from_square
        to_sq = move.to_square
        action = from_sq * 73  # Simplified
        if action < 4672:
            mask[action] = True
    return mask


def run_server(
    agent=None,
    env=None,
    device=None,
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False
):
    """
    Run visualization server.
    
    Args:
        agent: Trained agent
        env: Chess environment
        device: Torch device
        host: Host address
        port: Port number
        debug: Debug mode
    """
    app = create_app(agent, env, device, debug)
    socketio = SocketIO(app)
    
    print(f"\n🎮 Chess RL Visualization Server")
    print(f"   URL: http://{host}:{port}")
    print(f"   Agent loaded: {agent is not None}")
    print(f"\n   Buka browser dan kunjungi URL di atas untuk bermain!\n")
    
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
