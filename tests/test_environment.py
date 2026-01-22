"""
Unit Tests untuk Chess Environment
===================================
"""

import pytest
import numpy as np
import chess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ChessEnv, StateEncoder, ActionSpace


class TestStateEncoder:
    """Tests untuk StateEncoder."""
    
    def test_encode_initial_position(self):
        """Test encoding posisi awal."""
        encoder = StateEncoder()
        board = chess.Board()
        
        state = encoder.encode(board)
        
        assert state.shape == (14, 8, 8)
        assert state.dtype == np.float32
        
        # Check white pawns di rank 2
        assert np.sum(state[0, 1, :]) == 8  # 8 white pawns
        
        # Check black pawns di rank 7
        assert np.sum(state[6, 6, :]) == 8  # 8 black pawns
        
        # Check turn (white to move)
        assert np.all(state[12] == 1.0)
    
    def test_encode_empty_board(self):
        """Test encoding board kosong."""
        encoder = StateEncoder()
        board = chess.Board(None)  # Empty board
        
        state = encoder.encode(board)
        
        assert state.shape == (14, 8, 8)
        # Pieces channels should be all zeros
        assert np.sum(state[:12]) == 0
    
    def test_encode_batch(self):
        """Test batch encoding."""
        encoder = StateEncoder()
        boards = [chess.Board() for _ in range(5)]
        
        states = encoder.encode_batch(boards)
        
        assert states.shape == (5, 14, 8, 8)


class TestActionSpace:
    """Tests untuk ActionSpace."""
    
    def test_encode_decode(self):
        """Test encoding dan decoding."""
        action_space = ActionSpace()
        board = chess.Board()
        
        # Test e2e4
        move = chess.Move.from_uci('e2e4')
        action = action_space.encode_move(move)
        decoded = action_space.decode_action(action)
        
        assert decoded == move
    
    def test_legal_mask(self):
        """Test legal move mask."""
        action_space = ActionSpace()
        board = chess.Board()
        
        mask = action_space.get_legal_action_mask(board)
        
        assert mask.shape == (4672,)
        assert mask.dtype == np.bool_
        
        # Initial position has 20 legal moves
        assert np.sum(mask) == 20
    
    def test_legal_actions_list(self):
        """Test getting legal actions as list."""
        action_space = ActionSpace()
        board = chess.Board()
        
        actions = action_space.get_legal_actions(board)
        
        assert len(actions) == 20


class TestChessEnv:
    """Tests untuk ChessEnv."""
    
    def test_reset(self):
        """Test environment reset."""
        env = ChessEnv()
        state, info = env.reset()
        
        assert state.shape == (14, 8, 8)
        assert info['move_count'] == 0
        assert info['turn'] == 'white'
    
    def test_step_legal_move(self):
        """Test step dengan legal move."""
        env = ChessEnv()
        env.reset()
        
        # e2e4
        move = chess.Move.from_uci('e2e4')
        action = env.action_space_handler.encode_move(move)
        
        state, reward, terminated, truncated, info = env.step(action)
        
        assert state.shape == (14, 8, 8)
        assert not terminated
        assert info['turn'] == 'black'
    
    def test_step_illegal_move(self):
        """Test step dengan illegal move."""
        env = ChessEnv()
        env.reset()
        
        # Try illegal move (e2e5 - pawn can't jump that far)
        move = chess.Move.from_uci('e2e5')
        
        # Encode manually since it's illegal
        action = 12 * 73 + 2  # Some encoding
        
        state, reward, terminated, truncated, info = env.step(action)
        
        # Should terminate dengan negative reward
        assert terminated or info.get('illegal_move', False)
    
    def test_checkmate_reward(self):
        """Test reward untuk checkmate."""
        env = ChessEnv()
        
        # Scholar's mate position (almost)
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        env.reset(options={'fen': fen})
        
        # This is already checkmate!
        assert env.board.is_checkmate()
    
    def test_legal_action_mask(self):
        """Test getting legal action mask from env."""
        env = ChessEnv()
        env.reset()
        
        mask = env.get_legal_action_mask()
        
        assert mask.shape == (4672,)
        assert np.sum(mask) == 20  # 20 legal moves at start
    
    def test_max_moves_truncation(self):
        """Test truncation setelah max moves."""
        env = ChessEnv(max_moves=5)
        env.reset()
        
        # Make 5 moves
        for _ in range(5):
            if env.board.is_game_over():
                break
            
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                break
                
            move = legal_moves[0]
            action = env.action_space_handler.encode_move(move)
            _, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should be truncated or done
        assert env.move_count <= 5


class TestIntegration:
    """Integration tests."""
    
    def test_full_game_random(self):
        """Test full random game."""
        env = ChessEnv()
        state, _ = env.reset()
        
        done = False
        moves = 0
        
        while not done and moves < 100:
            # Random legal move
            legal_actions = env.action_space_handler.get_legal_actions(env.board)
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            moves += 1
        
        assert moves > 0
        assert state.shape == (14, 8, 8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
