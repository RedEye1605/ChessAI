"""
Action Space untuk Chess Environment
=====================================
Modul ini menangani encoding dan decoding gerakan catur ke/dari
representasi numerik yang dapat digunakan oleh neural network.

Pendekatan:
- Menggunakan encoding berbasis (from_square, to_square, promotion)
- Total action space: 64 * 73 = 4672 possible moves
  - 64 from squares
  - 73 possible destinations (8 directions x 7 max distance + 8 knight moves + 9 underpromotions)
"""

import numpy as np
import chess
from typing import List, Tuple, Optional, Dict


class ActionSpace:
    """
    Mengelola action space untuk chess moves.
    
    Action encoding:
    - Actions direpresentasikan sebagai integer 0-4671
    - Encoding: action = from_square * 73 + move_type
    - move_type mencakup arah, jarak, dan promosi
    """
    
    # Ukuran action space
    ACTION_SIZE = 4672  # 64 * 73
    
    # Move type indices (0-72)
    # Queen-like moves: 8 directions x 7 distances = 56
    # Knight moves: 8 moves = 8
    # Underpromotions: 3 piece types x 3 directions = 9
    
    # Direction vectors untuk queen-like moves
    QUEEN_DIRECTIONS = [
        (0, 1),   # Right
        (1, 1),   # Up-right
        (1, 0),   # Up
        (1, -1),  # Up-left
        (0, -1),  # Left
        (-1, -1), # Down-left
        (-1, 0),  # Down
        (-1, 1),  # Down-right
    ]
    
    # Knight move offsets
    KNIGHT_MOVES = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    
    # Underpromotion pieces
    UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    
    def __init__(self):
        """Inisialisasi action space dan lookup tables."""
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Membangun lookup tables untuk encoding/decoding yang cepat."""
        # Move to action lookup
        self.move_to_action: Dict[chess.Move, int] = {}
        # Action to move lookup
        self.action_to_move: Dict[int, chess.Move] = {}
        
        for from_square in range(64):
            from_row = from_square // 8
            from_col = from_square % 8
            
            move_type = 0
            
            # Queen-like moves (termasuk promosi ke queen)
            for direction_idx, (dr, dc) in enumerate(self.QUEEN_DIRECTIONS):
                for distance in range(1, 8):
                    to_row = from_row + dr * distance
                    to_col = from_col + dc * distance
                    
                    if 0 <= to_row < 8 and 0 <= to_col < 8:
                        to_square = to_row * 8 + to_col
                        action = from_square * 73 + move_type
                        
                        # Normal move
                        move = chess.Move(from_square, to_square)
                        self.move_to_action[move] = action
                        self.action_to_move[action] = move
                        
                        # Promosi ke queen (jika pawn reach last rank)
                        if (to_row == 7 or to_row == 0):
                            promo_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                            self.move_to_action[promo_move] = action
                    
                    move_type += 1
                    if move_type % 7 == 0:
                        break
                
                # Reset untuk direction berikutnya
                move_type = (direction_idx + 1) * 7
            
            # Knight moves
            move_type = 56
            for knight_idx, (dr, dc) in enumerate(self.KNIGHT_MOVES):
                to_row = from_row + dr
                to_col = from_col + dc
                
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_square = to_row * 8 + to_col
                    action = from_square * 73 + move_type
                    
                    move = chess.Move(from_square, to_square)
                    self.move_to_action[move] = action
                    self.action_to_move[action] = move
                
                move_type += 1
            
            # Underpromotions (hanya dari rank 7 untuk white atau rank 2 untuk black)
            move_type = 64
            if from_row == 6 or from_row == 1:  # Pawn pada second-to-last rank
                for promo_piece in self.UNDERPROMOTIONS:
                    # 3 directions: left-capture, forward, right-capture
                    for dc in [-1, 0, 1]:
                        if from_row == 6:  # White pawn going up
                            to_row = 7
                        else:  # Black pawn going down
                            to_row = 0
                        to_col = from_col + dc
                        
                        if 0 <= to_col < 8:
                            to_square = to_row * 8 + to_col
                            action = from_square * 73 + move_type
                            
                            promo_move = chess.Move(from_square, to_square, promotion=promo_piece)
                            self.move_to_action[promo_move] = action
                            self.action_to_move[action] = promo_move
                        
                        move_type += 1
    
    def encode_move(self, move: chess.Move) -> int:
        """
        Encode chess move ke action index.
        
        Args:
            move: chess.Move object
            
        Returns:
            int: Action index (0-4671)
            
        Raises:
            ValueError: Jika move tidak valid
        """
        if move in self.move_to_action:
            return self.move_to_action[move]
        
        # Fallback: encode manually
        return self._encode_move_manual(move)
    
    def _encode_move_manual(self, move: chess.Move) -> int:
        """Manual encoding untuk moves yang tidak ada di lookup table."""
        from_square = move.from_square
        to_square = move.to_square
        
        from_row, from_col = from_square // 8, from_square % 8
        to_row, to_col = to_square // 8, to_square % 8
        
        dr = to_row - from_row
        dc = to_col - from_col
        
        # Check untuk knight move
        if (abs(dr), abs(dc)) in [(2, 1), (1, 2)]:
            for idx, (knight_dr, knight_dc) in enumerate(self.KNIGHT_MOVES):
                if dr == knight_dr and dc == knight_dc:
                    return from_square * 73 + 56 + idx
        
        # Check untuk queen-like move
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            # Normalisasi direction
            dir_r = 0 if dr == 0 else dr // abs(dr)
            dir_c = 0 if dc == 0 else dc // abs(dc)
            
            for dir_idx, (d_r, d_c) in enumerate(self.QUEEN_DIRECTIONS):
                if d_r == dir_r and d_c == dir_c:
                    distance = max(abs(dr), abs(dc))
                    move_type = dir_idx * 7 + (distance - 1)
                    return from_square * 73 + move_type
        
        raise ValueError(f"Cannot encode move: {move}")
    
    def decode_action(self, action: int) -> chess.Move:
        """
        Decode action index ke chess move.
        
        Args:
            action: Action index (0-4671)
            
        Returns:
            chess.Move object
            
        Raises:
            ValueError: Jika action tidak valid
        """
        if action in self.action_to_move:
            return self.action_to_move[action]
        
        raise ValueError(f"Invalid action: {action}")
    
    def get_legal_action_mask(self, board: chess.Board) -> np.ndarray:
        """
        Generate mask untuk legal moves.
        
        Args:
            board: chess.Board object
            
        Returns:
            np.ndarray: Boolean mask shape (4672,) dimana True = legal move
        """
        mask = np.zeros(self.ACTION_SIZE, dtype=np.bool_)
        
        for move in board.legal_moves:
            try:
                action = self.encode_move(move)
                mask[action] = True
            except ValueError:
                # Skip moves yang tidak bisa di-encode
                pass
        
        return mask
    
    def get_legal_actions(self, board: chess.Board) -> List[int]:
        """
        Get list of legal action indices.
        
        Args:
            board: chess.Board object
            
        Returns:
            List[int]: List of legal action indices
        """
        return [self.encode_move(move) for move in board.legal_moves 
                if move in self.move_to_action]
    
    def sample_legal_action(self, board: chess.Board) -> int:
        """
        Sample random legal action.
        
        Args:
            board: chess.Board object
            
        Returns:
            int: Random legal action index
        """
        legal_actions = self.get_legal_actions(board)
        return np.random.choice(legal_actions)
    
    def action_to_uci(self, action: int) -> str:
        """
        Convert action ke UCI notation.
        
        Args:
            action: Action index
            
        Returns:
            str: UCI move string (e.g., "e2e4")
        """
        move = self.decode_action(action)
        return move.uci()
    
    def uci_to_action(self, uci: str) -> int:
        """
        Convert UCI notation ke action.
        
        Args:
            uci: UCI move string
            
        Returns:
            int: Action index
        """
        move = chess.Move.from_uci(uci)
        return self.encode_move(move)


class ActionSpaceSimple:
    """
    Versi sederhana dari action space menggunakan direct move mapping.
    Lebih straightforward tapi dengan action space yang lebih besar.
    
    Encoding: from_square * 64 + to_square (+ promotion offset)
    Total: 64 * 64 * 5 = 20480 (dengan 5 promotion options)
    """
    
    ACTION_SIZE = 20480
    
    def __init__(self):
        """Inisialisasi simple action space."""
        pass
    
    def encode_move(self, move: chess.Move) -> int:
        """Encode move dengan simple mapping."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Base action
        action = from_sq * 64 + to_sq
        
        # Promotion offset
        if move.promotion:
            promo_offset = {
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK: 3,
                chess.QUEEN: 4
            }
            action = 4096 + from_sq * 64 + to_sq + (promo_offset[move.promotion] - 1) * 4096
        
        return action
    
    def decode_action(self, action: int) -> chess.Move:
        """Decode action ke move."""
        if action < 4096:
            from_sq = action // 64
            to_sq = action % 64
            return chess.Move(from_sq, to_sq)
        else:
            # Promotion move
            promo_idx = (action - 4096) // 4096
            remainder = (action - 4096) % 4096
            from_sq = remainder // 64
            to_sq = remainder % 64
            
            promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
            return chess.Move(from_sq, to_sq, promotion=promo_pieces[promo_idx])
    
    def get_legal_action_mask(self, board: chess.Board) -> np.ndarray:
        """Generate mask untuk legal moves."""
        mask = np.zeros(self.ACTION_SIZE, dtype=np.bool_)
        
        for move in board.legal_moves:
            action = self.encode_move(move)
            mask[action] = True
        
        return mask
