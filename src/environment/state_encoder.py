"""
State Encoder untuk Chess Environment
======================================
Modul ini bertanggung jawab untuk mengkonversi board state dari python-chess
ke representasi tensor yang dapat diproses oleh neural network.

Representasi State:
- 14 channels total (14 x 8 x 8)
- Channels 0-5: Pieces putih (Pawn, Knight, Bishop, Rook, Queen, King)
- Channels 6-11: Pieces hitam (Pawn, Knight, Bishop, Rook, Queen, King)  
- Channel 12: Turn indicator (1 jika giliran putih)
- Channel 13: Castling rights & en passant (encoded)
"""

import numpy as np
import chess
from typing import Optional


class StateEncoder:
    """
    Encoder untuk mengkonversi chess.Board ke tensor representation.
    
    Attributes:
        piece_to_channel (dict): Mapping dari piece type ke channel index
    """
    
    # Mapping piece type ke channel offset
    PIECE_CHANNELS = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Jumlah total channels
    NUM_CHANNELS = 14
    
    def __init__(self):
        """Inisialisasi StateEncoder."""
        pass
    
    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encode chess board ke tensor representation.
        
        Args:
            board: chess.Board object
            
        Returns:
            np.ndarray: Tensor shape (14, 8, 8) dengan dtype float32
        """
        # Inisialisasi tensor kosong
        state = np.zeros((self.NUM_CHANNELS, 8, 8), dtype=np.float32)
        
        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Dapatkan row dan column
                row = square // 8
                col = square % 8
                
                # Dapatkan channel berdasarkan piece type dan color
                channel = self.PIECE_CHANNELS[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6  # Black pieces di channels 6-11
                
                state[channel, row, col] = 1.0
        
        # Encode turn (channel 12)
        if board.turn == chess.WHITE:
            state[12, :, :] = 1.0
        
        # Encode castling rights dan en passant (channel 13)
        state[13] = self._encode_auxiliary(board)
        
        return state
    
    def _encode_auxiliary(self, board: chess.Board) -> np.ndarray:
        """
        Encode auxiliary information (castling, en passant).
        
        Args:
            board: chess.Board object
            
        Returns:
            np.ndarray: Plane 8x8 dengan auxiliary info
        """
        aux = np.zeros((8, 8), dtype=np.float32)
        
        # Encode castling rights di corner squares
        if board.has_kingside_castling_rights(chess.WHITE):
            aux[0, 7] = 1.0  # h1
        if board.has_queenside_castling_rights(chess.WHITE):
            aux[0, 0] = 1.0  # a1
        if board.has_kingside_castling_rights(chess.BLACK):
            aux[7, 7] = 1.0  # h8
        if board.has_queenside_castling_rights(chess.BLACK):
            aux[7, 0] = 1.0  # a8
        
        # Encode en passant square
        if board.ep_square is not None:
            row = board.ep_square // 8
            col = board.ep_square % 8
            aux[row, col] = 0.5  # Nilai berbeda untuk membedakan dari castling
        
        return aux
    
    def encode_batch(self, boards: list) -> np.ndarray:
        """
        Encode batch of boards.
        
        Args:
            boards: List of chess.Board objects
            
        Returns:
            np.ndarray: Tensor shape (batch_size, 14, 8, 8)
        """
        batch_size = len(boards)
        states = np.zeros((batch_size, self.NUM_CHANNELS, 8, 8), dtype=np.float32)
        
        for i, board in enumerate(boards):
            states[i] = self.encode(board)
        
        return states
    
    def flip_perspective(self, state: np.ndarray) -> np.ndarray:
        """
        Flip state untuk perspektif pemain yang berbeda.
        Berguna untuk self-play dari kedua sisi.
        
        Args:
            state: Tensor shape (14, 8, 8)
            
        Returns:
            np.ndarray: Flipped state
        """
        flipped = np.zeros_like(state)
        
        # Flip piece channels (swap white dan black)
        flipped[0:6] = np.flip(state[6:12], axis=(1, 2))  # Black -> White position
        flipped[6:12] = np.flip(state[0:6], axis=(1, 2))  # White -> Black position
        
        # Flip turn
        flipped[12] = 1.0 - state[12]
        
        # Flip auxiliary dengan rotasi
        flipped[13] = np.flip(state[13], axis=(0, 1))
        
        return flipped


class StateEncoderExtended(StateEncoder):
    """
    Extended state encoder dengan additional features.
    Menambahkan channels untuk:
    - Attack maps
    - Mobility
    - Piece values
    - Move history (optional)
    """
    
    NUM_CHANNELS = 22  # Extended channels
    
    def __init__(self, include_attacks: bool = True, include_history: bool = False):
        """
        Inisialisasi extended encoder.
        
        Args:
            include_attacks: Include attack/defense maps
            include_history: Include move history planes
        """
        super().__init__()
        self.include_attacks = include_attacks
        self.include_history = include_history
        self.move_history = []
    
    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encode dengan extended features.
        
        Args:
            board: chess.Board object
            
        Returns:
            np.ndarray: Extended tensor representation
        """
        # Base encoding (14 channels)
        base_state = super().encode(board)
        
        if not self.include_attacks:
            return base_state
        
        # Extended state
        state = np.zeros((self.NUM_CHANNELS, 8, 8), dtype=np.float32)
        state[:14] = base_state
        
        # Attack maps (channels 14-15)
        state[14] = self._encode_attacks(board, chess.WHITE)
        state[15] = self._encode_attacks(board, chess.BLACK)
        
        # Defended pieces (channels 16-17)
        state[16] = self._encode_defended(board, chess.WHITE)
        state[17] = self._encode_defended(board, chess.BLACK)
        
        # Legal moves map (channels 18-19)
        state[18] = self._encode_legal_moves_from(board)
        state[19] = self._encode_legal_moves_to(board)
        
        # Piece values heatmap (channels 20-21)
        state[20] = self._encode_piece_values(board, chess.WHITE)
        state[21] = self._encode_piece_values(board, chess.BLACK)
        
        return state
    
    def _encode_attacks(self, board: chess.Board, color: chess.Color) -> np.ndarray:
        """Encode squares yang diserang oleh color."""
        attacks = np.zeros((8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            if board.is_attacked_by(color, square):
                row = square // 8
                col = square % 8
                # Normalisasi berdasarkan jumlah attackers
                attacks[row, col] = min(len(board.attackers(color, square)) / 4.0, 1.0)
        
        return attacks
    
    def _encode_defended(self, board: chess.Board, color: chess.Color) -> np.ndarray:
        """Encode pieces yang defended."""
        defended = np.zeros((8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == color:
                if board.is_attacked_by(color, square):
                    row = square // 8
                    col = square % 8
                    defended[row, col] = 1.0
        
        return defended
    
    def _encode_legal_moves_from(self, board: chess.Board) -> np.ndarray:
        """Encode squares yang bisa melakukan legal moves."""
        moves_from = np.zeros((8, 8), dtype=np.float32)
        
        for move in board.legal_moves:
            row = move.from_square // 8
            col = move.from_square % 8
            moves_from[row, col] += 0.125  # Max ~8 moves per piece
        
        return np.clip(moves_from, 0, 1)
    
    def _encode_legal_moves_to(self, board: chess.Board) -> np.ndarray:
        """Encode target squares dari legal moves."""
        moves_to = np.zeros((8, 8), dtype=np.float32)
        
        for move in board.legal_moves:
            row = move.to_square // 8
            col = move.to_square % 8
            moves_to[row, col] += 0.125
        
        return np.clip(moves_to, 0, 1)
    
    def _encode_piece_values(self, board: chess.Board, color: chess.Color) -> np.ndarray:
        """Encode piece values sebagai heatmap."""
        # Standard piece values
        PIECE_VALUES = {
            chess.PAWN: 0.1,
            chess.KNIGHT: 0.3,
            chess.BISHOP: 0.3,
            chess.ROOK: 0.5,
            chess.QUEEN: 0.9,
            chess.KING: 1.0
        }
        
        values = np.zeros((8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == color:
                row = square // 8
                col = square % 8
                values[row, col] = PIECE_VALUES[piece.piece_type]
        
        return values
