"""
Chess Environment untuk Reinforcement Learning
===============================================
Environment catur yang kompatibel dengan Gymnasium interface.
Dirancang untuk training agen RL dengan berbagai fitur:
- State representation yang kaya
- Reward shaping untuk pembelajaran yang lebih baik
- Legal move masking
- Self-play support
"""

import numpy as np
import chess
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from .state_encoder import StateEncoder, StateEncoderExtended
from .action_space import ActionSpace


class ChessEnv(gym.Env):
    """
    Gymnasium-compatible Chess Environment.
    
    Observation Space:
        Box(0, 1, shape=(14, 8, 8), dtype=float32)
        14 channels encoding board state
        
    Action Space:
        Discrete(4672)
        Encoded chess moves
        
    Rewards:
        +1.0  : Menang (checkmate lawan)
        -1.0  : Kalah (di-checkmate)
        +0.5  : Draw dengan material advantage
        0.0   : Draw
        -0.5  : Draw dengan material disadvantage
        +0.01 : Capture piece (scaled by piece value)
        -0.1  : Illegal move attempt
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi']}
    
    # Piece values untuk reward shaping
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King tidak bisa di-capture
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_moves: int = 200,
        reward_config: Optional[Dict[str, float]] = None,
        use_extended_state: bool = False
    ):
        """
        Inisialisasi Chess Environment.
        
        Args:
            render_mode: Mode rendering ('human', 'rgb_array', 'ansi')
            max_moves: Maksimum moves per game (untuk mencegah infinite games)
            reward_config: Custom reward configuration
            use_extended_state: Gunakan extended state encoder (22 channels)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.use_extended_state = use_extended_state
        
        # Inisialisasi encoder dan action space
        if use_extended_state:
            self.state_encoder = StateEncoderExtended()
            self._num_channels = 22
        else:
            self.state_encoder = StateEncoder()
            self._num_channels = 14
        
        self.action_space_handler = ActionSpace()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self._num_channels, 8, 8),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(ActionSpace.ACTION_SIZE)
        
        # Reward configuration
        self.reward_config = {
            'win': 1.0,
            'lose': -1.0,
            'draw': 0.0,
            'draw_advantage': 0.3,
            'draw_disadvantage': -0.3,
            'capture_scale': 0.02,
            'illegal_move': -1.0,
            'check_bonus': 0.01,
            'center_control': 0.005,
            'mobility_bonus': 0.001
        }
        if reward_config:
            self.reward_config.update(reward_config)
        
        # Internal state
        self.board: Optional[chess.Board] = None
        self.move_count: int = 0
        self.episode_rewards: List[float] = []
        
        # Untuk tracking
        self._last_material_balance = 0.0
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment ke state awal.
        
        Args:
            seed: Random seed
            options: Additional options (bisa berisi FEN untuk custom position)
            
        Returns:
            observation: State awal
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Buat board baru atau dari FEN
        if options and 'fen' in options:
            self.board = chess.Board(options['fen'])
        else:
            self.board = chess.Board()
        
        self.move_count = 0
        self.episode_rewards = []
        self._last_material_balance = self._calculate_material_balance()
        
        observation = self.state_encoder.encode(self.board)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Eksekusi satu langkah dalam environment.
        
        Args:
            action: Action index (0-4671)
            
        Returns:
            observation: State setelah action
            reward: Reward yang diterima
            terminated: True jika game selesai
            truncated: True jika game di-truncate (max moves)
            info: Additional information
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Decode action ke move
        try:
            move = self.action_space_handler.decode_action(action)
        except ValueError:
            # Invalid action
            reward = self.reward_config['illegal_move']
            observation = self.state_encoder.encode(self.board)
            return observation, reward, True, False, self._get_info(illegal=True)
        
        # Check apakah move legal
        if move not in self.board.legal_moves:
            reward = self.reward_config['illegal_move']
            observation = self.state_encoder.encode(self.board)
            return observation, reward, True, False, self._get_info(illegal=True)
        
        # Capture reward (sebelum move)
        captured_piece = self.board.piece_at(move.to_square)
        if captured_piece and captured_piece.color != self.board.turn:
            piece_value = self.PIECE_VALUES.get(captured_piece.piece_type, 0)
            reward += piece_value * self.reward_config['capture_scale']
        
        # Execute move
        self.board.push(move)
        self.move_count += 1
        
        # Check game termination
        if self.board.is_checkmate():
            # Current player yang baru saja move menang
            reward += self.reward_config['win']
            terminated = True
        elif self.board.is_stalemate():
            reward += self._get_draw_reward()
            terminated = True
        elif self.board.is_insufficient_material():
            reward += self._get_draw_reward()
            terminated = True
        elif self.board.is_fifty_moves():
            reward += self._get_draw_reward()
            terminated = True
        elif self.board.is_repetition(3):
            reward += self._get_draw_reward()
            terminated = True
        elif self.move_count >= self.max_moves:
            reward += self._get_draw_reward()
            truncated = True
        else:
            # Bonus rewards untuk situasi tertentu
            reward += self._calculate_shaping_rewards()
        
        self.episode_rewards.append(reward)
        observation = self.state_encoder.encode(self.board)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_draw_reward(self) -> float:
        """Hitung reward untuk draw berdasarkan material advantage."""
        material_balance = self._calculate_material_balance()
        
        if material_balance > 2.0:  # Signifikan advantage
            return self.reward_config['draw_disadvantage']  # Seharusnya menang
        elif material_balance < -2.0:
            return self.reward_config['draw_advantage']  # Bagus dapat draw
        else:
            return self.reward_config['draw']
    
    def _calculate_material_balance(self) -> float:
        """Hitung keseimbangan material dari perspektif pemain sekarang."""
        balance = 0.0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == self.board.turn:
                    balance += value
                else:
                    balance -= value
        
        return balance
    
    def _calculate_shaping_rewards(self) -> float:
        """Hitung reward shaping untuk encourage good play."""
        reward = 0.0
        
        # Check bonus
        if self.board.is_check():
            reward += self.reward_config['check_bonus']
        
        # Center control bonus (squares d4, d5, e4, e5)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for sq in center_squares:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                reward += self.reward_config['center_control']
        
        # Mobility bonus (normalized by typical move count)
        num_legal_moves = len(list(self.board.legal_moves))
        mobility_reward = (num_legal_moves / 35.0) * self.reward_config['mobility_bonus']
        reward += mobility_reward
        
        return reward
    
    def _get_info(self, illegal: bool = False) -> Dict[str, Any]:
        """Get additional information about current state."""
        info = {
            'move_count': self.move_count,
            'legal_moves': len(list(self.board.legal_moves)) if self.board else 0,
            'is_check': self.board.is_check() if self.board else False,
            'fen': self.board.fen() if self.board else '',
            'turn': 'white' if self.board and self.board.turn == chess.WHITE else 'black',
            'illegal_move': illegal,
            'episode_reward': sum(self.episode_rewards)
        }
        
        if self.board and self.board.is_game_over():
            info['game_result'] = self.board.result()
            info['game_over_reason'] = self._get_game_over_reason()
        
        return info
    
    def _get_game_over_reason(self) -> str:
        """Get reason for game over."""
        if self.board.is_checkmate():
            return 'checkmate'
        elif self.board.is_stalemate():
            return 'stalemate'
        elif self.board.is_insufficient_material():
            return 'insufficient_material'
        elif self.board.is_fifty_moves():
            return 'fifty_moves'
        elif self.board.is_repetition():
            return 'repetition'
        elif self.move_count >= self.max_moves:
            return 'max_moves'
        return 'unknown'
    
    def get_legal_action_mask(self) -> np.ndarray:
        """
        Get mask of legal actions.
        
        Returns:
            np.ndarray: Boolean mask shape (4672,)
        """
        return self.action_space_handler.get_legal_action_mask(self.board)
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render current board state.
        
        Returns:
            Rendered output based on render_mode
        """
        if self.render_mode == 'ansi':
            return str(self.board)
        elif self.render_mode == 'human':
            print(self.board)
            print(f"\nMove: {self.move_count}, Turn: {'White' if self.board.turn else 'Black'}")
            return None
        elif self.render_mode == 'rgb_array':
            return self._render_rgb()
        
        return None
    
    def _render_rgb(self) -> np.ndarray:
        """Render board sebagai RGB image."""
        try:
            import chess.svg
            import cairosvg
            from io import BytesIO
            from PIL import Image
            
            svg_data = chess.svg.board(self.board)
            png_data = cairosvg.svg2png(bytestring=svg_data.encode())
            image = Image.open(BytesIO(png_data))
            return np.array(image)
        except ImportError:
            # Fallback ke simple ASCII representation
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_board_fen(self) -> str:
        """Get current FEN string."""
        return self.board.fen() if self.board else ''
    
    def set_board_fen(self, fen: str):
        """Set board dari FEN string."""
        self.board = chess.Board(fen)
    
    def clone(self) -> 'ChessEnv':
        """Create a deep copy of the environment."""
        new_env = ChessEnv(
            render_mode=self.render_mode,
            max_moves=self.max_moves,
            reward_config=self.reward_config.copy(),
            use_extended_state=self.use_extended_state
        )
        new_env.board = self.board.copy()
        new_env.move_count = self.move_count
        new_env.episode_rewards = self.episode_rewards.copy()
        return new_env


class ChessEnvSelfPlay(ChessEnv):
    """
    Extended Chess Environment untuk self-play training.
    Menyediakan fitur tambahan untuk training dua sisi player.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_player = chess.WHITE
        self.game_history: List[Tuple[np.ndarray, int, float]] = []
    
    def step_for_player(
        self, 
        action: int, 
        player: chess.Color
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step dengan tracking per-player.
        
        Args:
            action: Action index
            player: Player color (chess.WHITE atau chess.BLACK)
            
        Returns:
            Same as step()
        """
        if self.board.turn != player:
            raise ValueError(f"Not {player}'s turn!")
        
        # Store state before action
        state_before = self.state_encoder.encode(self.board)
        
        result = self.step(action)
        
        # Store in history
        self.game_history.append((state_before, action, result[1]))
        
        return result
    
    def get_game_history(self) -> List[Tuple[np.ndarray, int, float]]:
        """Get history of all moves in this game."""
        return self.game_history
    
    def reset(self, **kwargs):
        self.game_history = []
        return super().reset(**kwargs)
    
    def flip_state_for_black(self, state: np.ndarray) -> np.ndarray:
        """Flip state agar dari perspektif Black."""
        return self.state_encoder.flip_perspective(state)


def make_chess_env(**kwargs) -> ChessEnv:
    """Factory function untuk membuat chess environment."""
    return ChessEnv(**kwargs)


def make_vectorized_envs(
    n_envs: int,
    **env_kwargs
) -> gym.vector.VectorEnv:
    """
    Buat vectorized environments untuk parallel training.
    
    Args:
        n_envs: Jumlah parallel environments
        **env_kwargs: Arguments untuk ChessEnv
        
    Returns:
        Vectorized environment
    """
    def make_env():
        return ChessEnv(**env_kwargs)
    
    return gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])
