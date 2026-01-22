#!/usr/bin/env python3
"""
Script untuk Bermain Melawan AI
================================
Bermain catur melawan AI yang sudah ditraining.

Penggunaan:
    python play.py --checkpoint checkpoints/best.pt          # Mode terminal
    python play.py --checkpoint checkpoints/best.pt --visual # Mode visual web
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment import ChessEnv
from src.models import create_network


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bermain Catur Melawan Chess RL Agent'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path ke model checkpoint (optional, random agent jika tidak ada)'
    )
    
    parser.add_argument(
        '--visual',
        action='store_true',
        help='Jalankan visual web interface'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port untuk web server (default: 5000)'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='white',
        choices=['white', 'black', 'random'],
        help='Pilih warna (default: white)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cuda, cpu)'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model dari checkpoint."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("⚠️ Checkpoint tidak ditemukan, menggunakan random agent")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {}).get('network', {})
    
    network = create_network({
        'input_channels': config.get('input_channels', 14),
        'num_filters': config.get('num_filters', 256),
        'num_residual_blocks': config.get('num_residual_blocks', 10),
        'action_size': config.get('action_size', 4672),
        'normalization': config.get('normalization', 'layer'),
        'dropout': 0.0
    })
    
    network.load_state_dict(checkpoint['network_state_dict'])
    network = network.to(device)
    network.eval()
    
    return network


def play_terminal(agent, env, device, player_color):
    """Play di terminal mode."""
    import chess
    
    print("\n" + "=" * 60)
    print("🎯 Chess RL - Mode Terminal")
    print("=" * 60)
    print("\nPerintah:")
    print("  - Ketik langkah dalam format UCI (contoh: e2e4)")
    print("  - 'quit' untuk keluar")
    print("  - 'board' untuk melihat papan")
    print("  - 'legal' untuk melihat langkah legal")
    print("  - 'undo' untuk membatalkan langkah")
    print("=" * 60 + "\n")
    
    state, _ = env.reset()
    
    while not env.board.is_game_over():
        print(env.board)
        print(f"\nGiliran: {'Putih' if env.board.turn else 'Hitam'}")
        
        is_player_turn = (
            (player_color == 'white' and env.board.turn == chess.WHITE) or
            (player_color == 'black' and env.board.turn == chess.BLACK)
        )
        
        if is_player_turn:
            # Player's turn
            while True:
                cmd = input("\n🎮 Langkah Anda: ").strip().lower()
                
                if cmd == 'quit':
                    print("\n👋 Terima kasih sudah bermain!")
                    return
                elif cmd == 'board':
                    print(env.board)
                    continue
                elif cmd == 'legal':
                    legal_moves = [m.uci() for m in env.board.legal_moves]
                    print(f"Langkah legal: {', '.join(legal_moves)}")
                    continue
                elif cmd == 'undo':
                    if len(env.board.move_stack) >= 2:
                        env.board.pop()
                        env.board.pop()
                        print("⬅️ Langkah dibatalkan")
                        print(env.board)
                    else:
                        print("⚠️ Tidak ada langkah untuk dibatalkan")
                    continue
                
                try:
                    move = chess.Move.from_uci(cmd)
                    if move in env.board.legal_moves:
                        action = env.action_space_handler.encode_move(move)
                        state, reward, done, truncated, info = env.step(action)
                        print(f"✓ Langkah: {cmd}")
                        break
                    else:
                        print("❌ Langkah tidak legal! Coba lagi.")
                except:
                    print("❌ Format tidak valid! Gunakan format UCI (contoh: e2e4)")
        else:
            # AI's turn
            print("\n🤖 AI sedang berpikir...")
            
            if agent is not None:
                import numpy as np
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    legal_mask = torch.BoolTensor(
                        env.get_legal_action_mask()
                    ).unsqueeze(0).to(device)
                    
                    log_probs, value = agent(state_tensor, legal_mask)
                    action = log_probs.argmax(dim=-1).item()
                
                move = env.action_space_handler.decode_action(action)
                print(f"🤖 AI bermain: {move.uci()} (nilai: {value.item():.2f})")
            else:
                # Random agent
                import random
                legal_moves = list(env.board.legal_moves)
                move = random.choice(legal_moves)
                action = env.action_space_handler.encode_move(move)
                print(f"🤖 AI (random) bermain: {move.uci()}")
            
            state, reward, done, truncated, info = env.step(action)
    
    # Game over
    print("\n" + "=" * 60)
    print(env.board)
    result = env.board.result()
    
    if result == '1-0':
        winner = "Putih"
    elif result == '0-1':
        winner = "Hitam"
    else:
        winner = "Seri"
    
    print(f"\n🏆 Game Selesai: {winner}!")
    print(f"   Hasil: {result}")
    print("=" * 60)


def play_visual(agent, env, device, port):
    """Run visual web interface."""
    print("\n🎨 Memulai visual interface...")
    
    from src.visualization.app import run_server
    
    run_server(
        agent=agent,
        env=env,
        device=device,
        port=port,
        debug=False
    )


def main():
    """Main function."""
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Device: {device}")
    
    # Load model
    agent = load_model(args.checkpoint, device)
    if agent is not None:
        print(f"✅ Model loaded: {args.checkpoint}")
    
    # Create environment
    env = ChessEnv()
    
    # Determine player color
    player_color = args.color
    if player_color == 'random':
        import random
        player_color = random.choice(['white', 'black'])
    
    print(f"🎨 Anda bermain sebagai: {player_color}")
    
    if args.visual:
        play_visual(agent, env, device, args.port)
    else:
        play_terminal(agent, env, device, player_color)


if __name__ == '__main__':
    main()
