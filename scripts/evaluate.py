#!/usr/bin/env python3
"""
Script Evaluasi untuk Chess RL
==============================
Evaluasi model yang sudah ditraining.

Penggunaan:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --stockfish
    python evaluate.py --checkpoint checkpoints/best.pt --games 100
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment import ChessEnv
from src.models import create_network
from src.evaluation import Evaluator, StockfishEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluasi Chess RL Agent'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path ke model checkpoint'
    )
    
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=50,
        help='Jumlah games untuk evaluasi (default: 50)'
    )
    
    parser.add_argument(
        '--stockfish',
        action='store_true',
        help='Evaluasi melawan Stockfish'
    )
    
    parser.add_argument(
        '--stockfish-path',
        type=str,
        default='stockfish',
        help='Path ke Stockfish executable'
    )
    
    parser.add_argument(
        '--stockfish-elo',
        type=int,
        default=1500,
        help='ELO limit untuk Stockfish (default: 1500)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cuda, cpu)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model dari checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get network config dari checkpoint
    config = checkpoint.get('config', {}).get('network', {})
    
    # Create network
    network = create_network({
        'input_channels': config.get('input_channels', 14),
        'num_filters': config.get('num_filters', 256),
        'num_residual_blocks': config.get('num_residual_blocks', 10),
        'action_size': config.get('action_size', 4672),
        'normalization': config.get('normalization', 'layer'),
        'activation': config.get('activation', 'relu'),
        'dropout': 0.0  # No dropout during evaluation
    })
    
    # Load weights
    network.load_state_dict(checkpoint['network_state_dict'])
    network = network.to(device)
    network.eval()
    
    return network


def main():
    """Main evaluation function."""
    print("\n" + "=" * 60)
    print("📊 Chess RL - Evaluasi Model")
    print("=" * 60 + "\n")
    
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Device: {device}")
    
    # Load model
    print(f"\n📂 Loading model: {args.checkpoint}")
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint tidak ditemukan: {args.checkpoint}")
        sys.exit(1)
    
    network = load_model(args.checkpoint, device)
    print("✅ Model loaded successfully")
    
    # Create environment
    env = ChessEnv()
    
    # Basic evaluation
    print(f"\n🎮 Evaluasi dasar ({args.games} games)...")
    evaluator = Evaluator(env, device)
    
    results = evaluator.evaluate_agent(
        network,
        num_games=args.games,
        verbose=args.verbose
    )
    
    print("\n" + "-" * 40)
    print("📈 Hasil Evaluasi Dasar:")
    print("-" * 40)
    print(f"   Games: {results['num_games']}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Draw Rate: {results['draw_rate']:.1%}")
    print(f"   Loss Rate: {results['loss_rate']:.1%}")
    print(f"   Avg Game Length: {results['avg_game_length']:.1f} moves")
    print(f"   Checkmate Rate: {results['checkmate_rate']:.1%}")
    
    # Stockfish evaluation
    if args.stockfish:
        print(f"\n🏆 Evaluasi vs Stockfish (ELO {args.stockfish_elo})...")
        
        sf_evaluator = StockfishEvaluator(
            stockfish_path=args.stockfish_path,
            elo_limit=args.stockfish_elo
        )
        
        if sf_evaluator.is_available():
            sf_results = sf_evaluator.evaluate_against_stockfish(
                network, env, device,
                num_games=min(args.games // 2, 10)  # Fewer games vs Stockfish
            )
            
            print("\n" + "-" * 40)
            print("📈 Hasil vs Stockfish:")
            print("-" * 40)
            print(f"   Games: {sf_results['total_games']}")
            print(f"   Win Rate: {sf_results['win_rate']:.1%}")
            print(f"   Draw Rate: {sf_results['draw_rate']:.1%}")
            print(f"   Avg Game Length: {sf_results['avg_game_length']:.1f} moves")
            print(f"   Estimated ELO: ~{sf_results['estimated_elo']}")
            
            # Move quality analysis
            print("\n🔍 Analisis kualitas langkah...")
            quality_results = sf_evaluator.analyze_agent_moves(
                network, env, device,
                num_positions=50
            )
            
            print("-" * 40)
            print("📊 Kualitas Langkah:")
            print("-" * 40)
            print(f"   Exact match dengan Stockfish: {quality_results['exact_match_rate']:.1%}")
            print(f"   Top 3 match: {quality_results['top3_match_rate']:.1%}")
            print(f"   Avg eval difference: {quality_results['avg_eval_difference']:.2f}")
            
            sf_evaluator.close()
        else:
            print("⚠️ Stockfish tidak tersedia")
    
    print("\n✅ Evaluasi selesai!")


if __name__ == '__main__':
    main()
