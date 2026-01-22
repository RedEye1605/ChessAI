#!/usr/bin/env python3
"""
Script Training untuk Chess RL
==============================
Main entry point untuk training agen catur.

Penggunaan:
    python train.py                          # Training dengan config default
    python train.py --config config/colab.yaml  # Training dengan config Colab
    python train.py --device cuda            # Force GPU
    python train.py --resume checkpoint.pt   # Resume training
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training import Trainer, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Training Chess RL dengan Adaptive Optimization'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path ke file konfigurasi (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device untuk training (default: auto)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path ke checkpoint untuk resume training'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override total timesteps dari config'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Override experiment name dari config'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (less steps, more logging)'
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine device to use."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🔥 GPU tersedia: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("💻 Menggunakan CPU (GPU tidak tersedia)")
    else:
        device = torch.device(device_arg)
        print(f"📱 Device: {device}")
    
    return device


def load_and_merge_config(args) -> dict:
    """Load config and merge with command line args."""
    # Load base config
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"⚠️ Config file tidak ditemukan: {config_path}")
        print("   Menggunakan config default...")
        config = {}
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"📄 Loaded config: {config_path}")
    
    # Override dengan command line args
    if args.timesteps is not None:
        config.setdefault('training', {})['total_timesteps'] = args.timesteps
    
    if args.experiment_name is not None:
        config.setdefault('general', {})['experiment_name'] = args.experiment_name
    
    if args.debug:
        config.setdefault('training', {})['total_timesteps'] = 10000
        config.setdefault('training', {})['eval_frequency'] = 5
        config.setdefault('training', {})['checkpoint_frequency'] = 10
        config.setdefault('logging', {})['level'] = 'DEBUG'
    
    return config


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("🎯 Chess RL Training dengan Adaptive Optimization")
    print("=" * 60 + "\n")
    
    # Parse arguments
    args = parse_args()
    
    # Get device
    device = get_device(args.device)
    
    # Load config
    config = load_and_merge_config(args)
    
    # Create trainer
    print("\n📦 Menyiapkan trainer...")
    trainer = Trainer(config, device)
    
    # Resume dari checkpoint jika specified
    if args.resume:
        print(f"\n📂 Melanjutkan dari checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Print training info
    print("\n" + "-" * 60)
    print("📊 Training Configuration:")
    print(f"   Total Timesteps: {config.get('training', {}).get('total_timesteps', 1000000):,}")
    print(f"   Batch Size: {config.get('ppo', {}).get('batch_size', 256)}")
    print(f"   Learning Rate: {config.get('ppo', {}).get('learning_rate', 3e-4)}")
    print(f"   Network Layers: {config.get('network', {}).get('num_residual_blocks', 10)}")
    print("-" * 60 + "\n")
    
    try:
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training dihentikan oleh user")
        print("   Menyimpan checkpoint terakhir...")
        trainer.save_checkpoint('interrupted')
        
    finally:
        trainer.close()
    
    print("\n✅ Training selesai!")
    print(f"   Checkpoint tersimpan di: {trainer.checkpoint_dir}")
    print(f"   Logs tersimpan di: {trainer.log_dir}")


if __name__ == '__main__':
    main()
