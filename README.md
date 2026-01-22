# 🎯 Chess RL dengan Adaptive Optimization

> **Enhancing Stability in Chess Reinforcement Learning with Adaptive Optimization**

Proyek ini mengembangkan agen catur menggunakan **Reinforcement Learning (RL)** dengan fokus pada **stabilitas pelatihan** melalui teknik **optimisasi adaptif**.

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Training di Google Colab](#training-di-google-colab)
- [Visualisasi](#visualisasi)
- [Struktur Proyek](#struktur-proyek)
- [Konfigurasi](#konfigurasi)
- [Evaluasi](#evaluasi)

## 🎮 Tentang Proyek

Proyek ini bertujuan untuk:

1. **Meningkatkan Stabilitas Training** - Mengatasi masalah ketidakstabilan gradien dan overfitting dalam RL
2. **Adaptive Optimization** - Menerapkan teknik optimisasi yang menyesuaikan diri dengan dinamika pelatihan
3. **Agen Catur yang Kuat** - Menciptakan agen yang dapat bermain catur dengan strategi yang efektif
4. **Generalisasi** - Mengembangkan framework yang dapat diterapkan ke domain RL lainnya

## ✨ Fitur Utama

### 🧠 Neural Network Architecture
- Policy-Value Network dengan Residual Blocks
- Layer Normalization untuk stabilitas
- Attention mechanisms (optional)

### 📈 Adaptive Optimization
- Learning Rate Warmup & Cosine Annealing
- Gradient Clipping (Global Norm, Per-Parameter, Adaptive)
- Dynamic Clip Range Adjustment
- Entropy Scheduling

### 🎯 PPO Algorithm
- Proximal Policy Optimization dengan stability enhancements
- Generalized Advantage Estimation (GAE)
- Self-play training mechanism

### 🎨 Visualisasi
- Web interface untuk melihat AI bermain catur
- Real-time game visualization
- Training progress dashboard

### 📊 Evaluasi
- Stockfish integration untuk benchmarking
- ELO rating estimation
- Comprehensive metrics tracking

## 🚀 Instalasi

### Prasyarat
- Python 3.9+
- CUDA (untuk GPU training, optional)

### Setup Lokal

```bash
# Clone repository
cd chess

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Stockfish (untuk evaluasi)

```bash
# Windows - download dari https://stockfishchess.org/download/
# Letakkan stockfish.exe di folder project atau tambahkan ke PATH

# Linux:
sudo apt-get install stockfish
```

## 💻 Penggunaan

### Training

```bash
# Training dengan konfigurasi default
python scripts/train.py

# Training dengan konfigurasi custom
python scripts/train.py --config config/custom.yaml

# Training dengan GPU
python scripts/train.py --device cuda

# Training v27 (latest version)
python scripts/train_v27.py
```

### Evaluasi

```bash
# Evaluasi model
python scripts/evaluate.py --checkpoint checkpoints/chess_v27_final.pt

# Evaluasi melawan Stockfish
python scripts/evaluate.py --checkpoint checkpoints/chess_v27_final.pt --stockfish
```

### Bermain Melawan AI

```bash
# Mode interaktif di terminal
python scripts/play.py --checkpoint checkpoints/chess_v27_final.pt

# Mode visual (web interface)
python scripts/web_server.py
```

### Visualisasi Web

```bash
# Jalankan web server
python scripts/web_server.py

# Buka browser di http://localhost:5000
```

## ☁️ Training di Google Colab

1. Buka notebook `notebooks/colab_training.ipynb`
2. Atau gunakan VS Code dengan Colab Extension:
   - Install [Colab Extension](https://marketplace.visualstudio.com/items?itemName=googlecolab.colab)
   - Buka notebook file
   - Sign in ke Google
   - Select Kernel > Colab > New Colab Server

### Quick Start Colab

```python
# Di Colab cell pertama
!git clone https://github.com/username/chess-rl.git
%cd chess-rl
!pip install -r requirements.txt

# Mulai training
!python train.py --device cuda
```

## 📁 Struktur Proyek

```
chess/
├── config/                 # File konfigurasi
│   ├── default.yaml       # Konfigurasi default
│   └── colab.yaml         # Konfigurasi untuk Colab
├── src/                    # Source code utama
│   ├── core/              # Domain & business logic
│   ├── environment/       # Chess environment
│   ├── models/            # Neural network
│   ├── optimization/      # Adaptive optimizer
│   ├── algorithms/        # PPO implementation
│   ├── training/          # Training loop
│   ├── stability/         # Stability monitoring
│   ├── evaluation/        # Evaluasi & Stockfish
│   └── visualization/     # Web interface
├── scripts/               # Entry point scripts
│   ├── train.py           # Training script
│   ├── train_v27.py       # Training v27 (latest)
│   ├── evaluate.py        # Evaluation script
│   ├── play.py            # Interactive play
│   └── web_server.py      # Web interface server
├── data/                  # Training data
│   ├── supervised_data.npz
│   └── opening_book.pkl
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── checkpoints/           # Model saves
├── deployment/            # Deployment files
├── logs/                  # Training logs
├── archive/               # Legacy files (not in git)
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## ⚙️ Konfigurasi

Semua hyperparameter dapat dikonfigurasi melalui file YAML di folder `config/`.

Parameter penting:
- `network.num_residual_blocks`: Kedalaman network (default: 10)
- `ppo.learning_rate`: Learning rate (default: 3e-4)
- `ppo.clip_range`: PPO clip range (default: 0.2)
- `adaptive_optimization.lr_scheduler`: Tipe LR scheduler
- `training.total_timesteps`: Total training steps

## 📈 Evaluasi

### Metrik yang Diukur

| Metrik | Deskripsi |
|--------|-----------|
| Win Rate | Persentase kemenangan |
| ELO Rating | Estimasi kekuatan rating |
| Policy Loss | Loss dari policy network |
| Value Loss | Loss dari value network |
| Entropy | Exploration level |
| Gradient Norm | Stabilitas training |

### Tensorboard

```bash
tensorboard --logdir logs
```

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan buat issue atau pull request.

## 📜 Lisensi

MIT License - Lihat [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [python-chess Library](https://python-chess.readthedocs.io/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Stockfish Engine](https://stockfishchess.org/)
