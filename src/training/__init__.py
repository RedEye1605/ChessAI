# training package
from .trainer import Trainer, load_config
from .self_play import SelfPlayManager

__all__ = ['Trainer', 'SelfPlayManager', 'load_config']

