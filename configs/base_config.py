from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class BaseConfig:
    # Common parameters
    seed: int = 0
    training_num: int = 1
    test_num: int = 1
    hidden_sizes: List[int] = (256, 256)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    buffer_size: int = 1_000_000
    
    # SAC specific
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = False
    alpha_lr: float = 3e-4
    
    # Training
    epoch: int = 200
    batch_size: int = 256
    step_per_epoch: int = 5000
    update_per_step: int = 1
    n_step: int = 1
    
    # Logging
    logdir: str = "log"
    render: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger: str = "tensorboard"
    watch: bool = True 