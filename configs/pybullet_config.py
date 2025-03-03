from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class PyBulletConfig(BaseConfig):
    task: str = "HumanoidDeepMimicWalkBulletEnv-v1"
    training_num: int = 1  # Override base config for parallel training
    hidden_sizes: tuple = (512, 256)  # Larger network for PyBullet
    start_timesteps: int = 10000
    step_per_collect: int = 1
    resume_path: str = None
    wandb_project: str = "humanoid.benchmark" 