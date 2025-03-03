from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class MujocoConfig(BaseConfig):
    task: str = "Humanoid-v4"
    start_timesteps: int = 10000
    step_per_collect: int = 1
    resume_path: str = 'ckpts/Humanoid-v4/sac/policy.pth'
    wandb_project: str = "mujoco.benchmark" 