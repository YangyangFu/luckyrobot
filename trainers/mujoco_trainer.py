from trainers.sac_trainer import SACTrainer
from configs.mujoco_config import MujocoConfig
from envs.mujoco_env import make_mujoco_env

class MujocoTrainer(SACTrainer):
    def setup_envs(self):
        env, train_envs, test_envs = make_mujoco_env(
            self.config.task,
            self.config.seed,
            self.config.training_num,
            self.config.test_num,
            obs_norm=False,
        )
        return env, train_envs, test_envs