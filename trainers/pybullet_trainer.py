from trainers.sac_trainer import SACTrainer
from configs.pybullet_config import PyBulletConfig
from envs.bullet_env import make_env, RescaleObservationWrapper, RescaleActionWrapper
from tianshou.env import SubprocVectorEnv

class PyBulletTrainer(SACTrainer):
    def setup_envs(self):
        env = make_env(env_name=self.config.task, norm_obs=False, norm_action=False)
        env = RescaleObservationWrapper(env)
        env = RescaleActionWrapper(env)
        
        train_envs = SubprocVectorEnv(
            [lambda: make_env(env_name=self.config.task, norm_obs=True, norm_action=False) 
             for _ in range(self.config.training_num)])
        
        render = self.config.render > 0 and self.config.watch
        test_envs = SubprocVectorEnv(
            [lambda: make_env(env_name=self.config.task, renders=render, norm_obs=True, norm_action=False) 
             for _ in range(self.config.test_num)])
        
        return env, train_envs, test_envs 