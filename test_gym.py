import gym
import gymnasium as gymn
from gymnasium import spaces

import pybullet_envs


# how to register v2 version by changing some registry info

#env = gym.make("HalfCheetahBulletEnv-v0")
#env = gym.make('RacecarBulletEnv-v0')
env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1', renders=True) 
#env = gym.make('HumanoidDeepMimicWalkBulletEnv-v2')


class GymToGymnasiumWrapper(gymn.Env):  # Inherit from gymnasium.Env directly
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})  # Preserve metadata

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False  # Modify based on custom termination logic if needed
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)  # Ensure compatibility with old Gym environments
        obs = self.env.reset()
        return obs, {}

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()



# test gymnasium env
# wrap gym env with gymnasium env 
nenv = GymToGymnasiumWrapper(env)


out = nenv.reset(seed=2025)
done = False

while not done:
    action = nenv.action_space.sample()
    print(action.shape)
    obs, reward, terminated, truncated, info = nenv.step(action)
    done = terminated or truncated
    nenv.render(mode='rgb_array')

nenv.close()

