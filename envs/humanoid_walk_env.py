import gym
import gymnasium as gymn
import pybullet_envs


class GymToGymnasiumWrapper(gymn.Env):  # Inherit from gymnasium.Env directly
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})  # Preserve metadata

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False  
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed) 
        obs = self.env.reset()
        return obs, {}

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()


def make_env(env_name='HumanoidDeepMimicWalkBulletEnv-v1', 
             renders=True):
    old_env = gym.make(env_name, renders=renders)
    env = GymToGymnasiumWrapper(old_env)
    return env



if __name__ == "__main__":
    env = make_env(env_name='HumanoidDeepMimicWalkBulletEnv-v1', 
                   renders=True)
    out = env.reset(seed=2025)
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(action.shape, obs.shape, reward)
        done = terminated or truncated
        env.render(mode='rgb_array')

    env.close()

