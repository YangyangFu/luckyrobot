import gym
import gymnasium as gymn
import pybullet_envs
import numpy as np

class GymToGymnasiumWrapper(gymn.Env):  # Inherit from gymnasium.Env directly
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = gymn.spaces.Box(
            low=env.observation_space.low, 
            high=env.observation_space.high, 
            shape=env.observation_space.shape, 
            dtype=np.float32)
        self.action_space = gymn.spaces.Box(
            low=env.action_space.low, 
            high=env.action_space.high, 
            shape=env.action_space.shape, 
            dtype=np.float32)

        self.metadata = getattr(env, "metadata", {})  # Preserve metadata
        self.reward_range = env.reward_range
        self._seed = None
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False  
        return obs, reward, done, truncated, info

    def seed(self, seed=None):
        self._seed = seed
        return self._seed

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

# scale the observations based on min/max values in a wrapper to [-1, 1]
class RescaleObservationWrapper(gymn.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gymn.spaces.Box), \
            "This wrapper only supports Box observation spaces."
        
        # Store original observation space limits
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        
        # Define new observation space scaled to [-1, 1]
        self.observation_space = gymn.spaces.Box(
            low=-np.ones_like(self.obs_low),
            high=np.ones_like(self.obs_high),
            dtype=np.float32
        )

    def observation(self, obs):
        """Rescale observation from original space to [-1, 1]."""
        return 2.0 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1.0
    

# scale the actions based on min/max values in a wrapper into [-1, 1]
class RescaleActionWrapper(gymn.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gymn.spaces.Box), \
            "This wrapper only supports Box action spaces."

        # Store original action space limits
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        # Define new action space scaled to [-1, 1]
        self.action_space = gymn.spaces.Box(
            low=-np.ones_like(self.act_low),
            high=np.ones_like(self.act_high),
            dtype=np.float32
        )

    def action(self, action):
        """Rescale action from [-1, 1] to original action space."""
        return self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)

    def reverse_action(self, action):
        """Rescale action from original space back to [-1, 1]."""
        return 2.0 * (action - self.act_low) / (self.act_high - self.act_low) - 1.0

        
def make_env(env_name='HumanoidDeepMimicWalkBulletEnv-v1', 
             renders=False, 
             test_mode=False,
             norm_obs=True,
             norm_action=True):
    old_env = gym.make(env_name, renders=renders, test_mode=test_mode)
    env = GymToGymnasiumWrapper(old_env)
    if norm_obs:
        env = RescaleObservationWrapper(env)
    if norm_action:
        env = RescaleActionWrapper(env)
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

