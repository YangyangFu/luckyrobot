import time
import numpy as np
from envs.humanoid_walk_env import make_env
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

def test_env():
    # Create environment
    print("Creating environment...")
    env = make_env(renders=False)  # Set renders=False for testing
    
    # Test spaces
    print("\nTesting spaces:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset:")
    obs, info = env.reset(seed=42)
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset info: {info}")
    
    # Test stepping
    print("\nTesting stepping:")
    n_steps = 100
    total_reward = 0
    
    for step in range(n_steps):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # render
        env.render(mode='rgb_array')
        
        # Print every 20 steps
        if step % 20 == 0:
            print(f"Step {step}:")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Action shape: {action.shape}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  Info: {info}")
        
        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps")
            break
        
    print(f"\nAverage reward per step: {total_reward / (step + 1):.3f}")
    
    # Test environment closing
    env.close()
    print("\nEnvironment closed successfully")


def test_vectorized_env():

    num_cpus = [1, 2, 16]
    env_name = 'HumanoidDeepMimicWalkBulletEnv-v1'
    for num_cpu in num_cpus:
        env = SubprocVectorEnv([lambda: make_env(env_name=env_name, renders=False) for _ in range(num_cpu)])
        env.reset()
        sampled_steps = 0
        time_start = time.time()
        while sampled_steps < 1000:
            act = np.random.choice(2, size=num_cpu)
            obs, rew, terminated, truncated, info = env.step(act)
            if np.sum(terminated):
                env.reset(np.where(terminated)[0])
            sampled_steps += num_cpu
        time_used = time.time() - time_start
        print(f"{time_used}s used to sample 1000 steps if using {num_cpu} cpus.")
    

if __name__ == "__main__":
    test_env()
    test_vectorized_env()
