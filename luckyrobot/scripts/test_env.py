import gymnasium as gym
import pybullet as p
import time
import numpy as np
from luckyrobot.envs.reach_env import ReachGraspEnv

def test_environment():
    """
    Test script to verify the environment works correctly.
    Tests:
    1. Environment creation
    2. Reset functionality
    3. Step functionality
    4. Rendering
    5. Action and observation spaces
    """
    
    # Create environment with rendering
    env = ReachGraspEnv(render_mode="human")
    
    # Test observation and action spaces
    print("\nSpace Information:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test random actions
    print("\nTesting random actions...")
    for i in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Print every 20 steps
        if i % 20 == 0:
            print(f"\nStep {i}:")
            print(f"Action taken: {action}")
            print(f"Observation: {obs}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            
        # Optional: sleep to see the movement
        time.sleep(0.01)
        
        if done:
            print("\nEpisode finished!")
            obs, _ = env.reset()
    
    # Test environment close
    env.close()
    print("\nEnvironment test completed!")

def test_target_visualization():
    """
    Test the target visualization and robot movement
    """
    env = ReachGraspEnv(render_mode="human")
    obs, _ = env.reset()
    
    # Move joints individually
    print("\nTesting individual joint movements...")
    for joint in range(7):
        print(f"\nTesting joint {joint}")
        for _ in range(50):
            action = np.zeros(8)  # 7 joints + 1 gripper
            action[joint] = 0.5
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.01)
        
        # Reset after testing each joint
        obs, _ = env.reset()
    
    env.close()

if __name__ == "__main__":
    print("Starting environment tests...")
    
    print("\n=== Basic Environment Test ===")
    test_environment()
    
    print("\n=== Target and Joint Movement Test ===")
    test_target_visualization() 