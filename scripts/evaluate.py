import torch
import time
import numpy as np
from luckyrobot.envs.reach_env import ReachGraspEnv
from luckyrobot.scripts.train import RoboticMLPipeline
import wandb

def create_evaluation_policy(config):
    """Recreate the policy architecture for evaluation"""
    pipeline = RoboticMLPipeline(config)
    env, _, _ = pipeline.create_environments()
    actor, critic1, critic2 = pipeline.create_networks(env)
    policy = pipeline.create_policy(env, actor, critic1, critic2)
    return policy, env

def evaluate(num_episodes=5, render=True, record_video=True):
    """Evaluate the trained policy"""
    
    # Load configuration
    with wandb.init(project="luckyrobot-reach-grasp", job_type="eval") as run:
        artifact = run.use_artifact('reach_grasp_policy:latest')
        artifact_dir = artifact.download()
        
        # Create policy and environment
        config = run.config
        policy, env = create_evaluation_policy(config)
        policy.load_state_dict(torch.load(f"{artifact_dir}/reach_grasp_policy.pth"))
        
        # Evaluation loop
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action from policy
                action = policy.select_action(state)
                
                # Execute action
                state, reward, done, _, info = env.step(action)
                total_reward += reward
                
                if render:
                    time.sleep(0.01)
                
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        env.close()

if __name__ == "__main__":
    evaluate() 