import gym
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import wandb
from torch.distributions import Normal, Independent
from typing import Tuple, Dict, Optional, Union
import os
from datetime import datetime
import json
from pathlib import Path
import time
import pybullet_envs  # Changed back to pybullet_envs

class WandbLogger:
    """Custom logger that implements Tianshou's logger interface for wandb"""
    
    def __init__(self):
        self.wandb = wandb
        
    def log_update_data(self, data: Dict, step: int) -> None:
        """Log training update data"""
        self.wandb.log({
            "train/loss": data.get("loss", 0),
            "train/policy_loss": data.get("policy_loss", 0),
            "train/value_loss": data.get("value_loss", 0),
            "train/step": step,
        })

    def log_test_data(self, data: Dict, step: int) -> None:
        """Log test data"""
        if "rew" in data:
            self.wandb.log({
                "test/reward": data["rew"],
                "test/reward_std": data.get("rew_std", 0),
                "test/length": data.get("len", 0),
                "test/step": step
            })

    def save_data(self, epoch: int, env_step: int, gradient_step: int, save_checkpoint_fn: Optional[Union[bool, None]] = None) -> None:
        """Save training data"""
        pass  # We handle saving through our own checkpoint system

class RoboticMLPipeline:
    """
    End-to-end ML pipeline for robotic control, demonstrating:
    1. Data Collection & Environment Interface
    2. Neural Network Architecture Design
    3. Policy Learning System
    4. Training Loop with Monitoring
    5. Model Evaluation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/run_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Initialize wandb for experiment tracking
        self.setup_logging(timestamp)
        
    def setup_logging(self, timestamp: str):
        """Setup experiment tracking and logging"""
        wandb.init(
            project="luckyrobot-reach-grasp",
            name=f"run_{timestamp}",
            config=self.config,
            tags=["robotic-control", "RL"],
            dir=str(self.exp_dir)
        )
        
        # Log system info
        wandb.log({
            "system/device": str(self.device),
            "system/cuda_available": torch.cuda.is_available(),
            "system/cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })
        
    def create_environments(self) -> Tuple[gym.Env, DummyVectorEnv, DummyVectorEnv]:
        """
        Step 1: Environment Setup and Data Collection Interface
        Creates training and testing environments with proper vectorization
        """
        # Base env for getting spaces
        env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1', render=False)  # Changed render_mode to render
        
        # Training environments
        train_envs = DummyVectorEnv(
            [lambda: gym.make('HumanoidDeepMimicWalkBulletEnv-v1', render=False) 
             for _ in range(self.config['num_train_envs'])]
        )
        
        # Test environments - one with rendering
        test_envs = DummyVectorEnv(
            [lambda: gym.make('HumanoidDeepMimicWalkBulletEnv-v1', 
                            render=True if i == 0 else False)  # Changed render_mode to render
             for i in range(self.config['num_test_envs'])]
        )
        return env, train_envs, test_envs
        
    def create_networks(self, env: gym.Env) -> Tuple[ActorProb, Critic, Critic]:
        """
        Step 2: Neural Network Architecture Design
        Defines actor-critic architecture for robotic control
        """
        # Network parameters
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]
        
        # Actor network (policy network)
        net_a = Net(
            state_shape, 
            hidden_sizes=self.config['actor_hidden_sizes'],
            activation=torch.nn.ReLU,
            device=self.device
        ).to(self.device)
        
        actor = ActorProb(
            net_a,
            action_shape,
            max_action=max_action,
            unbounded=True,
            device=self.device
        ).to(self.device)
        
        # Dual critics for Q-learning (value networks)
        def create_critic():
            net = Net(
                state_shape,
                action_shape,
                hidden_sizes=self.config['critic_hidden_sizes'],
                activation=torch.nn.ReLU,
                concat=True,
                device=self.device
            ).to(self.device)
            return Critic(net, device=self.device).to(self.device)
            
        critic1, critic2 = create_critic(), create_critic()
        
        # Initialize networks with orthogonal initialization
        self._initialize_networks(actor, critic1, critic2)
        
        return actor, critic1, critic2
        
    def _initialize_networks(self, *networks):
        """Initialize network parameters using orthogonal initialization"""
        for net in networks:
            for m in net.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
    
    def create_policy(self, env: gym.Env, actor: ActorProb, 
                     critic1: Critic, critic2: Critic) -> SACPolicy:
        """
        Step 3: Policy Learning System
        Sets up SAC policy with proper optimizers and hyperparameters
        """
        # Create optimizers
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.config['actor_lr'])
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.config['critic_lr'])
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.config['critic_lr'])

        # Create SAC policy with correct parameter order
        policy = SACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            action_space=env.action_space,
            tau=self.config['tau'],  # Target network update rate
            gamma=self.config['gamma'],  # Discount factor
            alpha=self.config['alpha'],  # Temperature parameter
            estimation_step=self.config['n_step_returns'],
            action_scaling=True,
            action_bound_method="clip"
        )
        return policy
    
    def setup_collectors(self, policy: SACPolicy, train_envs: DummyVectorEnv, 
                        test_envs: DummyVectorEnv) -> Tuple[Collector, Collector]:
        """
        Step 4: Data Collection System
        Creates collectors for gathering experience during training
        """
        # Create collectors first
        buffer = VectorReplayBuffer(
            self.config['buffer_size'],
            self.config['num_train_envs']
        )
        
        def _preprocess_fn(**kwargs):
            # Only convert observations and actions to tensors on GPU
            for key in ['obs', 'obs_next', 'act']:  # Removed 'rew' and 'done'
                if key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].to(self.device)
                    elif isinstance(kwargs[key], (np.ndarray, float, int)):
                        kwargs[key] = torch.as_tensor(kwargs[key], device=self.device)
                    print(f"Preprocessed {key}: device = {kwargs[key].device}")  # Debug print
            
            # Keep rewards and done flags as numpy arrays
            for key in ['rew', 'done']:
                if key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cpu().numpy()
                    elif not isinstance(kwargs[key], np.ndarray):
                        kwargs[key] = np.array(kwargs[key])
                    print(f"Preprocessed {key}: type = {type(kwargs[key])}")  # Debug print
            
            return kwargs
        
        train_collector = Collector(
            policy,
            train_envs,
            buffer,
            exploration_noise=True,
            preprocess_fn=_preprocess_fn
        )
        
        test_collector = Collector(
            policy, 
            test_envs,
            preprocess_fn=_preprocess_fn
        )
        
        # Add callback after creation
        def on_episode_end(collector, data: Batch):
            if data.done.any():  # Log when episode ends
                rewards = data.rew[data.done].cpu().numpy()  # Get rewards for completed episodes
                for reward in rewards:
                    wandb.log({
                        'training/step_reward': reward,
                        'training/step': collector.collect_step,
                    })
                    print(f"Episode ended with reward: {reward}, total steps: {collector.collect_step}")
        
        # Register the callback
        train_collector.on_episode_end = on_episode_end
        
        # Initialize collectors
        train_collector.reset()
        test_collector.reset()
        
        # Collect initial experience
        print("Collecting initial experience...")
        train_collector.collect(n_step=self.config['batch_size'])
        
        return train_collector, test_collector
    
    def save_checkpoint(self, epoch: int, policy: SACPolicy, mean_reward: float):
        """Save a training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': policy.state_dict(),
            'mean_reward': mean_reward,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.exp_dir / "latest_checkpoint.pth")
        
        # Save periodic checkpoint
        if epoch % self.config['save_interval'] == 0:
            torch.save(checkpoint, self.exp_dir / f"checkpoint_epoch_{epoch}.pth")
            wandb.save(str(self.exp_dir / f"checkpoint_epoch_{epoch}.pth"))
        
        # Save best model if this is the best reward
        if not hasattr(self, 'best_reward') or mean_reward > self.best_reward:
            self.best_reward = mean_reward
            torch.save(checkpoint, self.exp_dir / "best_model.pth")
            wandb.save(str(self.exp_dir / "best_model.pth"))

    def train(self):
        """
        Step 5: Training Loop with Monitoring
        Implements the main training loop with proper monitoring and evaluation
        """
        # Create components
        env, train_envs, test_envs = self.create_environments()
        actor, critic1, critic2 = self.create_networks(env)
        policy = self.create_policy(env, actor, critic1, critic2)
        train_collector, test_collector = self.setup_collectors(
            policy, train_envs, test_envs
        )
        
        # Initialize statistics tracking
        train_stats = {
            'epoch_times': [],
            'train_rewards': [],
            'test_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
        }
        
        start_time = time.time()
        
        def log_train_stats(epoch, env_step, reward_stat, losses=None):
            """Callback to log training statistics (called every epoch)"""
            epoch_time = time.time() - start_time
            train_stats['epoch_times'].append(epoch_time)
            
            # Get mean reward from stat
            mean_reward = reward_stat['mean'] if reward_stat is not None else 0.0
            
            # Prepare logging dict for epoch-level metrics
            log_dict = {
                'training/epoch': epoch,
                'training/step': env_step,
                'training/epoch_reward_mean': mean_reward,  # Renamed to clarify this is epoch-level
                'time/epoch_time': epoch_time,
                'time/steps_per_second': env_step / (epoch_time + 1e-6),
            }
            
            # Add reward std if available
            if reward_stat is not None and 'std' in reward_stat:
                log_dict['training/epoch_reward_std'] = reward_stat['std']  # Renamed to clarify
            
            # Log to wandb
            wandb.log(log_dict)
            
            # Print progress
            print(f"\nEpoch {epoch}")
            print(f"Steps: {env_step}")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"Time elapsed: {epoch_time:.2f}s")
            print(f"Steps per second: {env_step / (epoch_time + 1e-6):.2f}")
        
        def save_checkpoint_callback(epoch, env_step, mean_reward):
            """Callback to save checkpoints"""
            self.save_checkpoint(epoch, policy, mean_reward)
        
        # Create custom logger
        custom_logger = WandbLogger()
        
        # Training loop
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=self.config['max_epoch'],
            step_per_epoch=self.config['step_per_epoch'],
            step_per_collect=self.config['step_per_collect'],
            update_per_step=self.config['update_per_step'],
            episode_per_test=self.config['episode_per_test'],
            batch_size=self.config['batch_size'],
            stop_fn=lambda mean_rewards: mean_rewards >= self.config['target_reward'],
            logger=custom_logger,  # Use our custom logger instead of wandb directly
            save_best_fn=save_checkpoint_callback,
            save_checkpoint_fn=lambda epoch, env_step, gradient_step: epoch % self.config['save_interval'] == 0,
            verbose=True,
            train_fn=lambda epoch, env_step: log_train_stats(
                epoch, 
                env_step, 
                {'mean': train_collector.collect_step, 'std': 0.0},
                None
            ),
            test_fn=lambda epoch, env_step: log_train_stats(
                epoch, 
                env_step, 
                {'mean': test_collector.collect_step, 'std': 0.0},
                None
            )
        )
        
        # Final save
        self.save_checkpoint(self.config['max_epoch'], policy, result['best_reward'])
        
        # Log final statistics
        wandb.log({
            'final/best_reward': result['best_reward'],
            'final/total_steps': result['total_steps'],
            'final/total_time': time.time() - start_time,
        })
        
        return result

def main():
    # Configuration for the ML pipeline
    config = {
        # Environment settings
        'num_train_envs': 1,
        'num_test_envs': 1,
        
        # Network architecture - might need adjustment for new env
        'actor_hidden_sizes': [256, 256, 256],
        'critic_hidden_sizes': [256, 256, 256],
        
        # Learning parameters
        'actor_lr': 3e-4,  # Adjusted for humanoid
        'critic_lr': 3e-4,  # Adjusted for humanoid
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'n_step_returns': 1,
        
        # Training settings
        'max_epoch': 200,  # Increased for more complex task
        'step_per_epoch': 1000,  # Increased for more steps per epoch
        'step_per_collect': 100,
        'update_per_step': 0.1,
        'batch_size': 256,
        'buffer_size': 1000000,  # Increased for more complex environment
        'episode_per_test': 1,
        'target_reward': 2000,  # Adjusted for humanoid environment
        
        # Logging settings
        'save_interval': 10,
        'log_interval': 1,
        
        # Experiment tracking
        'exp_name': 'humanoid_deepmimic_baseline',
        'notes': 'SAC implementation for humanoid walking task',
    }
    
    # Create and run the pipeline
    pipeline = RoboticMLPipeline(config)
    pipeline.train()

if __name__ == "__main__":
    main() 