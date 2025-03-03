import os
import datetime
import time
import uuid 
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from typing import Tuple, Optional
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.highlevel.logger import LoggerFactoryDefault
from trainers.base_trainer import BaseTrainer

class SACTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Setup environments
        self.env, self.train_envs, self.test_envs = self.setup_envs()
                
        # Initialize networks and policy
        self.actor, self.critics = self._build_networks()
        self.policy = self._build_policy()
        
        # Setup collectors and buffer
        self.buffer = VectorReplayBuffer(config.buffer_size, len(self.train_envs))
        self.train_collector = Collector(self.policy, self.train_envs, self.buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, self.test_envs)

    def setup_envs(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def _build_networks(self) -> Tuple[ActorProb, Tuple[Critic, Critic]]:
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        
        # Actor network
        net_a = Net(state_shape=state_shape, hidden_sizes=self.config.hidden_sizes, device=self.config.device)
        actor = ActorProb(
            net_a,
            action_shape,
            device=self.config.device,
            unbounded=True,
            conditioned_sigma=True,
        ).to(self.config.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.config.actor_lr)
        
        # Critic networks
        net_c1 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=self.config.hidden_sizes,
            concat=True,
            device=self.config.device,
        )
        net_c2 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=self.config.hidden_sizes,
            concat=True,
            device=self.config.device,
        )
        critic1 = Critic(net_c1, device=self.config.device).to(self.config.device)
        critic2 = Critic(net_c2, device=self.config.device).to(self.config.device)
        

        return actor, (critic1, critic2)

    def _build_policy(self) -> SACPolicy:
        alpha = self._setup_alpha()
        
        policy = SACPolicy(
            actor=self.actor,
            actor_optim=torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr),
            critic=self.critics[0],
            critic_optim=torch.optim.Adam(self.critics[0].parameters(), lr=self.config.critic_lr),
            critic2=self.critics[1],
            critic2_optim=torch.optim.Adam(self.critics[1].parameters(), lr=self.config.critic_lr),
            tau=self.config.tau,
            gamma=self.config.gamma,
            alpha=alpha,
            estimation_step=self.config.n_step,
            action_space=self.env.action_space,
        )
        
        # Load pretrained weights if specified
        if self.config.resume_path and os.path.exists(self.config.resume_path):
            policy.load_state_dict(torch.load(self.config.resume_path, map_location=self.config.device))
            print(f"Loaded policy from: {self.config.resume_path}")
            
        return policy

    def _setup_alpha(self) -> float:
        if self.config.auto_alpha:
            target_entropy = -np.prod(self.env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.config.alpha_lr)
            return (target_entropy, log_alpha, alpha_optim)
        return self.config.alpha

    def train(self) -> None:
        # Setup logging
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        log_name = os.path.join(self.config.task, "sac", str(self.config.seed), now)
        log_path = os.path.join(self.config.logdir, log_name)
        
        logger = self._setup_logger(log_path, log_name)
        
        # Initialize training
        self.train_collector.reset()
        self.train_collector.collect(n_step=self.config.start_timesteps, random=True)
        
        # Create trainer
        trainer = OffpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.config.epoch,
            step_per_epoch=self.config.step_per_epoch,
            step_per_collect=self.config.step_per_collect,
            episode_per_test=self.config.test_num,
            batch_size=self.config.batch_size,
            save_best_fn=lambda p: self._save_policy(p, log_path),
            logger=logger,
            update_per_step=self.config.update_per_step,
            test_in_train=False,
        )
        
        return trainer.run()

    def _setup_logger(self, log_path: str, log_name: str):
        logger_factory = LoggerFactoryDefault()
        logger_factory.logger_type = self.config.logger
        if self.config.logger == "wandb":
            logger_factory.wandb_project = self.config.wandb_project
        
        # create a run id if not provided or not in the config
        run_id = str(uuid.uuid4())
        
        return logger_factory.create_logger(
            log_dir=log_path,
            experiment_name=log_name,
            run_id=run_id,
            config_dict=vars(self.config),
        )

    def _save_policy(self, policy, log_path: str) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def evaluate(self) -> None:
        self.test_collector.reset()
        stats = self.test_collector.collect(n_episode=self.config.test_num, render=self.config.render) 
        return stats
    
    def watch_and_save_video(self, video_dir: str) -> None:
        #check video dir exists
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        #create an environment with video recording
        env = gym.make(self.config.task, render_mode="rgb_array")
        env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)
        
        #set policy to evaluation mode
        self.policy.eval()
        
        state, info = env.reset(seed=self.config.seed)
        total_reward = 0.0
        done = False
        
        print("Starting GUI evaluation...")
        while not done:
            env.render()
            
            with torch.no_grad():
                action = self.policy.compute_action(state)
                
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            time.sleep(0.01)
            if done or truncated:
                break
                
        print("GUI Evaluation finished. Total reward:", total_reward)
        env.close()
        
        
        