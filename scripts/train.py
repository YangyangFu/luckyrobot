from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union 

import os
import gymnasium as gym
import torch
import pprint
import datetime
import argparse
import numpy as np
import time
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
import random 

from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.utils.net.common import ActorCritic
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

# customized env
from envs.bullet_env import make_env
from models.mlp import MLP
from envs.bullet_env import ScaleObservationWrapper

def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Pipeline:
    """Training pipeline DRL agents."""
    
    def __init__(self, args):
        """Initialize pipeline with arguments."""
        self.args = args
        self.env = None
        self.train_envs = None
        self.test_envs = None
        self.actor = None
        self.critic = None
        self.actor_critic = None
        self.policy = None
        self.train_collector = None
        self.test_collector = None
        self.logger = None
        self.log_path = None
        
    def setup_environments(self):
        """Create and configure training and testing environments."""
        self.env = make_env(env_name=self.args.task)
        self.args.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.args.action_shape = self.env.action_space.shape or self.env.action_space.n
        self.args.max_action = self.env.action_space.high[0]
        print("Observations shape:", self.args.state_shape)
        print("Actions shape:", self.args.action_shape)
        print("Action range:", np.min(self.env.action_space.low),
              np.max(self.env.action_space.high))
        
        #self.env = ScaleObservationWrapper(self.env)
        
        self.train_envs = SubprocVectorEnv(
            [lambda: make_env(env_name=self.args.task) for _ in range(self.args.training_num)])
        
        # only render when testing
        render = True if self.args.render > 0 else False
        render = render and self.args.watch 
        
        self.test_envs = SubprocVectorEnv(
            [lambda: make_env(env_name=self.args.task, renders=render) for _ in range(self.args.test_num)])
        
    def set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.train_envs.seed([self.args.seed + i for i in range(self.args.training_num)])
        
        # debug 
        env0 = self.train_envs._env_fns[0]()
        print(f"env0 seed: {env0._seed}")

        
        self.test_envs.seed([self.args.seed + i for i in range(self.args.test_num)])

        
    def build_model(self):
        """Build actor-critic model."""
        net_a = Net(self.args.state_shape, 
                 hidden_sizes=self.args.hidden_sizes,
                 activation=nn.ReLU, 
                 device=self.args.device)
        
        self.actor = ActorProb(
            net_a, 
            self.args.action_shape, 
            max_action=self.args.max_action,
            device=self.args.device
        ).to(self.args.device)
        
        net_c = Net(self.args.state_shape, 
                    hidden_sizes=self.args.hidden_sizes,
                    activation=nn.ReLU, 
                    device=self.args.device)
        
        self.critic = Critic(
            net_c, 
            device=self.args.device
        ).to(self.args.device)
        
        torch.nn.init.constant_(self.actor.sigma_param, -0.5)
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in self.actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
            
        self.actor_critic = ActorCritic(self.actor, self.critic)
        
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        optim = torch.optim.Adam(self.actor_critic.parameters(), lr=self.args.lr, eps=1e-5)
        
        lr_scheduler = None
        if self.args.lr_decay:
            max_update_num = np.ceil(
                self.args.step_per_epoch / self.args.step_per_collect) * self.args.epoch
            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
        
        return optim, lr_scheduler
        
    def create_policy(self, optim, lr_scheduler):
        """Create PPO policy."""
        def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]):
            loc, scale = loc_scale
            return Independent(Normal(loc, scale), 1)
                
        self.policy = PPOPolicy(
            actor=self.actor,
            critic=self.critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            max_grad_norm=self.args.max_grad_norm,
            vf_coef=self.args.vf_coef,
            ent_coef=self.args.ent_coef,
            reward_normalization=self.args.rew_norm,
            action_scaling=False,
            action_bound_method=self.args.bound_action_method,
            lr_scheduler=lr_scheduler,
            action_space=self.env.action_space,
            eps_clip=self.args.eps_clip,
            value_clip=self.args.value_clip,
            dual_clip=self.args.dual_clip,
            advantage_normalization=self.args.norm_adv,
            recompute_advantage=self.args.recompute_adv
        )
        
        if self.args.resume_path:
            self.policy.load_state_dict(
                torch.load(self.args.resume_path, map_location=self.args.device))
            print("Loaded agent from: ", self.args.resume_path)
            
    def setup_collectors(self):
        """Setup data collectors for training and testing."""
        if self.args.training_num > 1:
            buffer = VectorReplayBuffer(self.args.buffer_size, len(self.train_envs))
        else:
            buffer = ReplayBuffer(self.args.buffer_size)
            
        self.train_collector = Collector(
            self.policy, self.train_envs, buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, self.test_envs)
        
    def setup_logging(self):
        """Setup tensorboard logging."""
        self.log_path = os.path.join(self.args.logdir, self.args.task)
        writer = SummaryWriter(self.log_path)
        writer.add_text("args", str(self.args))
        self.logger = TensorboardLogger(writer)
        
    def train(self):
        """Training loop."""
        def save_best_fn(policy) -> None:
            # Only save the policy state dict
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))
                            
        result = OnpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=self.logger,
            test_in_train=False,
        ).run()
        
        return result
        
    def evaluate(self):
        """Evaluate trained policy.
        
        This method can be used both during training and for standalone evaluation.
        """
        self.policy.eval()
        self.test_envs.seed(self.args.seed)
        
        if self.args.save_buffer_name:
            buffer = VectorReplayBuffer(
                self.args.buffer_size,
                buffer_num=len(self.test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=self.args.frames_stack,
            )
            collector = Collector(
                self.policy, self.test_envs, buffer, exploration_noise=True)
            result = collector.collect(
                n_step=self.args.buffer_size, reset_before_collect=True)
            buffer.save_hdf5(self.args.save_buffer_name)
        else:
            print(f"Testing agent over {self.args.test_num} episodes...")
            self.test_collector.reset()
            result = self.test_collector.collect(
                n_episode=self.args.test_num, render=self.args.render)
        
        # Process and display results
        print(result)
        # Access statistics from the new CollectStats object
        rew_mean = result.returns_stat.mean
        rew_std = result.returns_stat.std
        length_mean = result.lens_stat.mean
        
        print(f"\nEvaluation Results over {result.n_collected_episodes} episodes:")
        print(f"Mean Reward: {rew_mean:.2f} ± {rew_std:.2f}")
        print(f"Mean Episode Length: {length_mean:.2f}")
        
        # Save results if requested (typically during standalone evaluation)
        if hasattr(self.args, 'save_results') and self.args.save_results:
            results = {
                "reward_mean": rew_mean,
                "reward_std": rew_std,
                "length_mean": length_mean,
                "n_episodes": result.n_collected_episodes
            }
            save_path = os.path.join(os.path.dirname(self.args.resume_path 
                                                    if self.args.resume_path 
                                                    else self.log_path), 
                                    "eval_results.txt")
            with open(save_path, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
            print(f"\nResults saved to {save_path}")
        
        return result
        
    def run(self):
        """Run the complete pipeline."""
        # Setup
        self.setup_environments()
        self.set_seed()
        self.build_model()
        optim, lr_scheduler = self.setup_optimization()
        self.create_policy(optim, lr_scheduler)
        self.setup_collectors()
        self.setup_logging()
        
        # Training
        if not self.args.watch:
            result = self.train()
            pprint.pprint(result)
        
        # Evaluation
        self.evaluate()

def main(args):
    pipeline = Pipeline(args)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HumanoidDeepMimicWalkBulletEnv-v1')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--step-per-epoch', type=int, default=1000)#!!!!!!!!!!!!!
    parser.add_argument('--step-per-collect', type=int, default=400)#!!!!!!!!!!!!!
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=1)
    # ppo special
    parser.add_argument('--rew-norm', type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    # bound action to [-1,1] using different methods. empty means no bounding
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log_ppo')
    parser.add_argument('--render', type=float, default=1.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)#'ckpts/policy.pth')
    parser.add_argument('--test-only', type=bool, default=False)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    # tunable parameters
    parser.add_argument('--lr', type=float, default=3e-04) #0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=4096)

    args = parser.parse_args()
    args.hidden_sizes= [2048, 128]
    
    main(args)# baselines [32, 32]