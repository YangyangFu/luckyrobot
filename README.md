# luckyrobot
Interview demo for LuckyRobot MLE position - Robot Learning in PyBullet

## requirements

Thank you for your interest in the Robotics ML Engineer position. We are excited about your background and would like to move forward with a practical exercise to better understand your approach to solving real-world robotics challenges.
This role requires someone who can demonstrate the ability to train robots using machine learning and contribute effectively to our projects. To assess your skills, we have prepared an open-ended exercise that allows you to showcase your expertise in a meaningful way.

### The Exercise
Your task is to design and implement a pipeline to train a robot to perform a perception, manipulation, or locomotion task of your choice (or a project you have completed in the past that fits this exercise). This can be done entirely in simulation using free tools, and we encourage creativity in defining the task.

### Deliverables

**Task Definition**
For this project, we will implement a robotic arm reaching and grasping task using reinforcement learning in simulation. The robot will learn to:
1. Reach a target position in 3D space
2. Orient its end-effector appropriately
3. Control a parallel gripper to grasp objects

This task is fundamental to robotic manipulation and represents a core capability needed in various applications from manufacturing to service robotics.

**Pipeline and Toolchain**
Our implementation uses the following pipeline:

1. **Simulation Environment**: 
   - PyBullet for physics simulation
   - Custom gymnasium environment for RL training

2. **Learning Framework**:
   - Tianshou (built on PyTorch) for RL implementation
   - DDPG (Deep Deterministic Policy Gradient) algorithm
   - Neural networks for policy and value functions

3. **Project Structure**:

```
luckyrobot/
├── envs/ # Custom PyBullet environments
├── models/ # Neural network architectures
├── configs/ # Configuration files
├── scripts/ # Training and evaluation scripts
├── assets/ # Robot URDFs and other assets
└── utils/ # Helper functions
```
**Demonstration**
Train the robot in simulation and record a short video (up to 5 minutes) using Loom or a similar tool. The video should:
Walk us through your process, highlighting any challenges faced and how they were addressed.
Show the final result of the trained robot performing the task.

**What Not to Do**
No presentations- we're only interested in code walkthroughs and a step-by-step breakdown of your AI pipeline.
Do not exceed 5 minutes - Think of this as a modern cover letter, a short introduction, not a deep dive

