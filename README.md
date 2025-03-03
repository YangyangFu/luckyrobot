# LuckyRobot

Interview demo for LuckyRobot MLE position - Humanoid Robot Learning in Mujoco/PyBullet.

## Task Definition

For this project, I implemented a humanoid walking task using reinforcement learning in simulation. The robot learned to walk in a natural human-like manner. This task is fundamental to robotic manipulation and represents a core capability needed in various applications from manufacturing to service robotics.

## Pipeline and Toolchain

My implementation uses the following pipeline:

1. **Simulation Environment**:
   - Mujoco and PyBullet for physics simulation.
   - Observation and action rescaling wrappers based on gym/gymnasium.

2. **Learning Framework**:
   - Tianshou (built on PyTorch) for RL implementation.

3. **Hyperparameter Tuning**:
   - Ray Tune.

## Project Structure

```
luckyrobot/
   ├── envs/ # Custom PyBullet environments
   ├── models/ # Neural network architectures
   ├── configs/ # Configuration files
   ├── scripts/ # Training and evaluation scripts
   ├── tests/ # Test files
   ├── trainers/ # training pipeline 
   └── utils/ # Helper functions
   ├── videos/ # Evaulation videos
   
```

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```


## Usage

To train the robot in simulation, run:

```sh
python scripts/train_mujoco.py
```

To evaluate the robot in simulation and watch the performance, run:

```sh
python scripts/train_mujoco.py --watch
```

For hyperparameter tuning, run
```sh
python scripts/train_mujoco_grids.py
```


## Demonstration
A demonstration video of the final trained policy is saved at `videos/` for reference.


## Bugs in Dependency
The pybullet `Humanoid` environment has bugs as listed below:

- deep_mimic_env.py line 118

   ```python
   self.observation_space = spaces.Box(observation_min, observation_min, dtype=np.float32)
   ```
