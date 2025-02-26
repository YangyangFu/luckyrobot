import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class ReachGraspEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # PyBullet initialization
        self.render_mode = render_mode
        connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        try:
            self.physics_client = p.connect(connection_mode)
        except p.error:
            # If GUI connection fails, fallback to DIRECT
            if connection_mode == p.GUI:
                print("Warning: GUI connection failed, falling back to DIRECT mode")
                self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane and robot
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # Add parallel gripper (simplified as two boxes)
        self.gripper_distance = 0.08  # Maximum gripper opening
        self.left_finger = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.05])
        self.right_finger = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.05])
        
        # Define spaces
        # Actions: 7 joint velocities + 1 gripper command
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(8,), 
            dtype=np.float32
        )
        
        # Observations: 7 joint positions + 7 joint velocities + 3 target position + 1 gripper state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(18,),
            dtype=np.float32
        )
        
        # Target object
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.7]
        )
        self.target_position = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset robot joints
        initial_positions = [0.0] * 7
        for i in range(7):
            p.resetJointState(self.robot, i, initial_positions[i])
        
        # Reset gripper
        self.gripper_state = 0.0  # closed
        
        # Generate new target position
        self.target_position = np.random.uniform(
            low=[0.2, -0.5, 0.2],
            high=[0.5, 0.5, 0.8]
        )
        
        # Visualize target
        if hasattr(self, 'target'):
            p.removeBody(self.target)
        self.target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            basePosition=self.target_position
        )
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get joint states
        joint_states = [p.getJointState(self.robot, i) for i in range(7)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Combine all observations
        obs = np.concatenate([
            joint_positions,
            joint_velocities,
            self.target_position,
            [self.gripper_state]
        ])
        return obs
    
    def step(self, action):
        # Add debug print
        #print(f"Environment step with action: {action}")
        
        # Apply joint actions
        for i in range(7):
            p.setJointMotorControl2(
                self.robot,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=action[i] * 0.5  # Scale down velocities
            )
        
        # Apply gripper action
        self.gripper_state = np.clip(
            self.gripper_state + action[7] * 0.1,  # Scale gripper movement
            0.0,  # Closed
            1.0   # Open
        )
        
        # Simulate
        for _ in range(20):  # Multiple steps for stability
            p.stepSimulation()
        
        # Get end effector position
        state = p.getLinkState(self.robot, 6)
        current_pos = np.array(state[0])
        
        # Calculate reward
        distance = np.linalg.norm(current_pos - self.target_position)
        reward = -distance  # Basic reward based on distance
        
        # Add bonus for reaching target
        if distance < 0.05:
            reward += 10.0
        
        # Check if done
        done = distance < 0.05 and self.gripper_state < 0.1  # Close to target and gripper closed
        
        return self._get_observation(), reward, done, False, {}
    
    def close(self):
        p.disconnect(self.physics_client) 