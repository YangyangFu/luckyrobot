from abc import ABC, abstractmethod
from configs.base_config import BaseConfig

class BaseTrainer(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config
        
    @abstractmethod
    def setup_envs(self):
        """Setup training and testing environments"""
        pass
    
    @abstractmethod
    def train(self):
        """Train the agent"""
        pass
    
    @abstractmethod
    def evaluate(self):
        """Evaluate the agent"""
        pass 
    
    @abstractmethod
    def watch_and_save_video(self, video_dir: str):
        """Watch the agent and save a video"""
        pass