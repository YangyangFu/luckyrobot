from configs.mujoco_config import MujocoConfig
from trainers.mujoco_trainer import MujocoTrainer

def main():
    # Load configuration
    config = MujocoConfig()
    
    # Create trainer
    trainer = MujocoTrainer(config)
    
    # Train or evaluate
    if config.watch and not config.render:
        stats = trainer.evaluate()
        print("Evaluation stats:", stats)
    elif config.watch and config.render:
        trainer.watch_and_save_video(video_dir="videos/Humanoid-v4")
    else:
        result = trainer.train()
        print("Training complete!")
        print(result)
        
        # Final evaluation
        eval_stats = trainer.evaluate()
        print("Final evaluation:", eval_stats)

if __name__ == "__main__":
    main() 