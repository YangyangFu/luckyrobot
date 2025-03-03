from configs.pybullet_config import PyBulletConfig
from trainers.pybullet_trainer import PyBulletTrainer

def main():
    # Load configuration
    config = PyBulletConfig()
    
    # Create trainer
    trainer = PyBulletTrainer(config)
    
    # Train or evaluate
    if config.watch:
        trainer.evaluate()
    else:
        result = trainer.train()
        print("Training complete!")
        print(result)
        
        # Final evaluation
        eval_stats = trainer.evaluate()
        print("Final evaluation:", eval_stats)

if __name__ == "__main__":
    main() 