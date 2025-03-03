# train the agent with grid search using Ray
import ray 
from ray import tune 

from configs.pybullet_config import PyBulletConfig
from trainers.pybullet_trainer import PyBulletTrainer




def train_pybullet(args):
    trainer = PyBulletTrainer(args)
    trainer.train()
    
        
def grid_search():
    args = PyBulletConfig()
    
    # Define the grid search space
    search_space = {
        "actor_lr": tune.grid_search([1e-4, 3e-4]),
        "critic_lr": tune.grid_search([1e-4, 3e-4]),
        "gamma": tune.grid_search([0.99, 0.999]),
        "tau": tune.grid_search([0.005, 0.01]),
        "alpha": tune.grid_search([0.2, 0.5]),
        "hidden_sizes": tune.grid_search([(512, 256), (1024, 512)]),
        "batch_size": tune.grid_search([64, 256]),
    }

    # test with a smaller search space
    #search_space = {
    #    "actor_lr": tune.grid_search([1e-4, 3e-4]),
    #}
    
    # add a trainable function
    def trainable_function(config):
        while True:
            args.actor_lr = config['actor_lr']
            args.critic_lr = config['critic_lr']
            args.gamma = config['gamma']
            args.tau = config['tau']
            args.alpha = config['alpha']
            args.hidden_sizes = config['hidden_sizes']
            args.batch_size = config['batch_size']
            args.device = 'cpu'
            train_pybullet(args)
            
    
    # Run the grid search
    tune.register_trainable("sac", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'sac_tuning': {
            "run": "sac",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": search_space,
            "local_dir": "ray_results",
        }
    })
    
    
if __name__ == "__main__":
    grid_search()
    