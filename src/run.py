from train import hyperparameter_tuning
import wandb


sweep_id = 's4v4a4ij'
wandb.agent(sweep_id, hyperparameter_tuning, count=1, project='bt')
