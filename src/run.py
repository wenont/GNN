from train import hyperparameter_tuning
import wandb


sweep_id = 'gh5qqcrr'
wandb.agent(sweep_id, hyperparameter_tuning, count=1, project='bt')
