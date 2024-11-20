from ast import mod
from os import error
import wandb
from utils import read_file_to_list
from train import train_test
import os.path as osp
import pandas as pd



model_name = 'MPNN'
path = osp.join(osp.dirname(__file__), '..', 'data', 'test_dataset.txt')
dataset_list = read_file_to_list(path)

success_list = []  
error_list = []

for dataset_name in dataset_list:
    try:
        train_test(dataset_name, model_name)
        success_list.append(dataset_name)
    except Exception as e:
        print(f"Error in {dataset_name}: {e}")
        error_list.append(dataset_name)
        continue

print(f"Success: {success_list}")
print(f"Error: {error_list}")