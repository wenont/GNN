import os.path as osp
from utils import read_file_to_list
from torch_geometric.datasets import TUDataset
import torch

path = osp.join(osp.dirname(__file__), '..', 'data', 'runnable_dataset.txt')
dataset_list = read_file_to_list(path)

for dataset_name in dataset_list:
    dataset = TUDataset(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TUDataset'), name=dataset_name, use_node_attr=True)
    _, perm = dataset.shuffle(return_perm=True)
    
    # save the perm to a file, using torch.save
    torch.save(perm, osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'seeds', f'{dataset_name}.pt'))
