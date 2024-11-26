'''
This file is used to generate the dataset for the model.

The dataset has the following structure:
- The dataset contains 100 samples.
- Each sample is a graph.
- Each graph has two groups of node, both groups have 100 nodes.
- There are no edges between nodes in a graph.
- The node features are two dimensional, and are normal distributed with mean 1 and variance 1 for the first group, and mean -1 and variance 1 for the second group.

How to generate the dataset:
- Use the pytorch geometric library to generate the dataset.
'''

from sympy import im
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_dataset():
    '''
    Generate the dataset.
    '''
    # Generate the dataset
    dataset = []
    for i in range(100):
        x_group1 = torch.normal(mean=1, std=1, size=(100, 2))
        x_group2 = torch.normal(mean=-1, std=1, size=(100, 2))
        x = torch.cat([x_group1, x_group2], dim=0)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset

def visualize_graph(graph):
    '''
    Visualize the graph. The position of the nodes are the node features.
    '''
    # Get the node features
    x = graph.x
    x = x.detach().numpy()
    x_group1 = x[:100]
    x_group2 = x[100:]

    # Create the graph
    G = nx.Graph()
    for i in range(100):
        G.add_node(i, pos=(x_group1[i][0], x_group1[i][1]))
        G.add_node(i+100, pos=(x_group2[i][0], x_group2[i][1]))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=10)
    plt.show()

    

    
if __name__ == '__main__':
    # Generate the dataset
    dataset = generate_dataset()
    print(len(dataset))
    visualize_graph(dataset[0])