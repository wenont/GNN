'''
This file is used to generate the dataset for the model.

The dataset has the following structure:
- The dataset contains 100 samples.
- Each sample is a graph.
- Each graph has two groups of node, both groups have 100 nodes.
- There are no edges between nodes in a graph.
- The node features are three dimensional. 
- the first two dimensions are the position of the node, sampled from a normal distribution with mean 1 and standard deviation 1 for the first group, and mean -1 and standard deviation 1 for the second group.
- the third dimension is the group of the node, 0 for the first group, 1 for the second group.


How to generate the dataset:
- Use the pytorch geometric library to generate the dataset.
'''
import argparse
from ast import parse
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p0, p1):
    '''
    Generate the dataset. The generated dataset has the following structure:
    - The dataset contains 100 samples.
    - Each sample is a graph.
    - Each graph has two groups of node, both groups have 100 nodes.
    - There are no edges between nodes in a graph.
    - The node features are three dimensional. 
    - the first two dimensions are the position of the node, sampled from a normal distribution with mean 1 and standard deviation 1 for the first group, and mean -1 and standard deviation 1 for the second group.
    - the third dimension is the group of the node, 0 for the first group, 1 for the second group.

    For the node connections:
    - The probability of connection between nodes in the same group is p0.
    - The probability of connection between nodes in different groups is p1.
    
    For the graph labels:
    - The graph label is 0 if the graph has more inter-group connections than intra-group connections.
    '''
    # Generate the dataset

    dataset = []
    for i in range(1):
        # Generate the node features
        x0 = np.random.multivariate_normal(mu0, sigma0 * np.eye(2), n0)
        x1 = np.random.multivariate_normal(mu1, sigma1 * np.eye(2), n1)
        x = np.concatenate([x0, x1], axis=0)
        # Add group labels as third dimension
        group_labels = np.concatenate([np.zeros(n0), np.ones(n1)])[:, np.newaxis]
        x = np.concatenate([x, group_labels], axis=1)
        x = torch.tensor(x, dtype=torch.float)

        # Generate the edge index
        edge_index = []
        for i in range(n0 + n1):
            for j in range(i + 1, n0 + n1):
                if x[i, 2] == x[j, 2]:
                    if np.random.rand() < p0:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                else:
                    if np.random.rand() < p1:
                        edge_index.append([i, j])
                        edge_index.append([j, i])

        # Generate the graph label
        num_intra_group_edges = 0
        num_inter_group_edges = 0
        for i in range(n0 + n1):
            for j in range(i + 1, n0 + n1):
                if x[i, 2] == x[j, 2]:
                    num_intra_group_edges += 1
                else:
                    num_inter_group_edges += 1

        if num_intra_group_edges > num_inter_group_edges:
            y = torch.tensor([0], dtype=torch.long)
        else:
            y = torch.tensor([1], dtype=torch.long)

        # Generate the data
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset
    

def visualize_graph(graph):
    '''
    Visualize the graph. The position of the nodes are the node features.
    '''
    # Get the node features
    x = graph.x
    x = x.detach().numpy()
    # Get the group labels
    groups = x[:, 2]
    x = x[:, :2]
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    # Add nodes to the graph
    for i, node in enumerate(x):
        G.nodes[i]['pos'] = node
    # Add edges to the graph
    for edge in graph.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])
    # Create the plot

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=groups, node_size=10)
    # Show the plot
    plt.show()

    # Save the plot
    plt.savefig("graph.png")
    
def len_eq_2(input):
    if len(input) == 2 and all([type(el) is int for el in input]):
        return input
    raise argparse.ArgumentTypeError("Parameter must be 2 integers")
    
if __name__ == '__main__':
    # Generate the dataset
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prior_distribution_params",
                        default=[100, 100], nargs='+', type=len_eq_2,
                        help="Parameters (n0, n1) to set prior distributions")
    parser.add_argument("--node_center_0",
                        default=[-1, 0], nargs="+", type=len_eq_2,
                        help="Node centers mu0")
    parser.add_argument("--node_center_1",
                        default=[1, 0], nargs="+", type=len_eq_2,
                        help="Node centers mu1")
    parser.add_argument("--sigmas",
                        default=[1, 5], nargs="+", type=len_eq_2,
                        help="Sigma0 and sigma1")
    parser.add_argument("--p0", default=0.1, type=float, help="Probability of connection between nodes in the same group")
    parser.add_argument("--p1", default=0.1, type=float, help="Probability of connection between nodes in different groups")
    args = parser.parse_args()

    n0, n1 = args.prior_distribution_params
    mu0, mu1 = args.node_center_0, args.node_center_1
    sigma0, sigma1 = args.sigmas
    p0, p1 = args.p0, args.p1

    dataset = generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p0, p1)
    print(len(dataset))
    visualize_graph(dataset[0])