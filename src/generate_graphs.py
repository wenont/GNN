import argparse
import os
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp


def generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p0, p1):
    '''
    Generate a synthetic dataset of graphs with specific properties and labels.

    The dataset contains `num_samples` graphs, where each graph has two groups of nodes.
    Nodes have three-dimensional features, and edges are created based on specified
    probabilities.

    Dataset structure:
    - Each graph has `2 * num_nodes_per_group` nodes, split equally into two groups.
    - Nodes in the same group have features with positions sampled from a normal 
      distribution:
        - Group 1: mean `mean_group1` (default: (1, 1)), standard deviation `std`.
        - Group 2: mean `mean_group2` (default: (-1, -1)), standard deviation `std`.
    - The third dimension of the node feature indicates the group (0 for Group 1, 1 for Group 2).

    Edge connections:
    - Nodes within the same group are connected with probability `p0`.
    - Nodes from different groups are connected with probability `p1`.

    Graph labels:
    - A graph label is `0` if the number of inter-group connections exceeds intra-group connections.
    - Otherwise, the label is `1`.

    Parameters:
        n0 (int): Number of graphs in the dataset.
        n1 (int): Number of graphs in the dataset.
        mu0 (list): Mean of the node features for Group 1.
        mu1 (list): Mean of the node features for Group 2.
        sigma0 (float): Standard deviation of the node features.
        sigma1 (float): Standard deviation of the node features.
        p0 (float): Probability of connection between nodes in the same group.
        p1 (float): Probability of connection between nodes in different groups.


    Returns:
        list: A list of PyTorch Geometric `Data` objects, each representing a graph
        in the dataset.    
    

    Example:
        >>> dataset = generate_dataset(100, 100, [-10, 0], [10, 0], 1, 1, 0.1, 0.1)
        >>> print(len(dataset))
        100
        >>> visualize_graph(dataset[0])
    '''
    # Generate the dataset

    dataset = []
    for i in range(300): # <------ the number of graphs in the dataset
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
            for j in range(i, n0 + n1):
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
                        default=[1, 1], nargs="+", type=len_eq_2,
                        help="Sigma0 and sigma1")
    parser.add_argument("--p0", default=0.1, type=float, help="Probability of connection between nodes in the same group")
    parser.add_argument("--p1", default=0.1, type=float, help="Probability of connection between nodes in different groups")
    args = parser.parse_args()

    n0, n1 = args.prior_distribution_params
    mu0, mu1 = args.node_center_0, args.node_center_1
    sigma0, sigma1 = args.sigmas
    p0, p1 = args.p0, args.p1

    for p in [0.85]:
        print(f"Generating dataset with p={p}")
        dataset_minus = generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p, p-0.1)
        print('Dataset minus generated')
        dataset = generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p, p)
        print('Dataset generated')
        dataset_plus = generate_dataset(n0, n1, mu0, mu1, sigma0, sigma1, p, p+0.1)
        print('Dataset plus generated')

        # sum the datasets
        dataset = dataset_minus + dataset + dataset_plus

        # Save the dataset
        print('Saving the dataset...')
        torch.save(dataset, osp.join(osp.dirname(__file__), '..', 'data', 'generated_dataset', f'p1={p}.pt'))
        print('Dataset saved!')