import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
import networkx as nx
import time
import math
from dataclasses import dataclass
from torch_geometric.datasets import TUDataset
import torch
from rich.progress import track


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    if s < 10:
        s = '0' + str(math.floor(s))
    else:
        s = str(math.floor(s))
    return f'{m}m {s}s'


@dataclass
class NetParams:
    hidden_channels: int
    num_epochs: int
    batch_size: int


def plot_training_results(dataset_name: str, netParams: NetParams, train_accs,
                          val_accs, train_losses, val_losses, is_temporal=True):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(f'Training results on {dataset_name}')

    ax1.plot(range(0, netParams.num_epochs+1, 10), train_accs)
    ax1.plot(range(0, netParams.num_epochs+1, 10), val_accs)
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend(['Train', 'Validation'])

    ax2.plot(range(0, netParams.num_epochs+1, 10), train_losses)
    ax2.plot(range(0, netParams.num_epochs+1, 10), val_losses)
    ax2.grid(True)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch number')
    ax2.legend(['Train', 'Validation'])
    fig.text(
        0.99, 0.01,
        f'HIDDEN_CHANNELS={netParams.hidden_channels}, \
            BATCH_SIZE={netParams.batch_size}',
        horizontalalignment='right', fontsize='xx-small', c='gray')
    if is_temporal:
        plt.savefig(f'./results/result_temporal.pdf')
    else:
        plt.savefig(f'./results/result_{dataset_name}.pdf')
    plt.show()


def get_average_degree(dataset_name, verbose=False):
    '''
    Calculate the average degree of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    degs = []

    for data in track(dataset, description=f'Calculating average degree for {dataset_name}', disable=not verbose):
        deg = degree(data.edge_index[0], data.num_nodes)
        degs.append(deg.mean().item())

    return sum(degs) / len(degs)


def get_average_shortest_path(dataset_name, show_errors=False, verbose=False):
    '''
    Calculate the average shortest path of the graph
    '''
    dataset = TUDataset(root='data/TUDataset', name=dataset_name,
                        use_node_attr=True)
    avg_shortest_paths = []
    num_errors = 0

    for data in track(
        dataset,
        description=f'Calculating average shortest path for {dataset_name}',
        disable=not verbose
    ):
        G = to_networkx(data)
        try:
            avg_shortest_paths.append(nx.average_shortest_path_length(G))
        except:
            print(f'Error in {data}')
            num_errors += 1
            continue
    if show_errors:
        print(f'Number of errors: {num_errors}')
        print(f'Error rate: {num_errors / len(dataset)}')

    return sum(avg_shortest_paths) / len(avg_shortest_paths)


def get_graph_diameter(dataset_name, show_errors=False, verbose=False):
    '''
    Calculate the diameter of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    diameters = []
    num_errors = 0

    for data in track(
            dataset,
            description=f'Calculating graph diameter for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        try:
            diameters.append(nx.diameter(G))
        except:
            print(f'Error in {data}')
            num_errors += 1
            continue
    if show_errors:
        print(f'Number of errors: {num_errors}')
        print(f'Error rate: {num_errors / len(dataset)}')

    return sum(diameters) / len(diameters)


def get_graph_density(dataset_name, verbose=False):
    '''
    Calculate the density of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    densities = []

    for data in track(
            dataset,
            description=f'Calculating graph density for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        densities.append(nx.density(G))

    return sum(densities) / len(densities)


def get_graph_clustering_coefficient(dataset_name, verbose=False):
    '''
    Calculate the clustering coefficient of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    clustering_coefficients = []

    for data in track(
        dataset,
        description=f'Calculating graph clustering coefficient for {
            dataset_name}',
        disable=not verbose

    ):
        G = to_networkx(data)
        clustering_coefficients.append(nx.average_clustering(G))

    return sum(clustering_coefficients) / len(clustering_coefficients)


def get_graph_transitivity(dataset_name, verbose=False):
    '''
    Calculate the transitivity of the graph
    '''

    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    transitivity = []

    for data in track(
            dataset,
            description=f'Calculating graph transitivity for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        transitivity.append(nx.transitivity(G))

    return sum(transitivity) / len(transitivity)


def get_graph_assortativity(dataset_name, verbose=False):
    '''
    Calculate the assortativity of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    assortativity = []

    for data in track(
            dataset,
            description=f'Calculating graph assortativity for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        assortativity.append(nx.degree_assortativity_coefficient(G))

    return sum(assortativity) / len(assortativity)


def get_average_closeness_centrality(dataset_name, verbose=False):
    '''
    Calculate the average closeness centrality of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    closeness_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average closeness centrality for {
                dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        closeness_centralities.append(sum(nx.closeness_centrality(G).values())
                                      / G.number_of_nodes())

    return sum(closeness_centralities) / len(closeness_centralities)


def get_average_betweenness_centrality(dataset_name, verbose=False):
    '''
    Calculate the average betweenness centrality of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    betweenness_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average betweenness centrality for {
                dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        betweenness_centralities.append(sum(nx.betweenness_centrality(G).values())
                                        / G.number_of_nodes())

    return sum(betweenness_centralities) / len(betweenness_centralities)


def get_average_eigenvector_centrality(dataset_name, verbose=False):
    '''
    Calculate the average eigenvector centrality of the graph, given the dataset name
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    eigenvector_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average eigenvector \
                          centrality for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        eigenvector_centralities.append(sum(nx.eigenvector_centrality(
            G, max_iter=100000).values()) / G.number_of_nodes())

    return sum(eigenvector_centralities) / len(eigenvector_centralities)


def wl_1d_color_count(dataset_name, verbose=False):
    '''
    Calculate the average number of coloring in 1WL of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    color_count_sum = []

    for data in track(
            dataset,
            description=f'Calculating average number of coloring in 1WL for {
                dataset_name}',
            disable=not verbose):
        Graph = to_networkx(data)
        # Initialize all nodes with the same color
        colors = {node: 0 for node in Graph.nodes()}
        stable = False

        # Perform color refinement
        while not stable:
            new_colors = {}
            for node in Graph.nodes():
                neighbor_colors = sorted([colors[neighbor] for neighbor
                                          in Graph.neighbors(node)])
                new_colors[node] = f"{colors[node]}|{
                    '|'.join(map(str, neighbor_colors))}"
            label_to_color = {}
            for new_label in new_colors.values():
                if new_label not in label_to_color:
                    label_to_color[new_label] = len(label_to_color)

            # Apply the new coloring
            new_colors = {node: label_to_color[new_colors[node]] for node
                          in Graph.nodes()}

            # Check for stability
            if colors == new_colors:
                stable = True
            colors = new_colors

        color_count_sum.append(len(set(colors.values())))

    return sum(color_count_sum) / len(color_count_sum)


def read_file_to_list(file_path):
    """
    Reads the content from a text file and converts it to a list.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: List of strings, where each string is a line from the file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()
            # Remove newline characters from each line
            content = [line.strip() for line in content]
            return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


class IMDBPreTransform(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
        return data


def load_dataset(dataset_name: str):
    if dataset_name == 'IMDB-BINARY':
        return TUDataset('data/TUDataset', name='IMDB-BINARY',
                         pre_transform=IMDBPreTransform(), forch_reload=True)
    else:
        return TUDataset('data/TUDataset', name=dataset_name, use_node_attr=True)


def get_dataloader(dataset, fold: int, batch_size=64,
                   is_10_fold_validation_enabled=True):
    """
    split the dataset into 10 fold, take one into dataloader and return that dataloader.
    If 10 fold validation is not enabled, return put all data into dataloader
    """

    if is_10_fold_validation_enabled is False:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n = len(dataset) // 10
    test_mask[fold*n:(fold+1)*n] = 1
    val_mask = torch.zeros(len(dataset), dtype=torch.bool)

    if fold == 9:
        val_mask[0:n] = 1
    else:
        val_mask[(fold+1)*n:(fold+2)*n] = 1

    train_dataset = dataset[~test_mask & ~val_mask]
    val_dataset = dataset[val_mask]
    test_dataset = dataset[test_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
