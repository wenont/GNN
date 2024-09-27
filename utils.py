import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
import networkx as nx
import time
import math
from dataclasses import dataclass
from torch_geometric.datasets import TUDataset
from tqdm import tqdm



def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
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

def plot_training_results(dataset_name: str, netParams: NetParams, train_accs, val_accs, train_losses, val_losses, is_temporal=True):
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
            f'HIDDEN_CHANNELS={netParams.hidden_channels}, BATCH_SIZE={netParams.batch_size}', 
            horizontalalignment='right', fontsize='xx-small', c='gray')
    if is_temporal:
        plt.savefig(f'./results/result_temporal.pdf')
    else:
        plt.savefig(f'./results/result_{dataset_name}.pdf')
    plt.show()

def get_average_degree(dataset_name):
    dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
    degs = []

    for data in dataset:
        deg = degree(data.edge_index[0], data.num_nodes)
        degs.append(deg.mean().item())    

    return sum(degs) / len(degs)

def get_average_shortest_path(dataset_name, show_errors=False):
    dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
    avg_shortest_paths = []

    num_errors = 0

    for data in dataset:
        G = to_networkx(data)
        try:
            avg_shortest_paths.append(nx.average_shortest_path_length(G))
        except:
            num_errors += 1
            continue
    if show_errors:
        print(f'Number of errors: {num_errors}')
        print(f'Error rate: {num_errors / len(dataset)}')
            
    return sum(avg_shortest_paths) / len(avg_shortest_paths)

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
    if dataset_name is 'IMDB-BINARY':
        return TUDataset('data/TUDataset', name='IMDB-BINARY', pre_transform=IMDBPreTransform(), forch_reload=True)
    else:
        return TUDataset('data/TUDataset', name=dataset_name, use_node_attr=True)

def get_dataloader(dataset, fold: int, batch_size=64, is_10_fold_validation_enabled=True):
    """
    split the dataset into 10 fold, take one into dataloader and return that dataloader. If 10 fold validation is not enabled, return put all data into dataloader
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader