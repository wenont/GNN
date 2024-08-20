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

