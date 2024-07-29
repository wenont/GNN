import matplotlib.pyplot as plt
import networkx as nx
import time
import math
from dataclasses import dataclass


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

def plot_training_results(dataset_name: str, netParams: NetParams, train_accs, val_accs, train_losses, val_losses):
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
    plt.savefig(f'./results/result_{dataset_name}.pdf')
    plt.show()


def get_dataloader(dataset, batch_size=32):
    from torch_geometric.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)