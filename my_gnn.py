import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from helper import timeSince, plot_training_results, NetParams
from net import GCN
import time


def run(dataset_name, hidden_channels=64, num_epochs=64, batch_size=200): 
    # Define dataset
    root_name = f'data/TUDataset/{dataset_name}'
    dataset = TUDataset(root=root_name, name=dataset_name, use_node_attr=True)
    dataset = dataset.shuffle()
    centered_line = f'Dataset: {dataset}'.center(80-2)
    print('=' * 80 + f'\n={centered_line}=\n' + '=' * 80 + '\n')


    train_dataset_size = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_dataset_size]
    test_dataset = dataset[train_dataset_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)
    # print(f'Model structure: {model}\n\n')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(loader):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        
        return loss_all/len(loader.dataset)

    @torch.no_grad()
    def val(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)


    test_acc = val(test_loader)
    print(f'Initial test accuracy: {test_acc:.4f}')

    # Train model
    train_accs = []
    test_accs = []
    losses = []
    start = time.time()
    for epoch in range(num_epochs+1):
        loss = train(train_loader)
        train_acc = val(train_loader)
        test_acc = val(test_loader)
        if epoch % 10 == 0:
            test_accs.append(test_acc)
            train_accs.append(train_acc)
            losses.append(loss)
            print(f'Epoch: {epoch:03d} ({timeSince(start)}), Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Plot results
    netParams = NetParams(hidden_channels, num_epochs, batch_size)
    plot_training_results(dataset_name, netParams, test_accs, train_accs, losses)

if __name__ == '__main__':
    hidden_channels = 64
    batch_size = 64
    num_epochs = 200
    dataset_name = 'DD'

    run(dataset_name, hidden_channels, num_epochs, batch_size)