import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from utils import timeSince, plot_training_results, NetParams
from net import GAT
import time


def get_training_result(dataset_name, hidden_channels=64, num_epochs=200, batch_size=64): 
    """
    This function die the following:
    - Load the dataset, using 80:20 train-val split, without the cross-validation.
    - Train the model
    - Get the training result of the dataset, including the training loss, validation loss, training accuracy, and
    - Plot the results.
    """
    
    # Define dataset
    dataset_num_features = int()
    dataset_num_classes = int()
    if 'tox21' not in dataset_name.lower(): # by default
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
        dataset = dataset.shuffle()
        centered_line = f'Dataset: {dataset}'.center(93)
        print('=' * 93 + f'\n{centered_line}\n' + '=' * 93 + '\n')


        train_dataset_size = int(len(dataset) * 0.8)
        train_dataset = dataset[:train_dataset_size]
        val_dataset = dataset[train_dataset_size:]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        dataset_num_features = dataset.num_features
        dataset_num_classes = dataset.num_classes
    else:
        train_dataset = TUDataset(root='data/TUDataset', name=dataset_name+'_training', use_node_attr=True)
        val_dataset = TUDataset(root='data/TUDataset', name=dataset_name+'_evaluation', use_node_attr=True)
        test_dataset = TUDataset(root='data/TUDataset', name=dataset_name+'_testing', use_node_attr=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        dataset_num_features = train_dataset.num_features
        dataset_num_classes = train_dataset.num_classes

    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = GAT(
        in_channels=dataset_num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset_num_classes,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(loader):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.long())
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        
        return loss_all/len(loader.dataset)

    @torch.no_grad()
    def val(loader):
        model.eval()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.long())
            loss_all += loss.item() * data.num_graphs
        return loss_all/len(loader.dataset)
    
    @torch.no_grad()
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)


    val_loss = val(val_loader)
    val_acc = test(val_loader)
    print(f'Initial val loss: {val_loss:.4f}, Initial val accuracy: {val_acc:.4f}')

    # Train model
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss, best_val_acc = float('inf'), 0

    start = time.time()
    for epoch in range(num_epochs+1):
        train_loss = train(train_loader)
        val_loss = val(train_loader)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
        if epoch % 10 == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            train_accs.append(train_acc)
            print(f'Epoch: {epoch:03d} ({timeSince(start)}), Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    print(f'Best validation loss on {dataset_name}: {best_val_loss:.4f} with the accuracy: {best_val_acc:.4f}\n')
    # Plot results
    netParams = NetParams(hidden_channels, num_epochs, batch_size)
    plot_training_results(dataset_name, netParams, train_accs, val_accs, train_losses, val_losses)

if __name__ == '__main__':

    HIDDEN_CHANNELS = 64
    MAX_NUM_EPOCHS = 200
    BATCH_SIZE = 64
    # DATASET_NAME = 'DD'
    DATASET_NAME = 'PROTEINS_full'
    
    print(DATASET_NAME)
    get_training_result(DATASET_NAME)