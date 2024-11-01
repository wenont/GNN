import torch
from torch_geometric.loader import DataLoader
from utils import timeSince, plot_training_results, NetParams, TrainParams, load_dataset, setup_wandb
from net import get_model
import time
import wandb


def train(dataset_name,  model_name, trainParams: TrainParams, is_wandb=False):
    """
    This function die the following:
    - Load the dataset, using 80:20 train-val split, without the cross-validation.
    - Train the model
    - Get the training result of the dataset, including the training loss, validation loss, training accuracy, and
    - Plot the results.
    """

    # Define dataset and dataloader
    dataset = load_dataset(dataset_name)

    n = len(dataset) // 10

    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_mask[0:n] = 1
    val_mask = torch.zeros(len(dataset), dtype=torch.bool)
    val_mask[n:2*n] = 1

    train_dataset = dataset[~test_mask & ~val_mask]
    val_dataset = dataset[val_mask]
    test_dataset = dataset[test_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=trainParams.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=trainParams.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=trainParams.batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Define model
    model = get_model(
        model_name=model_name,      
        in_channels=dataset.num_features,
        hidden_channels=trainParams.hidden_size,
        out_channels=dataset.num_classes,
        num_hidden_layers=trainParams.num_hidden_layers,
        dropout=trainParams.dropout
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainParams.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.0001)

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
        loss_all = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
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
    print(f'Initial val loss: {
          val_loss:.4f}, Initial val accuracy: {val_acc:.4f}')

    # Train model
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss, best_val_acc = float('inf'), 0

    start = time.time()
    default_patience = trainParams.patience
    for epoch in range(trainParams.num_epochs+1):
        train_loss = train(train_loader)
        val_loss = val(train_loader)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_accs.append(train_acc)

        if is_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch
            })

        if val_loss < best_val_loss:
            test_acc = test(test_loader)
            best_val_loss = val_loss
            best_test_acc = test_acc
            patience = default_patience
        else:
            patience -= 1
            if patience == 0:
                break
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d} ({timeSince(start)}), Train Loss: {train_loss:.4f}, Val Loss: {
                  val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    print(f'Best validation loss on {dataset_name}: {
          best_val_loss:.4f} with the test accuracy: {best_test_acc:.4f}\n')

    if is_wandb is False:
        print('Plotting the training results...')
        plot_training_results(dataset_name, train_accs, val_accs, train_losses, val_losses)
    return best_val_loss


def hyperparameter_tuning(config=None):
    with wandb.init(config=config):
        config = wandb.config
        trainParams = TrainParams(
            config.hidden_size, config.num_hidden_layers, config.num_epochs, config.batch_size, config.dropout, config.lr, config.default_patience
        )
        train(config.dataset_name, 'GCN', trainParams, is_wandb=True)

def get_generalization_error_from_a_dataset(dataset_name, model_name='GCN', hidden_dim=64, num_epochs=200, batch_size=64, dropout=0.5, lr=0.01, default_patience=20, split=False):
    # Load dataset
    # dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
    # dataset = dataset.shuffle()
    dataset = load_dataset(dataset_name)

    centered_line = f'Dataset: {dataset}'.center(93)
    print('=' * 93 + f'\n{centered_line}\n' + '=' * 93 + '\n')

    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(
        model_name=model_name,
        in_channels=dataset.num_features,
        hidden_channels=hidden_dim,
        out_channels=dataset.num_classes,
        dropout=dropout
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

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
        # print(f'Correct: {correct}, len(loader.dataset): {len(loader.dataset)}')
        return correct / len(loader.dataset)

    best_test_accuracy_list = []
    best_train_accuracy_list = []
    for i in range(10):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, min_lr=0.0001)
        print(f'Run {i+1}')

        test_mask = torch.zeros(len(dataset), dtype=torch.bool)

        n = len(dataset) // 10
        test_mask[i*n:(i+1)*n] = 1
        val_mask = torch.zeros(len(dataset), dtype=torch.bool)

        if i == 9:
            val_mask[0:n] = 1
        else:
            val_mask[(i+1)*n:(i+2)*n] = 1

        train_dataset = dataset[~test_mask & ~val_mask]
        val_dataset = dataset[val_mask]
        test_dataset = dataset[test_mask]

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        val_loss = val(val_loader)
        val_acc = test(val_loader)
        print(f'Initial val loss: {
              val_loss:.4f}, Initial val accuracy: {val_acc:.4f}')

        # Train model
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        best_val_loss, best_test_acc, best_train_acc = float('inf'), 0, 0

        start = time.time()
        patience = default_patience
        for epoch in range(1, num_epochs+1):
            train_loss = train(train_loader)
            val_loss = val(val_loader)
            train_acc = test(train_loader)
            val_acc = test(val_loader)
            scheduler.step(val_loss)

            wandb.log({
                f"{i+1}-Fold train_loss": train_loss,
                f"{i+1}-Fold val_loss": val_loss,
                f"{i+1}-Fold train_acc": train_acc,
                f"{i+1}-Fold val_acc": val_acc
            })

            if val_loss < best_val_loss:
                test_acc = test(test_loader)
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_train_acc = train_acc
                patience = default_patience
            else:
                patience -= 1
                if patience == 0:
                    break
            if epoch % 10 == 0:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                train_accs.append(train_acc)
                print(f'Epoch: {epoch:03d} ({timeSince(start)}), Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        best_test_accuracy_list.append(best_test_acc)
        best_train_accuracy_list.append(best_train_acc)
        print(f'Best validation loss on {dataset_name}: {
              best_val_loss:.4f} with the test accuracy: {best_test_acc:.4f} in the round {i+1}\n')

        # Plot results
        # netParams = NetParams(hidden_channels, num_epochs, batch_size)
        # plot_training_results(dataset_name, netParams, train_accs, val_accs, train_losses, val_losses)

    best_test_accuracy_list = torch.tensor(best_test_accuracy_list)
    best_train_accuracy_list = torch.tensor(best_train_accuracy_list)

    generation_errors = best_train_accuracy_list - best_test_accuracy_list

    print('---------------- Final Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(
        generation_errors.mean(), generation_errors.std()))

    return generation_errors.mean().item(), generation_errors.std().item()


if __name__ == '__main__':
    # sweep_id = setup_wandb()
    # print(sweep_id)
    # sweep_id = 'xd3livh8'
    # wandb.agent(sweep_id, hyperparameter_tuning, count=5, project='bt')


    #   hidden_size=128,
    #   num_hidden_layers=4,
    #   num_epochs=200,
    #   batch_size=64,
    #   dropout=0.5,
    #   lr=0.001,
    #   patience=20
    trainParams = TrainParams(
        128, 3, 200, 64, 0.5, 0.001, 5
    )
    model_name = 'GCN'
    dataset_name = 'DD'
    train(dataset_name, model_name, trainParams)