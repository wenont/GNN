import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from helper import timeSince
from net import GCN
import time


HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
NUM_EPOCHS = 200
DATASET = 'DD'

 
root_name = f'data/TUDataset/{DATASET}'
dataset = TUDataset(root=root_name, name=DATASET, use_node_attr=True)
dataset = dataset.shuffle()
print(f'Data: {dataset[0]}')

train_dataset_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_dataset_size]
test_dataset = dataset[train_dataset_size:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=dataset.num_classes,
).to(device)
print(f'Model structure: {model}\n\n')

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
    
    return loss_all/len(train_loader.dataset)

@torch.no_grad()
def test(loader):
     model.eval()
     correct = 0
     for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)


test_acc = test(test_loader)
print(f'Initial test accuracy: {test_acc:.4f}')

train_accs = []
test_accs = []
losses = []
start = time.time()
for epoch in range(NUM_EPOCHS+1):
    loss = train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 10 == 0:
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        losses.append(loss)
        print(f'Epoch: {epoch:03d} ({timeSince(start)}), Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f'Training results on {DATASET}')

ax1.plot(range(0, NUM_EPOCHS+1, 10), test_accs)
ax1.plot(range(0, NUM_EPOCHS+1, 10), train_accs)
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend(['Test', 'Train'])

ax2.plot(range(0, NUM_EPOCHS+1, 10), losses)
ax2.grid(True)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch number')
fig.text(
        0.99, 0.01, 
        f'HIDDEN_CHANNELS={HIDDEN_CHANNELS}, BATCH_SIZE={BATCH_SIZE}', 
        horizontalalignment='right', fontsize='xx-small', c='gray')
plt.savefig(f'./results/result_{DATASET}.pdf')
plt.show()
