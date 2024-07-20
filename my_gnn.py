import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
from helper import timeSince
import time


HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
NUM_EPOCHS = 200
DATASET = 'ENZYMES'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_name = f'data/{DATASET}'
dataset = TUDataset(root=root_name, name=DATASET)
dataset = dataset.shuffle()

train_dataset_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_dataset_size]
test_dataset = dataset[train_dataset_size:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, batch, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=dataset.num_classes,
).to(device)
print(f'Model structure: {model}\n\n')

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start = time.time()
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
fig.suptitle('Training results')

ax1.plot(range(0, NUM_EPOCHS+1, 10), test_accs)
ax1.plot(range(0, NUM_EPOCHS+1, 10), train_accs)
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend(title='Legend', labels=['Test', 'Train'])

ax2.plot(range(0, NUM_EPOCHS+1, 10), losses)
ax2.grid(True)
ax2.set_ylabel('Loss')
ax2.set_xlabel('epoch number')

plt_name = f'./results/result_{DATASET}.pdf'
plt.savefig(plt_name)
plt.show()