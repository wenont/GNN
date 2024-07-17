import torch
from torch import nn

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import matplotlib.pyplot as plt


HIDDEN_CHANNELS = 64
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()

train_dataset = dataset[:540]
test_dataset = dataset[540:]

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
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.classifier(x)
        return x

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=dataset.num_classes,
).to(device)
print(f'model: {model}')

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def train(loader):
    model.train()
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        train_acc = correct / len(loader.dataset)
        return loss, train_acc

def test(loader):
     model.eval()
     correct = 0
     for data in loader:
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)


test_acc = test(test_loader)
print(f'Initial test accuracy: {test_acc:.4f}')

test_accs = []
losses = []
for epoch in range(401):
    test_acc = test(test_loader)
    loss, train_acc = train(train_loader)
    if epoch % 10 == 0:
        test_accs.append(test_acc)
        losses.append(loss)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

plt.plot(range(0, 401, 10), test_accs, color = "y", marker = "o")
plt.xlabel('epoch number') 
plt.ylabel('test_accs')
plt.yscale('linear')
plt.grid(True)
plt.title('Test accuracy')
plt.savefig('test_accuracy.pdf')
plt.show()

# plt.plot(range(0, 401, 10), losses, color = "y", marker = "o")
# plt.xlabel('epoch number') 
# plt.ylabel('loss')
# plt.yscale('linear')
# plt.grid(True)
# plt.title('Loss')
# plt.show()