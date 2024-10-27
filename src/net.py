import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_hidden_layers=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers)])
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout


    def forward(self, x, edge_index, batch, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight).relu()
 
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight).relu()

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.fc1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_hidden_layers=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, batch, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.fc1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
    def reset_parameters(self):
        for (_, module) in self._modules.items():
            module.reset_parameters()

def get_model(model_name: str, in_channels: int, hidden_channels: int, out_channels: int, num_hidden_layers=4, dropout=0.5):
    if model_name == 'GCN':
        return GCN(in_channels, hidden_channels, out_channels, num_hidden_layers, dropout)
    elif model_name == 'GAT':
        return GAT(in_channels, hidden_channels, out_channels)
    else:
        raise ValueError('Model name not supported')