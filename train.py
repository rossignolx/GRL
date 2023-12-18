from dataset.read_graphs import GraphLogicDataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_add
import networkx as nx

from k_gnn import ThreeMalkin, ThreeGlobal, TwoMalkin
from k_gnn import DataLoader, GraphConv

ORDER_IS_THREE=True

BATCH=20
LAYERS=1
WIDTH=64
EPOCHS=100
FRAC=0.1
LR = 1e-3
class MyPreTransform:
    def __call__(self, data):
        data = ThreeGlobal()(data)
        return data

dataset = GraphLogicDataset(
    "variants/formula1/3Global",
    pre_transform=MyPreTransform()
)
dataset = dataset.shuffle()

if not ORDER_IS_THREE:
    dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
    num_i_2 = dataset.data.iso_type_2.max().item() + 1
    dataset.data.iso_type_2 = F.one_hot(
        dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
else:
    dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
    num_i_3 = dataset.data.iso_type_3.max().item() + 1
    dataset.data.iso_type_3 = F.one_hot(
        dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num = num_i_3 if ORDER_IS_THREE else num_i_2

        self.conv = GraphConv(num, 64)
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(LAYERS - 1):
            self.conv_layers.append(GraphConv(WIDTH, WIDTH))

        self.fc1 = torch.nn.Linear(WIDTH, WIDTH)
        self.fc2 = torch.nn.Linear(WIDTH, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, data):
        if not ORDER_IS_THREE:
            x = F.elu(self.conv(data.iso_type_2, data.edge_index_2))
            for layer in self.conv_layers:
                x = F.elu(layer(x, data.edge_index_2))

            nodes = data.assignment_index_2[0]
            set_ids = data.assignment_index_2[1]
        else:
            x = F.elu(self.conv(data.iso_type_3, data.edge_index_3))
            for layer in self.conv_layers:
                x = F.elu(layer(x, data.edge_index_3))

            nodes = data.assignment_index_3[0]
            set_ids = data.assignment_index_3[1]

        select = torch.index_select(x, 0, set_ids)
        x = scatter_add(select, nodes, dim=0)

        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                for x in module:
                    x.reset_parameters()


def eval(model, loader):
    model.eval()

    correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            total_nodes += data.node_labels.size(0)
            preds = model(data).max(1)[1]
            correct += torch.sum(preds == data.node_labels).item()
    return correct / total_nodes



def train(model: Net, train_loader, test_loader):
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    test_accuracies = []

    for i in range(EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)

            optim.zero_grad()
            pred = model(data)
            loss = F.nll_loss(pred, data.node_labels)
            loss.backward()
            optim.step()

            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        test_accuracy = eval(model, test_loader)
        test_accuracies.append(test_accuracy)

        print("Epoch {}/{}. Avg Train Loss: {:3f}, Test Accuracy: {}".format(i, EPOCHS, avg_loss, test_accuracy))
    return losses, test_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

test_mask = np.full(len(dataset), False)
test_size = int(FRAC * len(dataset))
test_mask[-test_size:] = True

test_dataset = dataset[test_mask]
train_dataset = dataset[~test_mask]

# val_size = int(FRAC * len(rest_dataset))
# val_mask = np.full(len(rest_dataset), False)
# val_mask[val_size:] = True
# val_dataset = rest_dataset[val_mask]
# train_dataset = rest_dataset[~val_mask]

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

train(model, train_loader, test_loader)






