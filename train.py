import torch
import os
import torch.nn.functional as F
import argparse
import sys
from dataset.read_graphs import GraphLogicDataset

from torch_scatter import scatter_add
from k_gnn import ThreeMalkin, ThreeGlobal, TwoMalkin, ConnectedThreeMalkin
from k_gnn import DataLoader, GraphConv
from utils.utils import get_transform

parser = argparse.ArgumentParser()
parser.add_argument('-layers', type=int, default=2)   # Number of GNN layers
parser.add_argument('-method', type=str, default='ThreeMalkin')    # Dataset being used
parser.add_argument('-repeat', type=int, default=1)
args = parser.parse_args()

METHOD = args.method
transform = get_transform(METHOD)

ORDER_IS_THREE = 'Three' in args.method
LAYERS = args.layers
REPEAT = args.repeat

FORMULA='formula1'
BATCH=20
WIDTH=64
EPOCHS=500
LR = 1e-3

LOG_DIR='logs/kgnn'
LOG_FILE=f'{METHOD}-{FORMULA}-epochs-{EPOCHS}-width-{WIDTH}-lr-{LR}-repeat-{REPEAT}-layers-{LAYERS}.txt'
print("Saving to {}".format(LOG_FILE))

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)
log_file_handler = open(LOG_PATH, 'w')

train_dataset = GraphLogicDataset(
    "variants/{}/train/{}".format(FORMULA, METHOD),
    pre_transform=transform
)
val_dataset = GraphLogicDataset(
    "variants/{}/val/{}".format(FORMULA, METHOD),
    pre_transform=transform
)
test_dataset = GraphLogicDataset(
    "variants/{}/test/{}".format(FORMULA, METHOD),
    pre_transform=transform
)


if not ORDER_IS_THREE:
    train_dataset.data.iso_type_2 = torch.unique(train_dataset.data.iso_type_2, True, True)[1]
    num_i_2 = train_dataset.data.iso_type_2.max().item() + 1
    train_dataset.data.iso_type_2 = F.one_hot(
        train_dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

    val_dataset.data.iso_type_2 = torch.unique(val_dataset.data.iso_type_2, True, True)[1]
    num_i_2 = val_dataset.data.iso_type_2.max().item() + 1
    val_dataset.data.iso_type_2 = F.one_hot(
        val_dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
    
    test_dataset.data.iso_type_2 = torch.unique(test_dataset.data.iso_type_2, True, True)[1]
    num_i_2 = test_dataset.data.iso_type_2.max().item() + 1
    test_dataset.data.iso_type_2 = F.one_hot(
        test_dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
else:
    train_dataset.data.iso_type_3 = torch.unique(train_dataset.data.iso_type_3, True, True)[1]
    num_i_3 = train_dataset.data.iso_type_3.max().item() + 1
    train_dataset.data.iso_type_3 = F.one_hot(
        train_dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

    val_dataset.data.iso_type_3 = torch.unique(val_dataset.data.iso_type_3, True, True)[1]
    num_i_3 = val_dataset.data.iso_type_3.max().item() + 1
    val_dataset.data.iso_type_3 = F.one_hot(
        val_dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

    test_dataset.data.iso_type_3 = torch.unique(test_dataset.data.iso_type_3, True, True)[1]
    num_i_3 = test_dataset.data.iso_type_3.max().item() + 1
    test_dataset.data.iso_type_3 = F.one_hot(
        test_dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

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


def train(model: Net, train_loader, val_loader):
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    val_accuracies = []

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

        val_accuracy = eval(model, val_loader)
        val_accuracies.append(val_accuracy)

        log_text = "Epoch {}/{}. Avg Train Loss: {:3f}, Val Accuracy: {}".format(i, EPOCHS, avg_loss, val_accuracy)
        print(log_text)
        log_file_handler.write(log_text + "\n")
    return losses, val_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

train(model, train_loader, val_loader)
test_accuracy = eval(model, test_loader)

text = "Final Test Accuracy: {}".format(test_accuracy)
print(text)
log_file_handler.write(text)
log_file_handler.close()






