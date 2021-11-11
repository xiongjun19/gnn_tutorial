# coding=utf8

# import torch_geometric as tg
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    # def forward(self, data):
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main(model_path):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    print(dataset[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = GCN(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train(model, data, optimizer)
    acc = eval_model(model, data)
    print(f"accuray is {acc}")
    torch.save(model, model_path)


def train(model, data, optimizer):
    model.train()
    epochs = 800
    for epoch in tqdm(range(epochs), f'training epoch '):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def eval_model(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (data.y[data.test_mask] == pred[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc


if __name__ == '__main__':
    main('gcn_cora.pth')
