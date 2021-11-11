# coding=utf8

import dgl
import dgl.data as dg_data
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        # h = F.dropout(h)
        h = self.conv2(g, h)
        return h


def main(model_path):
    ds, g = get_graph()
    model = GCN(g.ndata['feat'].shape[1], 16, ds.num_classes)
    train(g, model, model_path)


def get_graph():
    ds = dg_data.CoraGraphDataset()
    print("Number of categories: ", ds.num_classes)
    g = ds[0]
    print("Node features")
    print(g.ndata)
    print('Egde features')
    print(g.edata)
    return ds, g


def train(g, model, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    best_val_acc = 0.
    best_test_acc = 0.
    epochs = 200
    g = g.to(device)
    in_feat = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    train_mask = g.ndata['train_mask'].to(device)
    val_mask = g.ndata['val_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in tqdm(range(epochs), "iterating training"):
        logits = model(g, in_feat)
        preds = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = calc_metric(preds, labels, train_mask)
        val_acc = calc_metric(preds, labels, val_mask)
        test_acc = calc_metric(preds, labels, test_mask)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model, model_path)
        print('In epoch {}, loss: {:.4f}, train acc {:.4f} \
                val acc {:.4f} test acc {:.4f}'.format(epoch, loss.item(),
                    train_acc, best_val_acc, best_test_acc))

def calc_metric(preds, labels, mask):
    acc = (preds[mask] == labels[mask]).float().mean()
    return acc



if __name__ == '__main__':
    main("gcn_dgl_cora.pth")

