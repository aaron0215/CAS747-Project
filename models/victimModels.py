import torch
import torch.nn as nn
import torch.nn.functional as F
#the paper uses dgl for conv layer setup
#api reference: https://www.dgl.ai/dgl_docs/api/python/nn-pytorch.html
import dgl.nn as dglnn
from utils.metrics import *
import copy

class VictimModels():
    def __init__(self, feature, hid_dimension, num_classes, device, dropout, name, training=False):
        if name == 'GCN':
            self.model = GCN(feature, hid_dimension, num_classes, dropout, training)
        elif name == 'SGC':
            self.model = SGCModel(feature, hid_dimension)
        elif name == 'APPNP':
            self.model = APPNPModel(feature, num_classes, dropout, training)
        elif name == 'GraphSAGE':
            self.model = GraphSAGE(feature, hid_dimension, num_classes, dropout, training)
        else:
            raise Exception("Not support model name: {}".format(name))

        self.model.to(device)

    def train_model(self, g, index_split, epochs, lr, path):
        print("victim model --------------- train start")
        early_stopping = EarlyStopping()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        feature = g.ndata['feature']
        label = g.ndata['label']

        train_index = index_split['train_index']
        val_index = index_split['val_index']

        train_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(g, feature)
            loss = loss_fn(outputs[train_index], label[train_index])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                logits = self.model(g, feature)
                loss = loss_fn(logits[val_index], label[val_index])
                val_loss += loss.item()
                prediction = torch.argmax(logits, 1)
                total += label[val_index].size(0)
                correct += prediction[val_index].eq(label[val_index]).sum().item()

            train_loss /= len(train_index)
            val_loss /= len(val_index)
            val_acc = correct / total
            print(
                f"Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            early_stopping(val_loss, val_acc, epoch, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print("victim model --------------- train end")
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(early_stopping.best_state_dict)
        return self.model

    # re-train process will remove a proportion of nodes to avoid detection
    def train_with_removal(self, g, index_split, epochs, lr, path, uncertainty, proportion=0):
        print("victim model --------------- re-train start")
        early_stopping = EarlyStopping()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        feature = g.ndata['feature']
        label = g.ndata['label']

        train_index = index_split['train_index']
        # remove a proportion of high-uncertainty nodes if set
        if uncertainty is not None and proportion > 0:
            mask = torch.zeros(uncertainty.shape[0], device=feature.device)
            mask[index_split['train_index']] = 1
            train_unc = torch.where(mask == 1, uncertainty, 1)
            # select nodes with lower uncertainty for next training
            # need to sort by ascending order and then remove the last few nodes
            # sorted by descending order and remove the front nodes there might be untrained nodes being removed
            _, sorted_idx = torch.sort(train_unc, descending=False)
            train_index = sorted_idx[:int((1-proportion)*index_split['train_index'].shape[0])]
            print(f"Defense strategy: remove {proportion * index_split['train_index'].shape[0]} nodes before re-training")

        val_index = index_split['val_index']

        train_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(g, feature)
            loss = loss_fn(outputs[train_index], label[train_index])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                logits = self.model(g, feature)
                loss = loss_fn(logits[val_index], label[val_index])
                val_loss += loss.item()
                prediction = torch.argmax(logits, 1)
                total += label[val_index].size(0)
                correct += prediction[val_index].eq(label[val_index]).sum().item()

            train_loss /= len(train_index)
            val_loss /= len(val_index)
            val_acc = correct / total
            print(
                f"Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            early_stopping(val_loss, val_acc, epoch, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print("victim model --------------- re-train end")
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(early_stopping.best_state_dict)
        return self.model

    def test_model(self, g, index_split, path):
        print("victim model --------------- test start")
        self.model.eval()
        feature = g.ndata['feature']
        label = g.ndata['label']
        test_index  = index_split['test_index']
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()
        test_loss = 0
        with torch.no_grad():
            outputs = self.model(g, feature)
            loss = loss_fn(outputs[test_index], label[test_index])
            test_loss += loss.item()
            prediction = torch.argmax(outputs, 1)
            total += label[test_index].size(0)
            correct += prediction[test_index].eq(label[test_index]).sum().item()
            num_classes = label.max()
            sp, eo, delta_sp, delta_eo= calculate_fairness_metrics(prediction[test_index], label[test_index],
                                                                   g.ndata['sensitive'][test_index], num_classes)
            accuracy = correct / total
        print(f"Test Accuracy: {100 * accuracy:.2f}%")
        return accuracy, sp, eo, delta_sp, delta_eo

class EarlyStopping:
    def __init__(self, patience=15, min_delta_factor=0):
        self.patience = patience
        self.min_delta = min_delta_factor
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_acc = 0
        self.best_state_dict = None

    def __call__(self, val_loss, val_acc, epoch, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path)
        elif val_loss > self.best_loss - self.min_delta:
            # print(f"val_loss: {val_loss}")
            # print(f"gap value={(self.best_loss - self.min_delta)}")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path)
            self.counter = 0

# GCN: 2-layer architecture
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes, dropout=0.01, training=False):
        super(GCN, self).__init__()
        # TODO: check documentation, default value for norm doesn't work
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats, norm='both')
        self.conv2 = dglnn.GraphConv(hid_feats, num_classes, norm='both')
        self.dropout = dropout
        self.training = training

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(g, x)
        return x

# GraphSAGE: 2-layer architecture with mean pooling aggregation
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.01, training=False):
        super(GraphSAGE, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, 'mean')
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, 'mean')
        self.dropout = dropout
        self.training = training

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(g, x)
        return x

# APPNP: teleport prob = 0.2, iteration number = 1
class APPNPModel(nn.Module):
    def __init__(self, in_feats, num_classes, dropout=0.01, training=False):
        super(APPNPModel, self).__init__()
        self.lin = nn.Linear(in_feats, num_classes)
        self.conv = dglnn.APPNPConv(1, 0.2)
        self.dropout = dropout
        self.training = training

    def forward(self, g, features):
        x = self.lin(features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv(g, x)
        return x

# SGC: hop number = 1
class SGCModel(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(SGCModel, self).__init__()
        self.conv = dglnn.SGConv(in_feats, num_classes, 1)

    def forward(self, g, features):
        g = self.conv(g, features)
        return g