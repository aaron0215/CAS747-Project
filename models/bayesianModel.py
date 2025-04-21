import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
# from dgl.nn import EdgeWeightNorm

# dgl doc: https://www.dgl.ai/dgl_docs/guide/

class BayesianGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, T, theta, device):
        super(BayesianGNN, self).__init__()

        # According to the paper, we take two-layer GCN for estimating the uncertainty
        # directly controls weight metrics
        self.weight1 = nn.Parameter(torch.FloatTensor(in_feats, hid_feats))
        self.weight2 = nn.Parameter(torch.FloatTensor(hid_feats, out_feats))
        # activation function
        self.relu = nn.ReLU()
        # pooing layer
        nn.AvgPool1d(out_feats)
        # init parameters
        self.reset_parameters()
        self.T = T
        self.theta = theta
        self.device = device
        # init two binary masks by creating T tensors of input * output size for each layer
        self.mask1 = self.create_masks(in_feats, hid_feats)
        self.mask2 = self.create_masks(hid_feats, out_feats)
        # self.norm = EdgeWeightNorm('sym')
        self.dropout_rate = 0.005

    def reset_parameters(self):
        # Reinitialize learnable parameters
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight1, gain=gain)
        nn.init.xavier_uniform_(self.weight2, gain=gain)

    def create_masks(self, in_dim, out_dim):
        masks = []
        # sample T sets of vectors of realizations from the Bernoulli distribution according to paper
        for i in range(self.T):
            # follows bernoulli distribution
            mask = torch.bernoulli(torch.full((in_dim, out_dim), 1 - self.theta)).to(self.device)
            masks.append(mask)
        return masks

    def forward(self, g):
        results = []
        # approximate distribution of T sampled model parameters
        for i in range(self.T):
            with g.local_scope():
                # TODO: further read https://www.dgl.ai/dgl_docs/en/2.4.x/api/python/dgl.function.html
                # take feature -> multiply by edge_weight -> send msg -> sum all msg -> store aggregated information in res
                g.update_all(fn.u_mul_e('feature', 'edge_weight', 'msg'), fn.sum('msg', 'res'))
                # multiply first layer with first dropout mask
                # p1 = np.multiply(self.mask1[i], self.weight1)
                p1 = torch.mul(self.weight1, self.mask1[i])
                h = self.relu(torch.mm(g.ndata['res'], p1))
                g.ndata['res'] = h
                # take feature from last layer (aggr stored)
                g.update_all(fn.u_mul_e('res', 'edge_weight', 'msg'), fn.sum('msg', 'output'))
                # multiply second layer with second dropout mask
                # p2 = np.multiply(self.mask2[i], self.weight2)
                p2 = torch.mul(self.weight2, self.mask2[i])
                out = torch.mm(g.ndata['output'], p2)

                results.append(out)

        return results

    def train_model(self, g, lr):
        # follow the technique from paper by Kipf & Welling (2017)
        degree = g.in_degrees().float()
        degree = torch.pow(degree, -0.5)
        src, dst = g.edges()
        edge_weight = degree[src] * degree[dst]
        g.edata['edge_weight'] = edge_weight

        optimizer = torch.optim.AdamW(self.parameters(), lr)
        loss_fn = nn.CrossEntropyLoss()
        label = g.ndata['label']
        label_mask = (label >= 0)

        epochs = 500

        # Training loop
        self.train()
        for epoch in range(epochs):
            outputs = self.forward(g)

            loss = 0
            # computes loss for each of the T predictions and sums them up
            for output in outputs:
                loss += loss_fn(output[label_mask], label[label_mask])

            # use L2 regularization according to [9] paper they referenced
            l2_reg = (self.weight1 ** 2).mean() + (self.weight2 ** 2).mean()
            loss = loss / self.T + (1 - self.dropout_rate) / (2 * self.T) * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.calculate_uncertainty(g)

    def calculate_uncertainty(self, g):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(g)
            outputs = torch.stack(outputs, 0)
            outputs = F.softmax(outputs, 2)
            # estimate uncertainty by calculating the variance of T times
            uncertainty = torch.var(outputs, 0).mean(1)
            return uncertainty
