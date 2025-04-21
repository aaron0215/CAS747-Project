import random
from models.bayesianModel import *
from models.victimModels import *

class NodeInjection:
    def __init__(self, node_budget, num_target):
        self.node_budget = node_budget
        self.num_target = num_target

    def inject(self, g, uncertainty, ratio):
        # select nodes with the top k% model uncertainty U in each sensitive group as the target nodes
        # generate two uncertainty groups based on sensitive attribute
        sensitive_attr = g.ndata['sensitive']
        label = g.ndata['label']

        # mask_1 = (sensitive_attr == 1)
        # mask_0 = (sensitive_attr == 0)
        mask_1 = torch.logical_and(sensitive_attr == 1, label >= 0) # "only a part of the nodes have the label information"
        mask_0 = torch.logical_and(sensitive_attr == 0, label >= 0)
        mask_1 = mask_1.to(g.device)
        mask_0 = mask_0.to(g.device)

        uncertainty_group_1 = torch.zeros(g.num_nodes(), device=g.device)
        uncertainty_group_0 = torch.zeros(g.num_nodes(), device=g.device)

        uncertainty_group_1[mask_1] = uncertainty[mask_1]
        uncertainty_group_0[mask_0] = uncertainty[mask_0]

        count_1 = mask_1.sum().item()
        count_0 = mask_0.sum().item()

        # sort nodes by uncertainty
        sort_mask_1 = torch.where(mask_1, uncertainty_group_1, 0)
        sort_mask_0 = torch.where(mask_0, uncertainty_group_0, 0)
        _, index_1 = torch.sort(sort_mask_1, descending=True)
        _, index_0 = torch.sort(sort_mask_0, descending=True)

        # select top k% highest-uncertainty nodes
        selected_index_1 = index_1[:int(ratio * count_1)]
        selected_index_0 = index_0[:int(ratio * count_0)]
        # print(f"ratio = {ratio}, mask_1 = {len(mask_1)}, count1 = {count_1}")

        # inject nodes accordingly, set attribute to invalid
        g.add_nodes(self.node_budget)
        g.ndata['label'][-self.node_budget:] = -1
        g.ndata['sensitive'][-self.node_budget:] = -1

        # evenly inject nodes by splitting the index from half
        inject_num = self.node_budget // 2
        num_nodes = g.num_nodes()

        # source node is the index of all the targeted nodes
        # randomly select d target nodes from both group 1 and 0
        src_node_list = []
        for _ in range(inject_num):
            indices = random.sample(range(len(selected_index_1)), self.num_target)
            src_node_list.append(selected_index_1[indices])
        for _ in range(self.node_budget - inject_num):
            indices = random.sample(range(len(selected_index_0)), self.num_target)
            src_node_list.append(selected_index_0[indices])
        src_nodes = torch.cat(src_node_list)

        # destination node is the index of all the injected nodes.
        # injected nodes are added to the end of g, calculate indices based on node number
        index_injected_nodes = torch.arange(num_nodes - self.node_budget, num_nodes, device=g.device)

        # create 2D tensor and transpose it to alight each injected node with its target node
        injected_nodes_map = index_injected_nodes.repeat(self.num_target, 1)
        adjusted_nodes_map = torch.t(injected_nodes_map)
        dst_nodes = adjusted_nodes_map.flatten()

        src_nodes = src_nodes.to(g.device)
        dst_nodes = dst_nodes.to(g.device)
        # print(f"src_nodes size: {len(src_nodes)}, dest_nodes size: {len(dst_nodes)}")
        # connect
        g.add_edges(src_nodes, dst_nodes)
        g.add_edges(dst_nodes, src_nodes)

        return g

class FeatureOptimize(nn.Module):
    def __init__(self, g, in_feature, hid_dimension, num_classes, node_budget):
        super(FeatureOptimize, self).__init__()
        # use two-layer GCN model as surrogate model
        self.model = GCN(in_feature, hid_dimension, num_classes)

        feature = g.ndata['feature']
        # get feature bounds
        self.lower_bound = torch.min(feature, 0)[0].repeat(node_budget, 1)
        self.upper_bound = torch.max(feature, 0)[0].repeat(node_budget, 1)
        # initialize learnable feature matrix for injected nodes
        self.feature = nn.Parameter(torch.zeros(node_budget, in_feature).normal_(mean=0.5, std=0.5))

        self.node_budget = node_budget

    def forward(self, g):
        # combine original features with the features of injected nodes
        combined_features = torch.cat((g.ndata["feature"][:-self.node_budget], self.feature), dim=0)
        return self.model(g, combined_features)

    def optimize(self, g, index_split, lr, alpha, beta, max_iter, max_steps):
        train_index = index_split['train_index']
        label = g.ndata['label']
        sensitive = g.ndata['sensitive']

        label_train = g.ndata['label'][train_index]
        index_group1 = torch.where(sensitive[train_index] == 1)[0]
        # print("index group1=",index_group1)
        index_group0 = torch.where(sensitive[train_index] == 0)[0]

        # apply L2 regularization
        # optimizer for surrogate model, working on learnable parameters
        optimizer_GCN = torch.optim.Adam(self.model.parameters(), lr)
        # optimizer for injected nodes feature, working on feature optimization
        optimizer_Feature = torch.optim.Adam([self.feature], lr)

        # employ cross entropy loss
        loss_fn = nn.CrossEntropyLoss()

        # start training process
        for batch in range(max_iter):
            # train surrogate model
            for run in range(max_steps):
                logits = self(g)
                loss = loss_fn(logits[train_index], label[train_index])
                optimizer_GCN.zero_grad()
                loss.backward()
                optimizer_GCN.step()

            # train injected nodes
            for run in range(max_steps):
                logits = self(g)
                logits_train = logits[train_index]

                # set cross-entropy loss as classification loss (LCE)
                l_ce = loss_fn(logits_train, label_train)

                # calculate statistical parity loss (LSP)
                # the paper uses mean prediction to calculate
                mean_sp_group1 = torch.mean(logits_train[index_group1], 0)
                mean_sp_group0 = torch.mean(logits_train[index_group0], 0)
                # according to the equation ‖V0 - V1‖²₂, apply mse loss function here
                l_sp = F.mse_loss(mean_sp_group0, mean_sp_group1)

                # calculate equal opportunity loss (LEO)
                # create a mapping between label value and output mean
                mean_eo_group1 = torch.zeros(label_train.max() + 1)
                mean_eo_group0 = torch.zeros(label_train.max() + 1)
                # push to gpu to solve index out of boundary error
                mean_eo_group1 = mean_eo_group1.to(g.device)
                mean_eo_group0 = mean_eo_group0.to(g.device)
                # calculate output of each node on each label, iterate over all label values
                for i in range(label_train.max() + 1):
                # for i in torch.unique(label_train):
                    label_index_1 = torch.where(label_train[index_group1] == i)[0]
                    label_index_0 = torch.where(label_train[index_group0] == i)[0]
                    # calculate labels existed for both attribute
                    if len(index_group1) > 0 and len(index_group0) > 0:
                        mean_1 = torch.mean(logits_train[index_group1][label_index_1], 0)
                        mean_0 = torch.mean(logits_train[index_group0][label_index_0], 0)
                        mean_eo_group1[i] = mean_1[i]
                        mean_eo_group0[i] = mean_0[i]
                l_eo = F.mse_loss(mean_eo_group0, mean_eo_group1)

                # calculate feature constraint loss (LCF)
                # injected nodes are evenly injected into g, so calculate it with two half parts
                first_half = torch.mean(self.feature[:self.node_budget // 2])
                second_half = torch.mean(self.feature[self.node_budget // 2:])
                l_cf = F.mse_loss(first_half, second_half)

                # Total loss - Note the negative signs for fairness losses
                # We want to maximize unfairness (SP, EO differences)
                overall_loss = l_ce + alpha * -l_cf + beta * -(l_sp + l_eo)

                optimizer_Feature.zero_grad()
                overall_loss.backward()
                optimizer_Feature.step()

                # Clamp feature between min and max
                self.feature.data = torch.clamp(self.feature.data, self.lower_bound, self.upper_bound)

        # For discrete features, round to nearest integer
        g.ndata['feature'][-self.node_budget:] = torch.round(self.feature).detach()
        return g

class NIFA:
    def __init__(self, g, feature, hid_dimension, num_classes, device, T, theta, node_budget, edge_budget):
        self.bayesian_gnn = BayesianGNN(feature, hid_dimension, num_classes,T, theta, device).to(device)
        self.node_injector = NodeInjection(node_budget, edge_budget)
        self.feature_optimizer = FeatureOptimize(g, feature, hid_dimension, num_classes, node_budget).to(device)

    def attack(self, g, index_split, lr, ratio, max_iter, max_steps, alpha, beta):
        # process: train bayesian model to get uncertainty -> inject nodes -> train again to optimize features
        uncertainty = self.bayesian_gnn.train_model(g, lr)
        g = self.node_injector.inject(g, uncertainty, ratio)
        g = self.feature_optimizer.optimize(g, index_split, lr, alpha, beta, max_iter, max_steps)
        return g, uncertainty