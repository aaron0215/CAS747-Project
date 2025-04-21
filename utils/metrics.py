import torch

def calculate_fairness_metrics(predictions, labels, sensitive_attr, num_classes):
    # index for each sensitive group
    index_1 = torch.where(sensitive_attr == 1)[0]
    index_0 = torch.where(sensitive_attr == 0)[0]

    SP_list = []
    EO_list = []
    # print(f"number of class = {num_classes}")
    for y in range(num_classes + 1):
        # Statistical Parity (SP)
        # get the nodes are predicted as class y in each sensitive group
        pred_1 = torch.where(predictions[index_1] == y)[0]
        pred_0 = torch.where(predictions[index_0] == y)[0]

        # P(ˆy = y|s = 1) = size(prediction of group1) / size(all group1)
        p_1 = (pred_1.shape[0] / index_1.shape[0])
        p_0 = (pred_0.shape[0] / index_0.shape[0])
        # calculate P(ˆy = y|s = 0) − P(ˆy = y|s = 1)
        sp = p_1 - p_0
        SP_list.append(sp)

        # Equal Opportunity (EO)
        # get indicis of this label from each group
        label_1 = torch.where(labels[index_1] == y)[0]
        label_0 = torch.where(labels[index_0] == y)[0]

        # there are invalid labels
        if label_1.shape[0] <= 0 or label_0.shape[0] <= 0:
            eo = 0
            EO_list.append(eo)
            continue

        # get the nodes among each sensitive group that are truly class y and predicated as class y as well
        pred_1 = torch.where(predictions[index_1][label_1] == y)[0]
        pred_0 = torch.where(predictions[index_0][label_0] == y)[0]

        # calculate the fraction of nodes that are correctly predicted
        # P(ˆy=y | y = y, s = 1) = size(prediction) / size(all label y)
        p_1 = pred_1.shape[0] / label_1.shape[0]
        p_0 = pred_0.shape[0] / label_0.shape[0]
        # P(ˆy = y|y = y, s = 1) - P(ˆy = y|y = y, s = 0)
        eo = p_1 - p_0
        EO_list.append(eo)

    # ∆SP = E|P(ˆy = y|s = 0) − P(ˆy = y|s = 1)|
    delta_sp = torch.mean(torch.abs(torch.tensor(SP_list))).item()

    # ∆EO = E|P(ˆy = y|y = y, s = 0) − P(ˆy = y|y = y, s = 1)|
    delta_eo = torch.mean(torch.abs(torch.tensor(EO_list))).item()

    return SP_list, EO_list, delta_sp, delta_eo