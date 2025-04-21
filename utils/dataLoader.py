import torch
import dgl

def load_data(path):
    print(f"loading {path}")
    glist, label_dict = dgl.load_graphs(path)
    graph = glist[0]
    # print(graph.ndata.keys())
    idx_train = torch.where(graph.ndata['train_index'])[0]
    idx_val = torch.where(graph.ndata['val_index'])[0]
    idx_test = torch.where(graph.ndata['test_index'])[0]
    index_split = {'train_index': idx_train,
                   'val_index': idx_val,
                   'test_index': idx_test}
    return graph, index_split