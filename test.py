import torch
import sys
import os
import dgl
from utils.dataLoader import load_data

# print(f"Python version: {sys.version}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
# print(f"Environment path: {sys.executable}")
# print(f"Site packages location: {os.path.dirname(os.__file__)}/site-packages")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device", device)
graph, index_split = load_data("./data/pokec_z.bin")
print(graph.ndata.keys())
graph = graph.to(device)
in_dim = graph.ndata['feature'].shape[1]
hid_dim = 128
out_dim = max(graph.ndata['label']).item() + 1
label = graph.ndata['label']
print("index_split:",index_split)
print("label:",label)


