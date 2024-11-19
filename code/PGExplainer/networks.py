import torch
import torch.nn as nn
import torch_geometric.nn as gnn

# TODO: MLP with one hidden layer?
"""class MLP(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return self.model(x)"""
    

# Three layer GNN   (GCN) for Graph Classification based on PGExplainer Source code
class GraphGNN(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        
        self.hidden1 = gnn.conv.GraphConv(features, 20)   # GraphConvolution layer1: input dim = #features??, output dim = hidden dim = 20 , ReLu activation, bias = true in og config
        self.relu1 = nn.ReLU()
        self.hidden2 = gnn.conv.GraphConv(20, 20)         # GraphConvolution layer2: input dim = hidden dim = 20,
        self.relu2 = nn.ReLU()
        self.hidden3 = gnn.conv.GraphConv(20, 20)         # GraphConvolution layer3: 
        self.relu3 = nn.ReLU()
        
        self.lin = nn.Linear(20, output_dim)                    # fully connected layer(Dense) => nn.linear: input dim = hidden dim, output dim = output dim = classes?


    def forward(self, x, edge_index, batch = None):             # edge weights missing ; x, edge_index = feature_tensor, adjs_tensor?
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype= torch.long)   # all nodes belong to the same graph 0
            
        out = self.hidden1(x, edge_index)
        out = self.relu1(out)
        
        out = self.hidden2(out, edge_index)
        out = self.relu2(out)
        
        out = self.hidden3(out, edge_index)
        out = self.relu3(out)
        
        maxP = gnn.pool.global_max_pool(out, batch)               # max pool
        meanP = gnn.pool.global_mean_pool(out, batch)             # mean pool
        
        out = self.lin(torch.cat([maxP, meanP], -1))            # concat embeddings to get better representation of graph to classify => input_dim lin = 2* size of features?
        
        return out
    

# TODO: GCN for node classification