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
    def __init__(self, features = 10, output_dim = 2):
        super().__init__()
        
        self.hidden1 = gnn.conv.GraphConv(features, 20)   # GraphConvolution layer1: input dim = #features??, output dim = hidden dim = 20 , ReLu activation, bias = true in og config
        self.relu1 = nn.ReLU()
        self.hidden2 = gnn.conv.GraphConv(20, 20)         # GraphConvolution layer2: input dim = hidden dim = 20,
        self.relu2 = nn.ReLU()
        self.hidden3 = gnn.conv.GraphConv(20, 20)         # GraphConvolution layer3: 
        self.relu3 = nn.ReLU()
        
        self.lin = nn.Linear(20*2, output_dim)                    # fully connected layer(Dense) => nn.linear: input dim = hidden dim * 2 due to concat of pooling, output dim = output dim = classes


    def forward(self, x, edge_index, batch = None):             # edge weights missing ; x, edge_index = feature_tensor, transformed adjs_tensor
        # encoding net
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype= torch.long)   # all nodes belong to the same graph 0
            
        out = self.hidden1(x, edge_index)
        out = self.relu1(out)
        
        out = self.hidden2(out, edge_index)
        out = self.relu2(out)
        
        out = self.hidden3(out, edge_index)
        out = self.relu3(out)

        # TODO: node emdeggins (out hidden3) need to be returned seperatly for MLP input
        
        # classifier net
        maxP = gnn.pool.global_max_pool(out, batch)
        meanP = gnn.pool.global_mean_pool(out, batch)
        
        out = self.lin(torch.cat([maxP, meanP], -1))            # concat pooled embeddings to get better representation of graph to classify => input_dim lin = 2* size of features?
        
        return out
    

# TODO: GCN for node classification