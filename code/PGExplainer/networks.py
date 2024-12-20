import torch
import torch.nn as nn
import torch_geometric.nn as gnn

# TODO: PGExplainer MLP with one hidden layer?
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected (#input(node: 60, graph: 40), 64)
        # ReLu
        # Linear Fully connected (20, 1)

    def forward(self, x):
        return self.model(x)
    

# Three layer GNN(GCN) for Graph Classification based on PGExplainer paper/Source code
class GraphGNN(nn.Module):
    def __init__(self, features = 10, labels = 2):
        super(GraphGNN, self).__init__()
        
        # TODO: Try GCNConv
        self.hidden1 = gnn.SAGEConv(features, 20)   # GraphConvolution layer1: input dim = #features??, output dim = hidden dim = 20 , ReLu activation, bias = true in og config
        nn.init.xavier_uniform_(self.hidden1.lin_r.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_l.weight)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden2 = gnn.SAGEConv(20, 20)         # GraphConvolution layer2: input dim = hidden dim = 20,
        nn.init.xavier_uniform_(self.hidden1.lin_r.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_l.weight)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden3 = gnn.SAGEConv(20, 20)         # GraphConvolution layer3: 
        nn.init.xavier_uniform_(self.hidden1.lin_r.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_l.weight)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)               # Not used in PGExplainer
        
        self.lin = nn.Linear(20*2, labels)                    # fully connected layer(Dense) => nn.linear: input dim = hidden dim * 2 due to concat of pooling, output dim = output dim = classes


    def forward(self, x, edge_index, batch = None):             # edge weights missing ; x, edge_index = feature_tensor, transformed adjs_tensor
        # Encoding net
        emb = self.getNodeEmbeddings(x, edge_index)
        
        # Classifier net
        maxP = gnn.pool.global_max_pool(emb, batch)
        meanP = gnn.pool.global_mean_pool(emb, batch)
        
        out = self.lin(torch.cat([maxP, meanP], -1))            # concat pooled embeddings to get better representation of graph to classify => input_dim lin = 2* size of features?
        
        return out
    

    # Node embeddings need to be returned seperatly to the MLP and are also part of the forward pass
    def getNodeEmbeddings(self, x, edge_index):
        emb1 = self.hidden1(x, edge_index)
        #emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb1 = self.relu1(emb1)
        emb1 = self.dropout1(emb1)
        
        emb2 = self.hidden2(emb1, edge_index)
        #emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        emb2 = self.relu2(emb2)
        emb2 = self.dropout1(emb2)
        
        emb3 = self.hidden3(emb2, edge_index)
        #emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
        emb3 = self.relu3(emb3)
        emb3 = self.dropout1(emb3)

        return emb3
    


class NodeGNN(nn.Module):
    def __init__(self, features, labels):
        super(NodeGNN, self).__init__()
        
        # TODO: ADD BNorm, used in og code ??

        self.hidden1 = gnn.GraphConv(features, 20)
        nn.init.xavier_uniform_(self.hidden1.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_root.weight)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.ReLU()
        #self.dropout1 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden2 = gnn.GraphConv(20, 20)
        nn.init.xavier_uniform_(self.hidden2.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden2.lin_root.weight)
        self.bn2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden3 = gnn.GraphConv(20, 20)
        nn.init.xavier_uniform_(self.hidden3.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden3.lin_root.weight)
        self.bn3 = nn.BatchNorm1d(20)
        self.relu3 = nn.ReLU()
        #self.dropout3 = nn.Dropout(p=0.1)               # Not used in PGExplainer
        
        self.lin = nn.Linear(20*3, labels)                  # outputs of three hidden layers are by default concatenated in PGE Node classification and input into lin layer

    def forward(self, x, edge_index):
        # Encoding net
        emb = self.getNodeEmbeddings(x, edge_index)
        
        # Classifier net, no pooling in node network
        out = self.lin(emb)

        return out

    def getNodeEmbeddings(self, x, edge_index):
        embs = []

        emb1 = self.hidden1(x, edge_index)
        #emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        #emb1 = self.bn1(emb1)
        emb1 = self.relu1(emb1)
        #emb1 = self.dropout1(emb1)
        
        emb2 = self.hidden2(emb1, edge_index)
        #emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        #emb2 = self.bn2(emb2)
        emb2 = self.relu2(emb2)
        #emb2 = self.dropout1(emb2)
        
        emb3 = self.hidden3(emb2, edge_index)
        #emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
        #emb3 = self.bn3(emb3)
        emb3 = self.relu3(emb3)
        #emb3 = self.dropout1(emb3)

        embs = torch.cat([emb1, emb2, emb3], -1)
        
        return embs