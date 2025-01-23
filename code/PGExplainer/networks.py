import torch
import torch.nn as nn
import torch_geometric.nn as gnn

# TODO: PGExplainer MLP with one hidden layer?      Input: Concatenated node embeddings for each edge (graph), conc. node embeddings and embedding of node to be predicted? (node)
class MLP(nn.Module):
    def __init__(self, features = 60):
        super(MLP, self).__init__()

        # Fully connected (#input(node: 60, graph: 40), 64)
        self.hidden1 = nn.Linear(features, 20)
        # ReLu
        self.relu1 = nn.ReLU()
        # Linear Fully connected (20, 1)
        self.lin1 = nn.Linear(20,1)

    def forward(self, x):
        h1 = self.hidden1(x)
        r1 = self.relu1(h1)
        l1 = self.lin1(r1)
        return l1

    def loss(self, pOriginal, pSample, edge_ij, coefficientSizeReg, entropyReg):
        # pOriginal: Wahrscheinlichkeit bei original Graph Klasse 1/2 zu sein
        # pSample: Wahrscheinlichkeit bei sampled Graph Klasse 1/2 zu sein
        # size regularization
        sizeReg = torch.sum(edge_ij) * coefficientSizeReg       #We penalize large size of the explanation by adding the sum of all elements of the mask paramters as the regularization term

        # entropy regularization (Binary Cross Entropy beacuse we care for both classes)
        bce = -edge_ij * torch.log(edge_ij + 1e-8) - (1-edge_ij) * torch.log(1-edge_ij + 1e-8)         #we use element-wise entropy to encourage structural and node feature masks to be discrete
        entropyReg = entropyReg * torch.mean(bce)

        Loss = -torch.sum(pOriginal * torch.log(pSample + 1e-8)) + sizeReg + entropyReg                # use sum to get values for all class labels
        return Loss
    
    def getGraphEdgeEmbeddings(self, data_loader, modelGraphGNN):
        embeddings = []

        for batch_index, data in enumerate(data_loader):
            emb = modelGraphGNN.getNodeEmbeddings(data.x, data.edge_index)              # shape: 25 X 20 = Nodes X hidden_embs

            # Transform embeddings so that it contains the concatenated hidden_embs of each two connected nodes
            graph_embs = []
            for index in range(0, len(data.edge_index[0])):
                i = data.edge_index[0][index]
                j = data.edge_index[1][index]

                embCat = torch.cat([emb[i],emb[j]])                     # shape: 1 X 40 = 1Edge X hidden_embs_2Nodes
                graph_embs.append(embCat)                               # shape: ~50 X 40 = Edges X hidden_embs_2Nodes

            embeddings.append(torch.stack(graph_embs))                  # shape: 600 X ~50 X 40 = Graphs X Edges X hidden_embs_2Nodes

        # TODO: transform embeddings with cat or stack? Stack only applicable for same size tensors
        return embeddings
    
    def sampleGraph(self, w_ij, temperature):
        # Sample with reparam trick via edge weights
        epsilon = torch.rand(w_ij.size())       # shape: ~50 X 1 = EdgesOG X epsilon
        
        edge_ij = nn.Sigmoid()((torch.log(epsilon + 1e-8)-torch.log(1-epsilon + 1e-8)+w_ij)/temperature)    # shape: ~50 X 1 = EdgesOG X SampledEdgesProbability
        return edge_ij
    

# Three layer GNN(GCN) for Graph Classification based on PGExplainer paper/Source code
class GraphGNN(nn.Module):
    def __init__(self, features = 10, labels = 2):
        super(GraphGNN, self).__init__()
        
        # TODO: Try GCNConv
        self.hidden1 = gnn.GraphConv(features, 20)   # GraphConvolution layer1: input dim = #features??, output dim = hidden dim = 20 , ReLu activation, bias = true in og config
        nn.init.xavier_uniform_(self.hidden1.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_root.weight)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden2 = gnn.GraphConv(20, 20)         # GraphConvolution layer2: input dim = hidden dim = 20,
        nn.init.xavier_uniform_(self.hidden1.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_root.weight)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.hidden3 = gnn.GraphConv(20, 20)         # GraphConvolution layer3: 
        nn.init.xavier_uniform_(self.hidden1.lin_rel.weight)
        nn.init.xavier_uniform_(self.hidden1.lin_root.weight)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)               # Not used in PGExplainer
        
        self.lin = nn.Linear(20*2, labels)                    # fully connected layer(Dense) => nn.linear: input dim = hidden dim * 2 due to concat of pooling, output dim = output dim = classes


    def forward(self, x, edge_index, batch = None, edge_weights=None):             # edge weights missing ; x, edge_index = feature_tensor, transformed adjs_tensor
        # Encoding net
        emb = self.getNodeEmbeddings(x, edge_index, edge_weights)
        
        # Classifier net
        maxP = gnn.pool.global_max_pool(emb, batch)
        meanP = gnn.pool.global_mean_pool(emb, batch)
        
        out = self.lin(torch.cat([maxP, meanP], -1))            # concat pooled embeddings to get better representation of graph to classify => input_dim lin = 2* size of features?
        
        return out
    

    # Node embeddings need to be returned seperatly to the MLP and are also part of the forward pass
    def getNodeEmbeddings(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
            
        emb1 = self.hidden1(x, edge_index, edge_weights)
        #emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb1 = self.relu1(emb1)
        emb1 = self.dropout1(emb1)
        
        emb2 = self.hidden2(emb1, edge_index, edge_weights)
        #emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        emb2 = self.relu2(emb2)
        emb2 = self.dropout1(emb2)
        
        emb3 = self.hidden3(emb2, edge_index, edge_weights)
        #emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
        emb3 = self.relu3(emb3)
        emb3 = self.dropout1(emb3)                  # TODO: No dropout before out of embeddinggs??

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