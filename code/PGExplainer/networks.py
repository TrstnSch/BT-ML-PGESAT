import torch
import torch.nn as nn
import torch_geometric.nn as gnn    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Three layer GNN(GCN) for Graph Classification based on PGExplainer paper/Source code
class GraphGNN(nn.Module):
    def __init__(self, features = 10, labels = 2):
        super(GraphGNN, self).__init__()
        
        # TODO: Try GCNConv
        self.hidden1 = gnn.GraphConv(features, 20)   # GraphConvolution layer1: input dim = #features??, output dim = hidden dim = 20 , ReLu activation, bias = true in og config
        self.hidden2 = gnn.GraphConv(20, 20)         # GraphConvolution layer2: input dim = hidden dim = 20,
        self.hidden3 = gnn.GraphConv(20, 20)         # GraphConvolution layer3: 
        
        #self.bn1 = gnn.BatchNorm(20)
        #self.bn2 = gnn.BatchNorm(20)
        #self.bn3 = gnn.BatchNorm(20)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)               # Not used in PGExplainer

        self.lin = nn.Linear(20*2, labels)             # fully connected layer(Dense) => nn.linear: input dim = hidden dim * 2 due to concat of pooling, output dim = output dim = classes
        
        self.init_weights()


    def init_weights(self):
        """Xavier initialization for GraphConv layers and Linear layer"""
        for layer in [self.hidden1, self.hidden2, self.hidden3]:
            nn.init.xavier_uniform_(layer.lin_rel.weight)
            nn.init.xavier_uniform_(layer.lin_root.weight)
        
        nn.init.xavier_uniform_(self.lin.weight)
        
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)
            
            
    def forward(self, x, edge_index, batch = None, edge_weights=None):             # x, edge_index = feature_tensor, transformed adjs_tensor
        """Forward pass of the Graph Neural Network.
        
        Args:
            x: Node feature tensor [num_nodes, num_features]
            edge_index: Edge index tensor [2, num_edges]
            batch: Batch tensor for graph pooling (default: None)
            edge_weights: Optional edge weights
            
        Returns:
            out: Classification output tensor [batch_size, num_classes]
        """
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
            
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(device)
        
        # Encoding net
        emb = self.getNodeEmbeddings(x, edge_index, edge_weights)
        
        # Classifier net
        maxP = gnn.pool.global_max_pool(emb, batch)
        meanP = gnn.pool.global_mean_pool(emb, batch)
        
        out = self.lin(torch.cat([maxP, meanP], -1))            # concat pooled embeddings to get better representation of graph to classify => input_dim lin = 2* size of features?
        
        return out
    

    def getNodeEmbeddings(self, x, edge_index, edge_weights=None):
        """Get the node embeddings for the input graph, passed to forward. Have to be returned seperatly to be used the MLP training loop.
            Node embeddings are fed through all three layers and later pooled and concatenated.
            
        Args:
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]
            edge_weights (float tensor, optional): Edge weights, important for edge prediction in explainer. Defaults to None.

        Returns:
            flaot tensor: Node embeddings
        """
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(device)
            
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
            
        emb1 = self.hidden1(x, edge_index, edge_weights)
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)              # This improves model!? Used to normalize edge embeddings for graphs task, as they else get too high/low for BA-2Motif in explainer
        emb1 = self.relu(emb1)
        #emb1 = self.bn1(emb1)
        #emb1 = self.dropout(emb1)
        
        emb2 = self.hidden2(emb1, edge_index, edge_weights)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        emb2 = self.relu(emb2)
        #emb2 = self.bn2(emb2)
        #emb2 = self.dropout(emb2)
        
        emb3 = self.hidden3(emb2, edge_index, edge_weights)
        emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
        #emb3 = self.bn3(emb3)
        emb3 = self.relu(emb3)
        #emb3 = self.dropout(emb3)

        return emb3
    


class NodeGNN(nn.Module):
    def __init__(self, features, labels):
        super(NodeGNN, self).__init__()
        
        self.hidden1 = gnn.GraphConv(features, 20)
        self.hidden2 = gnn.GraphConv(20, 20)
        self.hidden3 = gnn.GraphConv(20, 20)
        
        # LayerNorm instead of BatchNorm(Did not work)
        self.bn1 = gnn.BatchNorm(20)
        self.bn2 = gnn.BatchNorm(20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)                # Not used in PGExplainer

        self.lin = nn.Linear(20*3, labels)              # outputs of three hidden layers are by default concatenated in PGE Node classification and input into lin layer
        
        self.init_weights()


    def init_weights(self):
        """Xavier initialization for GraphConv layers and Linear layer"""
        for layer in [self.hidden1, self.hidden2, self.hidden3]:
            nn.init.xavier_uniform_(layer.lin_rel.weight)
            nn.init.xavier_uniform_(layer.lin_root.weight)
        
        nn.init.xavier_uniform_(self.lin.weight)
        
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)
            
            
    def forward(self, x, edge_index, batch = None, edge_weights=None):
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(device)
            
        # Encoding net
        emb = self.getNodeEmbeddings(x, edge_index, edge_weights)
        
        # Classifier net, no pooling in node network
        out = self.lin(emb)
        
        #TODO: Softmax should be here according to paper?? Should not change much, as long as embeddings do not get it
        #out = nn.Softmax(out)
        return out
    

    def getNodeEmbeddings(self, x, edge_index, edge_weights=None):
        """Get the node embeddings for the input graph, passed to forward. Have to be returned seperatly to be used the MLP training loop.
            Node embeddings from each layer are concatenated.
        Args:
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]
            edge_weights (float tensor, optional): Edge weights, important for edge prediction in explainer. Defaults to None.

        Returns:
            flaot tensor: Node embeddings
        """
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(device)
            
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
            
        emb1 = self.hidden1(x, edge_index, edge_weights)
        emb1 = self.relu(emb1)
        emb1 = self.bn1(emb1)
        #emb1 = self.dropout(emb1)
        
        emb2 = self.hidden2(emb1, edge_index, edge_weights)
        emb2 = self.relu(emb2)
        emb2 = self.bn2(emb2)
        #emb2 = self.dropout(emb2)
        
        emb3 = self.hidden3(emb2, edge_index, edge_weights)
        emb3 = self.relu(emb3)
        #emb3 = self.dropout(emb3)

        embs = torch.cat([emb1, emb2, emb3], 1)
        
        return embs