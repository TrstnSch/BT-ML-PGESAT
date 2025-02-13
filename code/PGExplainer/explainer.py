import torch
import torch.nn as nn
import utils

# TODO: PGExplainer MLP with one hidden layer?      Input: Concatenated node embeddings for each edge (graph), conc. node embeddings and embedding of node to be predicted? (node)
class MLP(nn.Module):
    def __init__(self, GraphTask=True, hidden_dim=64):
        super(MLP, self).__init__()
        
        self.graphTask = GraphTask
        
        self.inputSize = 2 * 20 if GraphTask else 3 * 60

        self.model = nn.Sequential(                 # Fully connected (#input(node: 60, graph: 40), 64) => WRONG!
            nn.Linear(self.inputSize, hidden_dim),        # Embedding size for graph is 20, for node is 60. Input MLP for graph is 2*emb, for node is 3*emb
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                # Linear Fully connected (20, 1)
        )
        
        self.init_weights()  # Call the initialization function


    def init_weights(self):
        """Xavier Initialization for Linear Layers"""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier for weights
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Zero for bias

            
    def forward(self, modelGraphGNN, x, edge_index, nodeToPred=None):
        """Forward method of our model. This starts by calculating edge embeddings on thy fly.
        These are the inputs that are fed through the model. The model output edge weights are transformed
        so that both parts of the adj. matrix/edge_index hold identical weights for each edge(symmetric).

        Args:
            modelGraphGNN (GraphGNN): The downstream task for calculating the edge embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            float tensor: Edge weights
        """
        embeddings = self.getGraphEdgeEmbeddings(modelGraphGNN, x, edge_index, nodeToPred)
        
        # TODO: Check if this can remain like this for NodePred
        w_ij = self.model(embeddings).squeeze(1)
        
        # TODO: Take absolute value of w_ij to always have positive weights?!
        #w_ij = torch.abs(w_ij)
        
        # TODO: Validate
        w_ij_sym = utils.combineEdgeWeights(edge_index, w_ij)
        
        # TODO: Original uses structure that directly calls sampleGraph and puts through downstream taks, returns prediction for sample and saves edgeMask as global var
        return w_ij_sym


    def loss(self, pOriginal, pSample, edge_ij, coefficientSizeReg, entropyReg, coefficientL2Reg=0.0):
        """Loss of explanation model for singular (sampled) instance(graph)

        Args:
            pOriginal (float tensor): Probability of original graph to be class 1/2
            pSample (float tensor): Probability of sampled graph to be class 1/2
            edge_ij (float tensor): Probability of edge i,j to be in the explanation, Sigmoid applied
            coefficientSizeReg (float): Coefficient for size regularization
            entropyReg (float): Coefficient for entropy regularization

        Returns:
            float: Loss of explanation model
        """
        # size regularization: penalize large size of the explanation by adding the sum of all elements of the mask parameters as the regularization term
        # TODO: This should be on the weights, not the probabilites?
        sizeReg = torch.sum(edge_ij) * coefficientSizeReg

        # entropy regularization (Binary Cross Entropy beacuse we care for both classes) to encourage structural and node feature masks to be discrete
        bce = -edge_ij * torch.log(edge_ij + 1e-8) - (1-edge_ij) * torch.log(1-edge_ij + 1e-8)
        entropyReg = entropyReg * torch.mean(bce)
        
        # coefficientL2Reg is 0 in standard og config 
        l2norm = 0.0
        if not self.graphTask:
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    l2norm += torch.norm(param)

            l2norm = coefficientL2Reg * l2norm

        # TODO: sizeReg and/or entropyReg mess up weights to be negative
        Loss = -torch.sum(pOriginal * torch.log(pSample + 1e-8)) + entropyReg + sizeReg + l2norm              # use sum to get values for all class labels
        #Loss = torch.nn.functional.cross_entropy(pOriginal, pSample) + entropyReg + sizeReg
        #Loss = torch.nn.functional.cross_entropy(pSample, pOriginal) + entropyReg + sizeReg             # This is used in PyG impl.
        #Loss = -torch.log(pSample[torch.argmax(pOriginal)]) + entropyReg + sizeReg                      # This is used in og?
        return Loss
    
    
    def getGraphEdgeEmbeddings(self, modelGraphGNN, x, edge_index, nodeToPred=None):
        """Generate the edge embeddings from the node embeddings for each graph in the dataset used as input for the MLP.
        In case of Node prediction, this generates the edge embeddings for the computational graph 
        and appends the node embeddings for node to predict.

        Args:
            data_loader (DataLoader): DataLoader containing the dataset
            modelGraphGNN (GNN): NodeGNN/GraphGNN model to generate node embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            Listy<float tensor>: List of edge embeddings per graph
        """
        emb = modelGraphGNN.getNodeEmbeddings(x, edge_index)              # shape: 25 X 20 = Nodes X hidden_embs

        # Transform embeddings so that it contains the concatenated hidden_embs of each two connected nodes
        i, j = edge_index[0], edge_index[1]
        
        if nodeToPred is not None:
            # append embeddings (60d hidden features) for NodeToPred(startNode)
            # embCat[nodeToPred] dimension needs to be scaled up to edge_index length? Num_Edges X 60
            embCat = torch.cat([emb[i], emb[j], emb[nodeToPred].repeat(len(i), 1)], dim=1)
        else:
            embCat = torch.cat([emb[i], emb[j]], dim=1)
            
        return embCat
    

    def sampleGraph(self, w_ij, temperature):
        """Implementation of the reparametrization trick to sample edges from the edge weights. 
        If evaluating we only apply Sigmoid to get predictions from weights while eliminating randomness.

        Args:
            w_ij (float tensor): Edge weights from the MLP
            temperature (float): Current temperature for sampling

        Returns:
            float tensor: Probability of edge i,j to be in the explanation, Sigmoid applied
        """
        if self.training:
            epsilon = torch.rand(w_ij.size()) + 1e-8                   # shape: ~50 X 1 = EdgesOG X epsilon
        
            edge_ij = nn.Sigmoid()((torch.log(epsilon)-torch.log(1-epsilon)+w_ij)/temperature)    # shape: ~50 X 1 = EdgesOG X SampledEdgesProbability
        else:
            edge_ij = nn.Sigmoid()(w_ij)
            
        return edge_ij