import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math

#PGExplainer MLP with one hidden layer?      Input: Concatenated node embeddings for each edge (graph), conc. node embeddings and embedding of node to be predicted? (node)
class MLP(nn.Module):
    def __init__(self, GraphTask=True, hidden_dim=64, emb_dim=128):
        super(MLP, self).__init__()
        
        self.graphTask = GraphTask
        
        #self.inputSize = 2 * 20 if GraphTask else 3 * 60
        # emb*dim * 2 for both nodes per edge, * 3 since we take embeddings from 3 iterations
        self.inputSize = 2 * emb_dim * 3

        self.model = nn.Sequential(
            nn.Linear(self.inputSize, hidden_dim),          # Embedding size for graph is 20, for node is 60. Input MLP for graph is 2*emb, for node is 3*emb
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                        # Linear Fully connected (20, 1)
        )
        
        """nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),
            nn.ReLU(),"""
            
        
        self.init_weights()


    def init_weights(self):
        """Xavier Initialization for Linear Layers"""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier for weights
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Zero for bias

            
    #def forward(self, downstreamTask, x, edge_index, nodeToPred=None):
    def forward(self, downstreamTask, problem, nodeToPred=None):
        """Forward method of our model. This starts by calculating edge embeddings on thy fly.
        These are the inputs that are fed through the model. The model output edge weights are transformed
        so that both parts of the adj. matrix/edge_index hold identical weights for each edge(symmetric).

        Args:
            downstreamTask (GraphGNN): The downstream task for calculating the edge embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            float tensor: Edge weights
        """
        problem = problem
        
        embeddings = self.getGraphEdgeEmbeddings(downstreamTask, problem, nodeToPred)
        
        #print(f"Input embeddings for MLP: {embeddings}")
        #print(f"Max input embedding for MLP: {torch.max(embeddings)}")
        #print(f"Min input embedding for MLP: {torch.min(embeddings)}")
        #print("------------------------------------------")
        
        w_ij = self.model(embeddings).squeeze(1)
        
        # TODO: Take absolute value of w_ij to always have positive weights?!
        #w_ij = torch.abs(w_ij)
        
        # TODO: Validate, maybe move to sample if evaluating
        #w_ij_sym = utils.combineEdgeWeights(edge_index, w_ij)
        
        return w_ij


    # current_batch_edges contains batch_edges for the current sub_problem/graph
    def loss(self, pOriginal, pSample, edge_ij, current_batch_edges, coefficientSizeReg, coefficientEntropyReg, coefficientL2Reg=0.0, coefficientConsistency=0.0, coefficientConnect=0.0):
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
        entropyReg = coefficientEntropyReg * torch.mean(bce)
        
        # coefficientL2Reg is 0 in standard og config 
        l2norm = 0.0
        if not self.graphTask:
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    l2norm += torch.norm(param)

            l2norm = coefficientL2Reg * l2norm
            
            
        # My idea
        consistencyLoss = 0.0
        
        current_batch_clauses = torch.tensor(current_batch_edges[:, 1])
        clauses = torch.unique(current_batch_clauses)
        
        for clause_id in clauses:
            # apply i-th edge mask to the current_batch_edges to get connections for clause i
            mask = current_batch_clauses == clause_id
            #clause_edges = current_batch_edges[mask]
            clause_edge_probs = edge_ij[mask]
            
            if len(clause_edge_probs) > 1:
                clauses_var = torch.var(clause_edge_probs)
                consistencyLoss += clauses_var
            
        Loss = -torch.sum(pOriginal * torch.log(pSample + 1e-8)) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency               # use sum to get values for all class labels
        #Loss = torch.nn.functional.cross_entropy(pSample, pOriginal) + entropyReg + sizeReg             # This is used in PyG impl.
        #Loss = -torch.log(pSample[torch.argmax(pOriginal)]) + entropyReg + sizeReg                      # This is used in og?
        
        return Loss
    
    
    #def getGraphEdgeEmbeddings(self, downstreamTask, x, edge_index, nodeToPred=None):
    def getGraphEdgeEmbeddings(self, downstreamTask, problem, nodeToPred=None):
        """Generate the edge embeddings from the node embeddings for each graph in the dataset used as input for the MLP.
        In case of Node prediction, this generates the edge embeddings for the computational graph 
        and appends the node embeddings for node to predict.

        Args:
            data_loader (DataLoader): DataLoader containing the dataset
            downstreamTask (GNN): NodeGNN/GraphGNN model to generate node embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            Listy<float tensor>: List of edge embeddings per graph
        """
        problem = problem
        
        _, _, all_l_emb, all_c_emb = downstreamTask.forward(problem)              # shape: 25 X 20 = Nodes X hidden_embs
        
        # Concat l_emb and c_emb where edge between l_emb and c_emb??
        # i has to be source literals connected to j clauses
        
        #Take last element from l_emb/c_emb, as it is output form "last hidden layer" -> Not hidden layer, as all layers share the same parameters
        l_emb = all_l_emb[-1]           # Shape: (n_literals, emb_dim=128)
        c_emb = all_c_emb[-1]           # Shape: (n_clauses, emb_dim=128)
        
        iterations = downstreamTask.opts['iterations']
        
        l_emb_interm1 = all_l_emb[math.floor(iterations * 0.5)]           # Shape: (n_literals, emb_dim=128)
        c_emb_interm1 = all_c_emb[math.floor(iterations * 0.5)]           # Shape: (n_clauses, emb_dim=128)
        
        l_emb_interm2 = all_l_emb[math.floor(iterations * 0.75)]           # Shape: (n_literals, emb_dim=128)
        c_emb_interm2 = all_c_emb[math.floor(iterations * 0.75)]           # Shape: (n_clauses, emb_dim=128)
        
        l_embs_cat = torch.cat([l_emb, l_emb_interm1, l_emb_interm2], dim=1)
        c_embs_cat = torch.cat([c_emb, c_emb_interm1, c_emb_interm2], dim=1)
        
        # This does not grant larger edge weights
        """l_emb = F.normalize(all_l_emb[-1], p=2, dim=1)  # L2 normalization
        c_emb = F.normalize(all_c_emb[-1], p=2, dim=1)  # L2 normalization"""
            

        # Transform embeddings so that it contains the concatenated hidden_embs of each two connected nodes
        # edges = problem.L_unpack_indices is edge_index? If so, edge index is from literal to clause?
        # -> edge_index = edges and i from literal, j from clause?
        #i, j = problem.batch_edges[0], problem.batch_edges[1]
        
        # Maybe remove to.(device and just use it in forward?)
        i = torch.tensor(problem.batch_edges[:,0]).to(device)
        j = torch.tensor(problem.batch_edges[:,1]).to(device)
        
        if nodeToPred is not None:
            # append embeddings (60d hidden features) for NodeToPred(startNode)
            # embCat[nodeToPred] dimension needs to be scaled up to edge_index length? Num_Edges X 60
            """embCat = torch.cat([emb[i], emb[j], emb[nodeToPred].repeat(len(i), 1)], dim=1)"""
        else:
            """embCat = torch.cat([emb[i], emb[j]], dim=1)"""
            #embCat = torch.cat([l_emb[i], c_emb[j]], dim=1)
            embCat = torch.cat([l_embs_cat[i], c_embs_cat[j]], dim=1)
            
        return embCat
    

    def sampleGraph(self, w_ij, temperature=1):
        """Implementation of the reparametrization trick to sample edges from the edge weights. 
        If evaluating we only apply Sigmoid to get predictions from weights while eliminating randomness.

        Args:
            w_ij (float tensor): Edge weights from the MLP
            temperature (float): Current temperature for sampling

        Returns:
            float tensor: Probability of edge i,j to be in the explanation, Sigmoid applied
        """
        if self.training:
            epsilon = torch.rand(w_ij.size()).to(device) + 1e-8                   # shape: ~50 X 1 = EdgesOG X epsilon

            edge_ij = nn.Sigmoid()((torch.log(epsilon)-torch.log(1-epsilon)+w_ij)/temperature)    # shape: ~50 X 1 = EdgesOG X SampledEdgesProbability
        else:
            edge_ij = nn.Sigmoid()(w_ij)
            
        return edge_ij
    
    
    
    
    
    
    
    

class MLP_SAT(nn.Module):
    """This model predicts clauses in the form of edges belonging to each clause. Therefore, edge logits are meaned for all edges that belong to the same clause.
    Also, during the sampling the randomness is added per clause instead of per edge.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, GraphTask=True, hidden_dim=64, emb_dim=128, complex_architecture=True, three_embs=True):
        super(MLP_SAT, self).__init__()
        
        self.graphTask = GraphTask
        self.three_embs = three_embs
        
        # emb*dim * 2 for both nodes per edge; * 3 since we take embeddings from 3 iterations
        self.inputSize = 2 * emb_dim * 3 if three_embs else 2 * emb_dim

        if complex_architecture:
            self.model = nn.Sequential(
                nn.Linear(self.inputSize, 256),          # Embedding size for graph is 20, for node is 60. Input MLP for graph is 2*emb, for node is 3*emb
                nn.ReLU(),
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 20),
                nn.ReLU(),
                nn.Linear(20, 1)                        # Linear Fully connected (20, 1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputSize, hidden_dim),          # Embedding size for graph is 20, for node is 60. Input MLP for graph is 2*emb, for node is 3*emb
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)                        # Linear Fully connected (20, 1)
            )
            
        self.init_weights()


    def init_weights(self):
        """Xavier Initialization for Linear Layers"""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier for weights
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Zero for bias

            
    #def forward(self, downstreamTask, x, edge_index, nodeToPred=None):
    def forward(self, downstreamTask, problem, nodeToPred=None):
        """Forward method of our model. This starts by calculating edge embeddings on thy fly.
        These are the inputs that are fed through the model. The model output edge weights are transformed
        so that both parts of the adj. matrix/edge_index hold identical weights for each edge(symmetric).

        Args:
            downstreamTask (GraphGNN): The downstream task for calculating the edge embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            float tensor: Edge weights
        """
        embeddings = self.getGraphEdgeEmbeddings(downstreamTask, problem, nodeToPred)
        
        #print(f"Input embeddings for MLP: {embeddings}")
        #print(f"Max input embedding for MLP: {torch.max(embeddings)}")
        #print(f"Min input embedding for MLP: {torch.min(embeddings)}")
        #print("------------------------------------------")
        
        w_ij = self.model(embeddings).squeeze(1)
        
        # TODO: ADD hard constraint for network to learn to include all edges of each clause
        batch_clauses = torch.tensor(problem.batch_edges[:, 1])
        clauses, inverse_indices = torch.unique(batch_clauses, dim=0, return_inverse=True)
        
        for clause_id in clauses:
            # apply i-th edge mask to the current_batch_edges to get connections for clause i
            mask = batch_clauses == clause_id
            #clause_edges = current_batch_edges[mask]
            clause_edge_probs = w_ij[mask]
            
            if len(clause_edge_probs) > 1:
                clause_mean_weight = torch.mean(clause_edge_probs)
                w_ij[mask] = clause_mean_weight
        
        return w_ij, clauses, inverse_indices


    # current_batch_edges contains batch_edges for the current sub_problem/graph
    def loss(self, pOriginal, pSample, edge_ij, current_batch_edges, coefficientSizeReg, coefficientEntropyReg, coefficientL2Reg=0.0, coefficientConsistency=0.0, bce=False):
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
        entropyReg = coefficientEntropyReg * torch.mean(bce)
        
        # coefficientL2Reg is 0 in standard og config 
        l2norm = 0.0
        if not self.graphTask:
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    l2norm += torch.norm(param)

            l2norm = coefficientL2Reg * l2norm
            
            
        # My idea
        consistencyLoss = 0.0
        
        if coefficientConsistency > 0.0:
            current_batch_clauses = torch.tensor(current_batch_edges[:, 1])
            clauses = torch.unique(current_batch_clauses)
            
            for clause_id in clauses:
                # apply i-th edge mask to the current_batch_edges to get connections for clause i
                mask = current_batch_clauses == clause_id
                #clause_edges = current_batch_edges[mask]
                clause_edge_probs = edge_ij[mask]
                
                if len(clause_edge_probs) > 1:
                    clauses_var = torch.var(clause_edge_probs)
                    consistencyLoss += clauses_var
            
        if bce is True:
            Loss = torch.nn.functional.binary_cross_entropy(pSample, pOriginal) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency
        else:
            Loss = -torch.sum(pOriginal * torch.log(pSample + 1e-8)) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency               # use sum to get values for all class labels
        #Loss = torch.nn.functional.cross_entropy(pSample, pOriginal) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency             # This is used in PyG impl.
        #Loss = torch.nn.functional.binary_cross_entropy(pSample, pOriginal) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency
        #Loss = -torch.log(pSample) + entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency                      # This is used in og? Named logits but softmax should be applied
        
        # Try this and check whether this is the loss used in original code?
        """orgininal_preds = torch.argmax(pOriginal, dim=1)
        rows_pSample = torch.arange(pSample.size(0))
        Loss = -torch.sum(torch.log(pSample[rows_pSample, orgininal_preds] + 1e-8))+ entropyReg + sizeReg + l2norm + consistencyLoss * coefficientConsistency"""
        
        return Loss
    
    
    #def getGraphEdgeEmbeddings(self, downstreamTask, x, edge_index, nodeToPred=None):
    def getGraphEdgeEmbeddings(self, downstreamTask, problem, nodeToPred=None):
        """Generate the edge embeddings from the node embeddings for each graph in the dataset used as input for the MLP.
        In case of Node prediction, this generates the edge embeddings for the computational graph 
        and appends the node embeddings for node to predict.

        Args:
            data_loader (DataLoader): DataLoader containing the dataset
            downstreamTask (GNN): NodeGNN/GraphGNN model to generate node embeddings
            x (float tensor): Node features [num_nodes, num_features]
            edge_index (float tensor): Adjacency matrix/edge_index [2, num_edges]

        Returns:
            Listy<float tensor>: List of edge embeddings per graph
        """
        problem = problem
        
        _, _, all_l_emb, all_c_emb = downstreamTask.forward(problem)              # shape: 25 X 20 = Nodes X hidden_embs
        
        # Concat l_emb and c_emb where edge between l_emb and c_emb??
        # i has to be source literals connected to j clauses
        
        #Take last element from l_emb/c_emb, as it is output form "last hidden layer" -> Not hidden layer, as all layers share the same parameters
        l_emb = all_l_emb[-1]           # Shape: (n_literals, emb_dim=128)
        c_emb = all_c_emb[-1]           # Shape: (n_clauses, emb_dim=128)
        
        iterations = downstreamTask.opts['iterations']
        
        # Use concatenation of embeddings from middel, 3/4 and last iteration if wanted, or just last
        if self.three_embs:
            l_emb_interm1 = all_l_emb[math.floor(iterations * 0.5)]           # Shape: (n_literals, emb_dim=128)
            c_emb_interm1 = all_c_emb[math.floor(iterations * 0.5)]           # Shape: (n_clauses, emb_dim=128)
            
            l_emb_interm2 = all_l_emb[math.floor(iterations * 0.75)]           # Shape: (n_literals, emb_dim=128)
            c_emb_interm2 = all_c_emb[math.floor(iterations * 0.75)]           # Shape: (n_clauses, emb_dim=128)
            
            l_embs_cat = torch.cat([l_emb, l_emb_interm1, l_emb_interm2], dim=1)
            c_embs_cat = torch.cat([c_emb, c_emb_interm1, c_emb_interm2], dim=1)
        else:
            l_embs_cat = l_emb
            c_embs_cat = c_emb
        # This does not grant larger edge weights
        """l_emb = F.normalize(all_l_emb[-1], p=2, dim=1)  # L2 normalization
        c_emb = F.normalize(all_c_emb[-1], p=2, dim=1)  # L2 normalization"""
            

        # Transform embeddings so that it contains the concatenated hidden_embs of each two connected nodes
        # edges = problem.L_unpack_indices is edge_index? If so, edge index is from literal to clause?
        # -> edge_index = edges and i from literal, j from clause?
        #i, j = problem.batch_edges[0], problem.batch_edges[1]
        
        # Maybe remove to.(device and just use it in forward?)
        i = torch.tensor(problem.batch_edges[:,0]).to(device)
        j = torch.tensor(problem.batch_edges[:,1]).to(device)
        
        if nodeToPred is not None:
            # append embeddings (60d hidden features) for NodeToPred(startNode)
            # embCat[nodeToPred] dimension needs to be scaled up to edge_index length? Num_Edges X 60
            """embCat = torch.cat([emb[i], emb[j], emb[nodeToPred].repeat(len(i), 1)], dim=1)"""
        else:
            """embCat = torch.cat([emb[i], emb[j]], dim=1)"""
            # For using only last emb
            #embCat = torch.cat([l_emb[i], c_emb[j]], dim=1)
            # For also using intermediate embs
            embCat = torch.cat([l_embs_cat[i], c_embs_cat[j]], dim=1)
            
        return embCat
    

    def sampleGraph(self, w_ij, clauses, inverse_indices, temperature=1):
        """Implementation of the reparametrization trick to sample edges from the edge weights. 
        If evaluating we only apply Sigmoid to get predictions from weights while eliminating randomness.

        Args:
            w_ij (float tensor): Edge weights from the MLP
            temperature (float): Current temperature for sampling

        Returns:
            float tensor: Probability of edge i,j to be in the explanation, Sigmoid applied
        """
        if self.training:
            # TODO: Try sampling randomness for all clause edges collectively instead of per each edge?
            # FIXME: unique W-ij does not guarantee randomness per edge pair, as multiple edge pairs might have same logit
            #unique_vals, inverse_indices = torch.unique(w_ij, return_inverse=True)
            rand_vals = torch.rand(len(clauses), device=w_ij.device) + 1e-8
            epsilon = rand_vals[inverse_indices].reshape(w_ij.shape)
            
            #epsilon = torch.rand(w_ij.size()).to(device) + 1e-8                   # shape: ~50 X 1 = EdgesOG X epsilon

            edge_ij = nn.Sigmoid()((torch.log(epsilon)-torch.log(1-epsilon)+w_ij)/temperature)    # shape: ~50 X 1 = EdgesOG X SampledEdgesProbability
        else:
            edge_ij = nn.Sigmoid()(w_ij)
            
        return edge_ij
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')