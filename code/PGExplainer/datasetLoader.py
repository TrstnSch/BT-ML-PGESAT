import torch
import pickle
from typing import Literal
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.graph_generator import TreeGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
from torch_geometric.datasets.motif_generator import GridMotif
from torch_geometric.datasets.motif_generator import CustomMotif
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os
from torch_geometric.utils import to_undirected

class GaussianFeatureTransform:
    def __init__(self, num_features=10, mean=0.0, std=1.0):
        self.num_features = num_features
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.x = self.mean + self.std * torch.randn((data.num_nodes, self.num_features))  # Gaussian with mean and std
        return data
    
class OnesDimTransform:
    def __init__(self, num_features=10):
        self.num_features = num_features

    def __call__(self, data):
        data.x = torch.ones((data.num_nodes, self.num_features))  # 10D features of ones for each node
        return data
    
class ReplaceFeatures(object):
    def __call__(self, data):
        num_nodes = data.x.shape[0]  # Get number of nodes
        data.x = torch.ones((num_nodes, 10))  # Replace x with 10d ones
        return data
    
# This seems to work
class addGroundTruth(object):
    def __call__(self, data):
        motif_node_indices = torch.arange(20,25)
        ground_truth_indices = []

        for index in range(0, len(data.edge_index[0])):
            if data.edge_index[0][index] in motif_node_indices and data.edge_index[1][index] in motif_node_indices:
                ground_truth_indices.append(index)

        groundTruthMask = torch.zeros_like(data.edge_index[0], dtype=torch.bool)
        groundTruthMask[ground_truth_indices] = 1

        data.gt_mask = groundTruthMask
        return data

class addGroundTruthMUTAG(object):
    def __init__(self):
        self.edge_labels = self.load_edge_labels()

    def load_edge_labels(self):
        """
        Load the edge labels from the given txt file.
        The file format assumes one label per edge across all graphs.
        """
        edge_labels = []
        with open("datasets/Mutagenicity/raw/Mutagenicity_edge_gt.txt", 'r') as f:
            for line in f:
                edge_labels.append(int(line.strip()))
        return edge_labels
    
    def __call__(self, data):
        """
        This method is called to apply the transformation to each graph in the dataset.
        Adds edge labels to the Data object.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        if num_edges > len(self.edge_labels):
            raise ValueError(f"Not enough edge labels available for {num_edges} edges in this graph.")
        
        # Extract the correct number of edge labels for this graph
        labels = self.edge_labels[:num_edges]
        self.edge_labels = self.edge_labels[num_edges:]  # Remove used labels

        # Convert the labels to a tensor and add it to the data object
        data.gt_mask = torch.tensor(labels, dtype=torch.bool)

        return data




def loadGraphDataset (datasetName: Literal['BA-2Motif','MUTAG'], manual_seed=42) :
    
    labels = 2
    
    # Original paper 800 graphs, 2024 paper 1000 graphs. Use BA2MotifDataset?
    if datasetName == 'BA-2Motif' :
        dataset = BA2MotifDataset('datasets', pre_transform=addGroundTruth())                   #transform=ReplaceFeatures()    10d feature vector of 10 times 0.1 instead of 1, seems to make no difference

    if datasetName == 'MUTAG':
        transformMUTAG= addGroundTruthMUTAG()
        dataset = TUDataset(os.getcwd() + "/datasets", "Mutagenicity", pre_transform=transformMUTAG)

        dataset.download()
        

    # Implement splits  TODO: Move outside
    """generator1 = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator1)        # this "shuffels" data into 3 splits! Use a generator for fixed set with seed

    return train_dataset, val_dataset, test_dataset"""
    return dataset, labels




"""def loadNodeDataset (datasetName: Literal['BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid']) :       # TODO: perturb graphs?!
    # TODO: BAGraph() does not create accurate BA Graphs. Graphs can have unconnected nodes
    # no node features assigned
    if datasetName == 'BA-Shapes' :
        dataset = ExplainerDataset(
            graph_generator=BAGraph(300, 1),
            motif_generator=HouseMotif(),
            num_motifs=80,
            num_graphs=1,
            transform=T.Constant()
        )
        labels = 4
        # 4 labels

    if datasetName == 'BA-Community' :
        dataset = ExplainerDataset(
            graph_generator=BAGraph(300, 1),
            motif_generator=HouseMotif(),
            num_motifs=80,
            num_graphs=2,
            transform=GaussianFeatureTransform(10)      # Initializes 10d feature with gaussian distribution
        )
        # TODO: Comine both graphs?!?!?!? dataset[0] and dataset[1]  HOW?!?!?
        labels = 8

    if datasetName == 'Tree-Cycles' :           # Should only append motifs to base graph
        dataset = ExplainerDataset(
            graph_generator=TreeGraph(8),
            motif_generator=CycleMotif(6),
            num_motifs=80,
            num_graphs=1,
            transform=OnesDimTransform(10)      # appends 10d value 1 node feature for every node
        )
        labels = 2

    if datasetName == 'Tree-Grid' :         # Should only append motifs to base graph
        edge_indices = [
            [0, 1],
            [0, 3],
            [1, 4],
            [3, 4],
            [1, 2],
            [2, 5],
            [4, 5],
            [3, 6],
            [6, 7],
            [4, 7],
            [5, 8],
            [7, 8],
            [1, 0],
            [3, 0],
            [4, 1],
            [4, 3],
            [2, 1],
            [5, 2],
            [5, 4],
            [6, 3],
            [7, 6],
            [7, 4],
            [8, 5],
            [8, 7],
        ]
        structure = Data(
            num_nodes=9,
            edge_index=torch.tensor(edge_indices).t().contiguous(),
            y=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )
        dataset = ExplainerDataset(
            graph_generator=TreeGraph(8),
            motif_generator=CustomMotif(structure),             # Use custom motif since GridMotif has 3 instead of 1 label
            num_motifs=80,
            num_graphs=1,
            transform=OnesDimTransform(10)      # appends 10d value 1 node feature for every node
        )
        labels = 2
    
    # TODO: Move outside
    transform = T.RandomNodeSplit('train_rest', 1, num_val = 0.1, num_test = 0.1)

    data = transform(dataset[0])

    return labels, data"""


def loadOriginalNodeDataset (datasetName = Literal['BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid']):
    mapping = {
    'BA-Shapes': 'syn1',
    'BA-Community': 'syn2',
    'Tree-Cycles': 'syn3',
    'Tree-Grid': 'syn4'
    }
    labelMapping = {
    'BA-Shapes': 4,
    'BA-Community': 8,
    'Tree-Cycles': 2,
    'Tree-Grid': 2
    }

    file_name = mapping[datasetName]
    file_path = 'datasets/original/' + file_name + '.pkl'
    
    # SHAPE FOR ALL NODE TASKS: adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    adj = torch.tensor(data[0], dtype=torch.float64)
    edge_index = adj.nonzero().t().contiguous()
    edge_index_undirected = to_undirected(edge_index)
    
    # TODO: gt is kept as matrx/edge_index. Impractical
    edge_label_matrix = torch.tensor(data[8], dtype=torch.float64)
    gt = edge_label_matrix.nonzero().t().contiguous()
    gt_undirected = to_undirected(gt)



    """# TODO: Create mask for edge_index_undirected over gt_undirected
    # Ensure the edges are sorted (important for direct comparison)
    edges_1 = edge_index_undirected.T  # Shape: [N, 2] (edge pairs as rows)
    edges_2 = gt_undirected.T  # Shape: [M, 2] (edge pairs as rows)

    # Create a mask that is True if the edge from edge_index_1 exists in edge_index_2
    gt_mask = torch.zeros(edge_index_undirected.size(1), dtype=torch.bool)  # Initialize a mask of False

    # Loop through each edge in edge_index_1 and check if it is in edge_index_2
    for i, edge in enumerate(edges_1):
        # Check if edge exists in edge_index_2 (order matters)
        if any(torch.equal(edge, e) for e in edges_2):
            gt_mask[i] = True"""
    


    # Convert edges to tuples for easier set operations
    edges_1 = set(map(tuple, edge_index_undirected.T.tolist()))  # Set of edges from edge_index_1
    edges_2 = set(map(tuple, gt_undirected.T.tolist()))  # Set of edges from edge_index_2

    # Find the intersection of edges
    intersection = edges_1 & edges_2

    # Create a mask based on the intersection
    gt_mask = torch.tensor([tuple(edge) in intersection for edge in edge_index_undirected.T.tolist()], dtype=torch.bool)




    x=torch.tensor(data[1], dtype=torch.float32)
    
    y_full = torch.zeros_like(torch.tensor(data[2], dtype=torch.float64))
    y_train, y_val, y_test = torch.tensor(data[2], dtype=torch.float64), torch.tensor(data[3], dtype=torch.float64), torch.tensor(data[4], dtype=torch.float64)
    train_mask, val_mask, test_mask = torch.tensor(data[5], dtype=torch.bool), torch.tensor(data[6], dtype=torch.bool), torch.tensor(data[7], dtype=torch.bool)

    # TODO: Optimize, Use logical OR?
    y_full[train_mask] = y_train[train_mask]
    y_full[val_mask] = y_val[val_mask]
    y_full[test_mask] = y_test[test_mask]
    
    y_full = torch.argmax(y_full, dim=1)
    
    data = Data(x=x,edge_index=edge_index_undirected, y=y_full, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, gt_mask=gt_mask)

    return data, labelMapping[datasetName]


# Probably not needed
def loadOriginalGraphDataset (datasetName: Literal['BA2Motif','MUTAG']):
    mapping = {
    'BA2Motif': 'BA2-motif',
    'MUTAG': 'Mutagenicity'
    }

    file_name = mapping[datasetName]
    file_path = 'datasets/original/' + file_name + '.pkl'
    
    # SHAPE FOR GRPAH TASKS: 
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    return data