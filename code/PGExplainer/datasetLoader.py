import torch
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





def loadGraphDataset (datasetName: Literal['BA2Motif','MUTAG']) :
    
    # Original paper 800 graphs, 2024 paper 1000 graphs. Use BA2MotifDataset?
    if datasetName == 'BA2Motif' :
        dataset = BA2MotifDataset('datasets')                   #transform=ReplaceFeatures()    10d feature vector of 10 times 0.1 instead of 1, seems to make no difference
    

    if datasetName == 'MUTAG':
        dataset = TUDataset(os.getcwd() + "/datasets", "Mutagenicity")

        dataset.download()

    # Implement splits  TODO: Move outside
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])        # this "shuffels" data into 3 splits! Use a generator for fixed set with seed

    return train_dataset, val_dataset, test_dataset




def loadNodeDataset (datasetName: Literal['BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid']) :       # TODO: perturb graphs?!
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

    return labels, data



