import torch
from typing import Literal
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.graph_generator import TreeGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
from torch_geometric.datasets.motif_generator import GridMotif
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os


def loadGraphDataset (datasetName: Literal['BA2Motif','MUTAG']) :
    
    # Original paper 800 graphs, 2024 paper 1000 graphs. Use BA2MotifDataset?
    if datasetName == 'BA2Motif' :
        dataset = BA2MotifDataset('datasets')                   # 10d feature vector of 10 times 0.1 instead of 1        
    

    if datasetName == 'MUTAG':
        # TODO: check if reimplementation needed. OG uses 10d features?!
        dataset = TUDataset(os.getcwd() + "/datasets", "Mutagenicity")

        dataset.download()

    # Implement splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])        # this "shuffels" data into 3 splits! Use a generator for fixed set with seed

    return train_dataset, val_dataset, test_dataset




def loadNodeDataset (datasetName: Literal['BA-Shapes', 'BA-Community', 'Tree-Cycles', 'TreeGrid']) :
    # no node features assigned
    if datasetName == 'BA-Shapes' :
        dataset = ExplainerDataset(
            graph_generator=BAGraph(300, 1),
            motif_generator=HouseMotif(),
            num_motifs=80,
            num_graphs=1,
        )

    if datasetName == 'BA-Community' :
        dataset = ExplainerDataset(
            graph_generator=BAGraph(300, 1),
            motif_generator=HouseMotif(),
            num_motifs=80,
            num_graphs=2,
            #transform=T.Constant()      # TODO: use 2 gaussian distributions
        )
    if datasetName == 'Tree-Cycles' :
        dataset = ExplainerDataset(
            graph_generator=TreeGraph(8),
            motif_generator=CycleMotif(),
            num_motifs=80,
            num_graphs=1,
            #transform=T.Constant()      # TODO: appends value 1 node feature for every node?
        )

    if datasetName == 'TreeGrid' :
        dataset = ExplainerDataset(
            graph_generator=TreeGraph(8),
            motif_generator=GridMotif(),
            num_motifs=80,
            num_graphs=1,
            #transform=T.Constant()      # TODO: appends value 1 node feature for every node?
        )
    
    return dataset