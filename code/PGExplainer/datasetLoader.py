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
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os
# import pickle


def loadDataset (datasetName: Literal['BA2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'TreeGrid'], batch_size: int) :
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

    # Original paper 800 graphs, 2024 paper 1000 graphs. Use BA2MotifDataset?
    if datasetName == 'BA2Motif' :
        """# 400 Barabasi-Albert graphs (20 nodes, following PGE/GNN?) attached with one house motif
        dataset1 = ExplainerDataset(
            graph_generator=BAGraph(20, 1),
            motif_generator=HouseMotif(),
            num_motifs=1,
            num_graphs=400,
            transform=T.Constant()      # appends value 1 node feature for every node
        )
        
        # set y to 0 for House. HOW?!?!?!?! 
        for i, data in enumerate(dataset1):
            data.y = torch.tensor([0])
            #print(data.y)

        # 400 Barabasi-Albert graphs (20 nodes, following PGE/GNN?) attached with one five-node-cycle motif
        dataset2 = ExplainerDataset(
            graph_generator=BAGraph(20, 1),
            motif_generator=CycleMotif(5),
            num_motifs=1,
            num_graphs=400,
            transform=T.Constant()
        )

        # set y to 1 for Cycle
        for i, data in enumerate(dataset2):
            data.y = torch.tensor([1])

        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])      # no features => use 10 dimensional vector with all 1s

        # TODO: create masks?!

        train_loader = DataLoader(dataset, batch_size, True)
        test_loader = DataLoader(dataset, batch_size, True)

        # torch_geometric.transforms.RandomLinkSplit for train/test/valid"""

        dataset = BA2MotifDataset('datasets')                   # 10d feature vector of 10 times 0.1 instead of 1        
    

    if datasetName == 'MUTAG':
        # TODO: check if reimplementation needed. OG uses 10d features?!
        dataset = TUDataset(os.getcwd() + "/datasets", "Mutagenicity")

        dataset.download()

    # Implement splits
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])        # this "shuffels" data into 3 splits! Use a generator for fixed set with seed

    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, True)
    test_loader = DataLoader(test_dataset, batch_size)

    return (train_loader, val_loader, test_loader)
    

# TODO: visualizeDataset
def visualizeDataset () : 
    return