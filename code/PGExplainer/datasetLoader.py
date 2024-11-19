import torch
from typing import Literal
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
from torch_geometric.datasets import TUDataset
import pickle




def loadDataset (datasetName: Literal['BA2Motif','MUTAG']) :
    if datasetName == 'BA2Motif' :

        """# 500 Barabasi-Albert graphs (25 nodes, following PGE/GNN?) attached with one house motif
        dataset1 = ExplainerDataset(
            graph_generator=BAGraph(25, 1),
            motif_generator=HouseMotif(),
            num_motifs=1,
            num_graphs=500
        )

        # 500 Barabasi-Albert graphs (25 nodes, following PGE/GNN?) attached with one five-node-cycle motif
        dataset2 = ExplainerDataset(
            graph_generator=BAGraph(25, 1),
            motif_generator=CycleMotif(5),
            num_motifs=1,
            num_graphs=500
        )

        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])      # no features => use 10 dimensional vector with all 1s

        #node_features = torch.ones((dataset.num_nodes, 10))                 # shape (4,10)

        #dataset.node_features = node_features      """

        with open('datasets/BA-2motif.pkl', 'rb') as f:
            adjs, feas, labels = pickle.load(f)
            #dataset = pickle.load(f)

        # TODO: data shuffling + create masks?!

        return adjs, feas, labels                     # Load into DataLoader with batches?
    
    if datasetName == 'MUTAG':
        # TODO: load file from original repo instead
        dataset = TUDataset(root="data/TUDataset", name="MUTAG")

        dataset.download()

        return dataset
    
    # TODO: load nodeC datasets
    

# TODO: visualizeDataset
def visualizeDataset () : 
    return