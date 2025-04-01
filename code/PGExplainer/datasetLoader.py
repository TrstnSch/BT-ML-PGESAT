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

import numpy as np
    
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
        
        # Replace features with 10d ones instead of 0.1s
        num_nodes = data.x.shape[0]  # Get number of nodes
        data.x = torch.ones((num_nodes, 10))  # Replace x with 10d ones
        
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
        dataset = BA2MotifDataset('datasets', pre_transform=addGroundTruth(), force_reload=True)                   #transform=ReplaceFeatures()    10d feature vector of 10 times 0.1 instead of 1, seems to make no difference

    if datasetName == 'MUTAG':
        """transformMUTAG= addGroundTruthMUTAG()
        dataset = TUDataset(os.getcwd() + "/datasets", "Mutagenicity", pre_transform=transformMUTAG)

        dataset.download()"""
        edge_lists, graph_labels, edge_label_lists, node_label_lists = loadOriginalMUTAG()
        dataset = transformMUTAG(edge_lists, graph_labels, edge_label_lists, node_label_lists)
        

    # Implement splits  TODO: Move outside
    """generator1 = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator1)        # this "shuffels" data into 3 splits! Use a generator for fixed set with seed

    return train_dataset, val_dataset, test_dataset"""
    return dataset, labels


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

# This is taken from the original code
def loadOriginalMUTAG ():
    pri = './datasets/'+'MutagOriginal'+'/'+'Mutagenicity'+'_'

    file_edges = pri+'A.txt'
    # file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1
    # print(starts)
    # print(node2graph)
    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, node_label_lists



def transformMUTAG (edge_lists, graph_labels, edge_label_lists, node_label_lists):
    dataList = []
    for i in range(len(edge_lists)):
        sources, targets = zip(*edge_lists[i])
        edge_index = torch.tensor([sources, targets], dtype=torch.int64)
        
        # Features, node_label_lists contains label for each node -> Transfomrm to 14d one-hot vector?
        #node_label_lists[i]
        node_features = torch.nn.functional.one_hot(torch.tensor(node_label_lists[i], dtype=torch.long), num_classes=14)
        
        gt_mask = torch.tensor(edge_label_lists[i], dtype=torch.bool)
        
        # TODO: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
        data = Data(x=node_features.clone().detach().to(torch.float32),y=torch.tensor(graph_labels[i], dtype=torch.long),edge_index=edge_index,gt_mask=gt_mask)
        dataList.append(data)
        
    return dataList