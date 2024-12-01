import torch
import torch.nn as nn

def evaluateGNN(gnn, data_loader):
    
    loss = nn.CrossEntropyLoss() 

    gnn.eval()

    batch_size = data_loader.batch_size

    num_batches = 0.0
    losses = 0.0
    acc_sum = 0

    for batch_index, data in enumerate(data_loader):
        batch_size_ratio = len(data)/batch_size
        num_batches += batch_size_ratio

        with torch.no_grad():
            out = gnn.forward(data.x, data.edge_index, data.batch)
            currLoss = loss(out, data.y)

            pred = out.argmax(dim=1)                     

            acc_sum += torch.sum(pred == data.y)         # DONE: should work with batches!?
            
        losses += batch_size_ratio * currLoss.item()

    return  acc_sum/(num_batches*batch_size), losses/num_batches