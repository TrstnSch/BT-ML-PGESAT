import torch
import torch.nn as nn

def evaluateGNN(gnn, val_loader, test_loader):
    
    loss = nn.CrossEntropyLoss() 

    gnn.eval()

    batch_size = val_loader.batch_size

    #train_acc, val_acc, test_acc = 0.0
    num_batches = 0.0
    valLosses = 0.0
    testLosses = 0.0
    test_acc_sum = 0
    val_acc_sum = 0

    for batch_index, data in enumerate(val_loader):
        batch_size_ratio = len(data)/batch_size
        num_batches += batch_size_ratio

        with torch.no_grad():
            out = gnn.forward(data.x, data.edge_index, data.batch)
            valLoss = loss(out, data.y)

            valPred = out.argmax(dim=1)                     

            val_acc_sum += torch.sum(valPred == data.y)         # DONE: should work with batches!?
            

        valLosses += batch_size_ratio * valLoss

    val_acc = val_acc_sum/(num_batches*batch_size)          # ROUNDING ERROR SOMEWHERE. Works on batch sizes 1 and 2 but not 16, because loader size 100 not divisible by 16
    num_batches = 0.0

    for batch_index, data in enumerate(test_loader):
        batch_size_ratio = len(data)/batch_size
        num_batches += batch_size_ratio

        with torch.no_grad():
            out = gnn.forward(data.x, data.edge_index, data.batch)
            testLoss = loss(out, data.y)

            testPred = out.argmax(dim=1)                    

            test_acc_sum += torch.sum(testPred == data.y)       # DONE: should work with batches!?
            

        testLosses += batch_size_ratio * testLoss
        
    test = valLosses
    test_acc = test_acc_sum/(num_batches*batch_size)

    return  test, val_acc, valLosses/num_batches, test_acc, testLosses/num_batches        # dividing by num_batches correct? should be