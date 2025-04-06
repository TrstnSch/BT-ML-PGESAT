import trainExplainer_trainSplit
import sys

for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="BA-Shapes", save_model=False, wandb_project="Replication-Seeded-trainSplit-sweepConfig-BA-Shapes", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="BA-Community", save_model=False, wandb_project="Replication-Seeded-trainSplit-sweepConfig-BA-Community", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="Tree-Cycles", save_model=False, wandb_project="Replication-Seeded-trainSplit-sweepConfig-Tree-Cycles", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="Tree-Grid", save_model=False, wandb_project="Replication-Seeded-trainSplit-sweepConfig-Tree-Grid", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="BA-2Motif", save_model=False, wandb_project="Replication-Seeded-trainSplit-NOsweepConfig-BA-2Motif", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask = trainExplainer_trainSplit.trainExplainer(dataset="MUTAG", save_model=False, wandb_project="Replication-Seeded-trainSplit-sweepConfig-MUTAG", runSeed=i)