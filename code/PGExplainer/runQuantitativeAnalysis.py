import trainExplainer_trainSplit
import sys
import numpy as np

for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="BA-Shapes", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-BA-Shapes", runSeed=i)
    #TODO: Return final mean_test_auc for each run. Calculate mean and std of all runs. 
    
for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="BA-Community", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-BA-Community", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="Tree-Cycles", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-Tree-Cycles", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="Tree-Grid", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-Tree-Grid", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="BA-2Motif", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-BA-2Motif", runSeed=i)
    
for i in range (10):
    mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(dataset="MUTAG", save_model=False, wandb_project="Replication-bestDT-oldSweep-trainSplit-MUTAG", runSeed=i)
    

datasets = [
    "BA-Shapes", "BA-Community", "Tree-Cycles",
    "Tree-Grid", "BA-2Motif", "MUTAG"
]

for dataset in datasets:
    testAUCs = []
    individual_aurocs_tests = []
    testInfTimes = []

    print(f"\nRunning experiments on dataset: {dataset}, with hyperparameters specified in configs")

    for i in range(10):
        mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(
            dataset=dataset,
            save_model=False,
            wandb_project=f"Final-Replication-Inductive-{dataset}",
            runSeed=i
        )
        testAUCs.append(testAUC)
        individual_aurocs_tests.append(np.mean(individual_aurocs_test))
        testInfTimes.append(testInfTime)

    # Convert to NumPy arrays for easier math
    testAUCs = np.array(testAUCs)
    individual_aurocs_tests = np.array(individual_aurocs_tests)
    testInfTimes = np.array(testInfTimes)

    # Summary stats
    print(f"Results for {dataset}:")
    print(f"  Test AUC      - Mean: {np.mean(testAUCs):.4f}, Std: {np.std(testAUCs):.4f}")
    print(f"  Individual AUROCs - Mean: {np.mean(individual_aurocs_tests):.4f}, Std: {np.std(individual_aurocs_tests):.4f}")
    print(f"  Inference Time    - Mean: {np.mean(testInfTimes):.4f}s, Std: {np.std(testInfTimes):.4f}s")