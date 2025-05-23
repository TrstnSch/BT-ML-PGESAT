import trainExplainer_trainSplit
import sys
import numpy as np

datasets = [
    "BA-2Motif"
]
#"BA-Shapes", "BA-Community" , "Tree-Cycles", "Tree-Grid", , "MUTAG"
for dataset in datasets:
    testAUCs = []
    individual_aurocs_tests = []
    testInfTimes = []

    print(f"\nRunning experiment on dataset: {dataset}, with hyperparameters specified in configs")

    for i in range(10):
        mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_trainSplit.trainExplainer(
            dataset=dataset,
            save_model=True,
            wandb_project=f"PLOTS-HE-Final-Replication-Inductive-{dataset}",
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