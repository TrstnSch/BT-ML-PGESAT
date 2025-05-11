import trainExplainer_OneTrainAllEval
import sys
import numpy as np

datasets = [
    "BA-Community", "Tree-Grid", "BA-Shapes", "Tree-Cycles"
]

for dataset in datasets:
    testAUCs = []
    individual_aurocs_tests = []
    testInfTimes = []

    print(f"\nRunning OneNodeTrainAllEval experiment on dataset: {dataset}, with hyperparameters specified in configs")

    for i in range(10):
        mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime = trainExplainer_OneTrainAllEval.trainExplainer(
            dataset=dataset,
            save_model=False,
            wandb_project=f"Experiment-OneNodeTrainAllEval-{dataset}",
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