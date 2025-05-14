import trainExplainer_NeuroSAT
import sys
import numpy as np

datasets = [
    "NeuroSAT-soft"
]
#"NeuroSAT-hard", 
opts = {
        'out_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000',
        'logging': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/log/dataset_train_8_size4000.log',
        'val_reals_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_val_reals',
        'test_reals_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_test_reals',
        'val_gt_edges_per_problem': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_val_gt_edges_per_problem',
        'test_gt_edges_per_problem': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_test_gt_edges_per_problem',
        'n_pairs': 100,  # Anzahl der zu generierenden Paare
        'min_n': 8,
        'max_n': 8,
        'p_k_2': 0.3,
        'p_geo': 0.4,
        'max_nodes_per_batch': 4000,
        'one_pair': False,
        'emb_dim': 128,
        'iterations': 26,
    }


for dataset in datasets:
    testAUCs = []
    individual_aurocs_tests = []
    testInfTimes = []

    print(f"\nRunning experiment on dataset: {dataset}, with hyperparameters specified in configs")

    for i in range(10):
        mlp, downstreamTask, individual_aurocs_test = trainExplainer_NeuroSAT.trainExplainer(
            dataset=dataset,
            save_model=True,
            wandb_project=f"Final-NeuroSAT-Experiment-{dataset}",
            runSeed=i,
            opts=opts
        )
        individual_aurocs_tests.append(np.mean(individual_aurocs_test))

    # Convert to NumPy arrays for easier math
    individual_aurocs_tests = np.array(individual_aurocs_tests)

    # Summary stats
    print(f"Results for {dataset}:")
    print(f"  Individual AUROCs - Mean: {np.mean(individual_aurocs_tests):.4f}, Std: {np.std(individual_aurocs_tests):.4f}")