import trainExplainer
import sys

for i in range (10):
    mlp, downstreamTask = trainExplainer.trainExplainer(dataset=sys.argv[1], save_model=sys.argv[2], wandb_project="Replication-Seeded", runSeed=i)