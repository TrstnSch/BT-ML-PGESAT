import trainExplainer
import sys

mlp, downstreamTask = trainExplainer.trainExplainer(dataset=sys.argv[1], save_model=sys.argv[2])