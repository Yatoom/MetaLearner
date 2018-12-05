import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import rankdata, kendalltau
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Settings
flows = [8315, 6970, 8317, 6969]
cache_dir = "cache"
results_dir = "results"
print(os.path.realpath(__file__))


# Functions
def store_json(data, filename):
    with open(os.path.join(results_dir, filename), "w+") as f:
        json.dump(data, f, indent=4, sort_keys=True)


for flow_id in flows:
    # Load files
    groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
    params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
    metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()

    # Get dummies and query
    frame = pd.get_dummies(params)
    frame["group"] = groups
    frame["acc"] = metrics["predictive_accuracy"]
    frame = frame.sort_values("group").reset_index(drop=True)

    # Get data
    groups = frame.pop("group")
    y = frame.pop("acc")
    X = frame

    # Convert to numpy
    groups = np.array(groups)
    y = np.array(y)
    X = np.array(X)
    unique_groups = np.unique(groups)

    # Rank data
    ranked_y = np.zeros_like(y)
    for g in unique_groups:
        indices = groups == g
        ranks = rankdata(y[indices])
        ranked_y[indices] = np.array(ranks / np.max(ranks) * 1000).astype(int)

    # Ranker
    ranker = LGBMRanker(n_estimators=500, learning_rate=0.05, num_leaves=16, label_gain=np.arange(0, 1001, 1))

    logo = LeaveOneGroupOut()

    correlations = []
    for train_index, test_index in tqdm(logo.split(X, y, groups)):
        unique, counts = np.unique(groups[train_index], return_counts=True)
        ranker.fit(X[train_index], ranked_y[train_index], group=counts)
        predictions = ranker.predict(X[test_index])
        correlation, p_value = kendalltau(ranked_y[test_index], predictions)
        print(np.unique(groups[test_index]), correlation)
        correlations.append(correlation)
    print("Mean correlation: ", np.mean(correlations))