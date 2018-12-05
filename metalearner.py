import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.stats import rankdata, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Settings
flows = [8315, 6970, 8317, 6969]
cache_dir = "cache"
results_dir = "new_results"
print(os.path.realpath(__file__))

# Get metafeatures
metafeatures = pd.read_csv(os.path.join(cache_dir, "metafeatures2.csv"), index_col=0)

# Functions
def store_json(data, filename):
    with open(os.path.join(results_dir, filename), "w+") as f:
        json.dump(data, f, indent=4, sort_keys=True)

result = {}
for num_leaves in [4, 8, 16, 32]:
    for mode in ["default", "scaled"]:
        for flow_id in flows:
            result[flow_id] = {}

            # Load files
            groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
            params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
            metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()
            # metas = pd.read_json(os.path.join(cache_dir, f"{flow_id}_meta.json")).sort_index()
            # metas = pd.read_csv(os.path.join(cache_dir, f"metafeatures2.json"))
            metas = metafeatures.loc[groups].reset_index(drop=True)

            # Sorting
            indices = groups.argsort()
            groups = groups.iloc[indices].reset_index(drop=True)
            params = params.iloc[indices].reset_index(drop=True)
            metrics = metrics.iloc[indices].reset_index(drop=True)
            metas = metas.iloc[indices].reset_index(drop=True)
            unique_groups = np.unique(groups)

            # Rescale kappa
            metrics = metrics.astype(float)
            metrics["kappa"] = metrics["kappa"] / 2 + 0.5

            # Converting
            params = pd.get_dummies(params)

            # Get data
            data = {
                "params": np.array(params),
                "metas": np.array(metas),
                "both": np.array(pd.concat([params, metas], axis=1, sort=False))
            }

            # Regressor and LOGO
            estimator = LGBMRegressor(n_estimators=500, num_leaves=num_leaves, learning_rate=0.05, min_child_samples=1, verbose=-1)
            logo = LeaveOneGroupOut()

            for metric in metrics.columns:
                result[flow_id][metric] = {}
                y = np.array(metrics[metric])

                # Convert y
                if mode == "scaled":
                    y_converted = np.zeros_like(y)
                    for g in unique_groups:
                        indices = groups == g
                        selection = y[indices]
                        y_converted[indices] = (selection - np.mean(selection)) / np.std(selection)
                else:
                    y_converted = y

                for kind, X in data.items():
                    result[flow_id][metric][kind] = {}
                    correlations = []
                    mse_losses = []
                    rmse_losses = []
                    for train_index, test_index in tqdm(logo.split(X, y_converted, groups)):
                        estimator.fit(X[train_index], y_converted[train_index])
                        y_pred = estimator.predict(X[test_index])
                        correlation, p_value = kendalltau(y_converted[test_index], y_pred)
                        correlation = 0 if np.isnan(correlation) else correlation
                        mse = mean_squared_error(y_converted[test_index], y_pred)
                        correlations.append(correlation)
                        mse_losses.append(mse)
                        rmse_losses.append(np.sqrt(mse))
                    result[flow_id][metric][kind]["correlation"] = np.mean(correlations)
                    result[flow_id][metric][kind]["mse"] = np.mean(mse_losses)
                    result[flow_id][metric][kind]["rmse"] = np.mean(rmse_losses)
                    store_json(result, f"result-{mode}-{num_leaves}.json")