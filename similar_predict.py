import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arbok import ParamPreprocessor
from lightgbm import LGBMRegressor, plot_importance
from scipy.stats import kendalltau
from sklearn import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Settings

flows = [8315, 6970, 8317, 6969]
cache_dir = "cache"
results_dir = "results"
print(os.path.realpath(__file__))

# Get metafeatures
metafeatures = pd.read_csv(os.path.join(cache_dir, "metafeatures2.csv"), index_col=0)


# Functions
def store_json(data, filename):
    with open(os.path.join(results_dir, filename), "w+") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def get_n_similar(n_neighbors, meta_per_task, train_groups, test_groups):
    nn = NearestNeighbors()
    nn.fit(meta_per_task.loc[train_groups].fillna(meta_per_task.loc[train_groups].mean()))
    dist, neighbors = nn.kneighbors(meta_per_task.loc[test_groups].fillna(0), n_neighbors)
    similar = meta_per_task.loc[train_groups].index[neighbors.reshape(-1)].tolist()
    return dist, similar


all_results = {flow: {} for flow in flows}
for flow_id in flows:

    # Load files
    groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
    params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
    # metas = pd.read_json(os.path.join(cache_dir, f"{flow_id}_meta.json")).sort_index()
    metas = metafeatures.loc[groups].reset_index(drop=True)
    metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()
    unique_groups = np.unique(groups)

    # Standardize meta_per_task
    task_index = metafeatures.index
    task_columns = metafeatures.columns
    meta_per_task = StandardScaler().fit_transform(metafeatures)
    meta_per_task = pd.DataFrame(meta_per_task, index=task_index, columns=task_columns)

    # Rescale kappa
    metrics = metrics.astype(float)
    metrics["kappa"] = metrics["kappa"] / 2 + 0.5

    # Setup preprocessing
    # param_preprocessing = ParamPreprocessor(names=params.columns)

    # Transforming
    # params_transformed = param_preprocessing.fit_transform(params)
    # params_transformed_frame = pd.DataFrame(params_transformed, columns=param_preprocessing.names)
    params_transformed_frame = pd.get_dummies(params)

    # Combine data
    combined = pd.concat([params_transformed_frame, metas], axis=1, sort=False)
    X_params = np.array(params_transformed_frame).astype(float)
    X_metas = np.array(metas).astype(float)
    X = np.array(combined.values).astype(float)
    types = np.zeros(X.shape[1])
    bounds = np.array([(np.min(i), np.max(i)) for i in X.T])
    num_params = X_params.shape[1]
    num_metas = X_metas.shape[1]

    y = np.array(metrics["predictive_accuracy"].sort_index())

    # Convert y
    y_converted = np.zeros_like(y)
    for g in unique_groups:
        indices = groups == g
        selection = y[indices]
        y_converted[indices] = (selection - np.mean(selection)) / np.std(selection)
        # y_converted[indices] = selection - np.mean(selection)

    # Pre train regressors
    # estimators = {}
    # for g in unique_groups:
    #     indices = groups == g
    #     trained = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=4).fit(X[indices], y_converted[indices])
    #     estimators[g] = trained

    logo = LeaveOneGroupOut()

    correlations = []
    for train_index, test_index in tqdm(logo.split(X, y, groups)):

        # Get groups
        test_groups = np.unique(groups[test_index])
        train_groups = np.unique(groups[train_index])

        # Get most similar tasks
        distance, tasks = get_n_similar(3, meta_per_task, train_groups, test_groups)
        task_indices = np.any([groups == t for t in tasks], axis=0)
        regressor = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=4).fit(X_params[task_indices], y_converted[task_indices])
        # regressor = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=4, min_child_samples=1).fit(combined.iloc[train_index], y_converted[train_index])
        plot_importance(regressor, max_num_features=10)
        plt.show()
        predictions = regressor.predict(X_params[test_index])
        # votes = []
        # for t in tasks:
        #     p = estimators[t].predict(X[test_index])
        #     votes.append(p)

        # predictions = np.mean(votes, axis=0)
        correlation, p_value = kendalltau(y[test_index], predictions)
        correlations.append(correlation)
        print(len(train_groups), test_groups, correlation)

    correlation = np.mean(correlations)
    print(f"Mean correlation of flow {flow_id}: {correlation}")
