import json
import os
import time

import numpy as np
import pandas as pd
from arbok import ParamPreprocessor
from lightgbm import LGBMRegressor
from scipy.stats import kendalltau
from sklearn import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Settings

flows = [8315, 6970, 8317, 6969]
cache_dir = "cache"
results_dir = "results"
print(os.path.realpath(__file__))


# Functions
def store_json(data, filename):
    with open(os.path.join(results_dir, filename), "w+") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def get_n_similar(n_neighbors, meta_per_task, train_groups, test_groups):
    nn = NearestNeighbors()
    nn.fit(meta_per_task.loc[train_groups])
    dist, neighbors = nn.kneighbors(meta_per_task.loc[test_groups], n_neighbors)
    similar = meta_per_task.loc[train_groups].index[neighbors.reshape(-1)].tolist()
    return dist, similar


all_results = {flow: {} for flow in flows}
for flow_id in flows:

    # Load files
    groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
    params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
    metas = pd.read_json(os.path.join(cache_dir, f"{flow_id}_meta.json")).sort_index()
    metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()

    # Get metafeatures per task
    meta_per_task = metas.copy()
    meta_per_task["task"] = groups
    meta_per_task = meta_per_task.groupby("task").first()

    # Rescale kappa
    metrics = metrics.astype(float)
    metrics["kappa"] = metrics["kappa"] / 2 + 0.5

    # Setup preprocessing
    param_preprocessing = ParamPreprocessor(names=params.columns)

    # Transforming
    params_transformed = param_preprocessing.fit_transform(params)
    params_transformed_frame = pd.DataFrame(params_transformed, columns=param_preprocessing.names)

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

    # Pre train regressors
    estimators = {}
    for g in np.unique(groups):
        indices = groups == g
        selection = y[indices]
        y_converted = (selection - np.mean(selection)) / np.std(selection)
        trained = LGBMRegressor(num_leaves=4).fit(X[indices], y_converted[indices])
        estimators[g] = clone(trained)


    flow_results = {name: {} for name, regressor in regressors}
    for name, regressor in regressors:
        regressor_results = {metric: {} for metric in metrics.columns}
        for metric in metrics.columns[::-1]:
            y = np.array(metrics[metric])
            logo = LeaveOneGroupOut()
            correlations = []
            fitting_times = []
            prediction_times = []

            # Converting
            y_converted = np.zeros_like(y)
            for g in np.unique(groups):
                indices = groups == g
                selection = y[indices]
                y_converted[indices] = (selection - np.mean(selection)) / np.std(selection)

            for train_index, test_index in tqdm(logo.split(X, y, groups)):
                test_groups = np.unique(groups[test_index])
                train_groups = np.unique(groups[train_index])

                # Get most similar tasks
                distance, tasks = get_n_similar(5, meta_per_task, train_groups, test_groups)

                for i in tasks:


                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                yc_train, yc_test = y_converted[train_index], y_converted[test_index]
                g_train, g_test = groups[train_index], groups[test_index]

                # # Converting
                # y_converted = np.zeros_like(y_train)
                # for g in np.unique(g_train):
                #     indices = g_train == g
                #     selection = y_train[indices]
                #     y_converted[indices] = (selection - np.mean(selection)) / np.std(selection)

                # r = build_params_model(num_params)
                # r1 = LGBMRegressor()
                # r2 = LGBMRegressor()

                # Fitting
                time_0 = time.time()
                # intermediate = cross_val_predict(r1, X[train_index], y_train, groups=g_train, cv=GroupKFold(10))
                # residuals = y_train - intermediate
                # r1.fit(X_params[train_index], yc_train)
                # r2.fit(X[train_index], y_train)
                # meta_scaler = StandardScaler()
                # param_scaler = StandardScaler()
                # X_metas_t = meta_scaler.fit_transform(X_metas[train_index])
                # X_params_t = param_scaler.fit_transform(X_params[train_index])
                # r.fit(X_params_t, yc_train, epochs=1, batch_size=500, verbose=0)
                regressor.fit(X_params[train_index], yc_train)

                # r.fit(
                #     {"metas": X_metas_t, "params": X_params_t},
                #     {"main_output": y_converted[train_index], "param_output": y_converted[train_index]},
                #     epochs=10,
                #     batch_size=500
                # )

                # Predicting
                time_1 = time.time()
                # p1 = r1.predict(X_params[test_index])
                # p2 = r2.predict(X[test_index])
                # predictions = p1 + p2
                # X_metas_t = meta_scaler.transform(X_metas[test_index])
                # X_params_t = param_scaler.transform(X_params[test_index])
                # predictions = r.predict(X_params_t)
                predictions = regressor.predict(X_params[test_index])
                # predictions = r.predict(
                #     {"metas": X_metas_t, "params": X_params_t}
                # )[0]
                # print(predictions)
                time_2 = time.time()

                # Times
                fitting_times.append(time_1 - time_0)
                prediction_times.append(time_2 - time_1)

                # Correlation
                correlation, p_value = kendalltau(y_test, predictions)
                correlations.append(correlation)
                g = np.unique(groups[test_index])
                print(g, correlation)

            correlation = np.mean(correlations)
            fitting_time = np.mean(fitting_times)
            prediction_time = np.mean(prediction_times)
            regressor_results[metric] = {
                "mean_correlation": correlation,
                "mean_fitting_time": fitting_time,
                "mean_prediction_time": prediction_time,
                "correlations": correlations,
                "fitting_times": fitting_times,
                "prediction_times": prediction_times,
            }
            positive = np.mean(np.array(correlations) > 0)
            print(f"Flow {flow_id} using {name} on {metric}")
            print(f"Kendall's tau: {correlation} | Positive: {positive}%",
                  f"| Fitting (s): {fitting_time} | Predicting (s): {prediction_time}"
                  )
        store_json(regressor_results, f"{flow_id}-{name}.json")
        flow_results[name] = regressor_results
