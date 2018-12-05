import os

import numpy as np
import pandas as pd
from arbok import ParamPreprocessor
from lightgbm import LGBMRegressor
from scipy.stats import kendalltau
from sklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

flow_id = 6840
cache_dir = "cache"
results_dir = "results"
print(os.path.realpath(__file__))

groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
metas = pd.read_json(os.path.join(cache_dir, f"{flow_id}_meta.json")).sort_index()
metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()

param_preprocessing = ParamPreprocessor(names=params.columns)
params_transformed = param_preprocessing.fit_transform(params)
params_t = pd.DataFrame(params_transformed, columns=param_preprocessing.names)

# Combine data
combined = pd.concat([params_t, metas], axis=1, sort=False)
X_params = np.array(params_t).astype(float)
X_metas = np.array(metas).astype(float)
X = np.array(combined.values).astype(float)
types = np.zeros(X.shape[1])
bounds = np.array([(np.min(i), np.max(i)) for i in X.T])

metrics = metrics.astype(float)
y = np.array(metrics["predictive_accuracy"])
y_converted = np.zeros_like(y)
for g in np.unique(groups):
    indices = groups == g
    selection = y[indices]
    y_converted[indices] = (selection - np.mean(selection)) / np.std(selection)

# Select where numerical strategy equals mean
# selection = np.array(params["conditionalimputer__strategy"] == "mean")
# X = X[selection]
# y = y[selection]
# groups = groups[selection]

logo = LeaveOneGroupOut()
splits = [(i, j) for i, j in logo.split(X, y, groups)]

grid = {
    "n_estimators": [500],
    "learning_rate": [0.05],
    "num_leaves": [4],
    "reg_alpha": [0],
    "reg_lambda": [0],
    # "min_child_samples": [20],
    # "max_depth": [4, 6, 8, 10, 12]
}


def kendallscorer(y_true, y_pred):
    correlation, p_value = kendalltau(y_true, y_pred)
    if np.isnan(correlation):
        return 0
    return correlation


scorer = make_scorer(kendallscorer)
g = GridSearchCV(LGBMRegressor(verbose=-1), param_grid=grid, cv=splits, verbose=10, scoring=scorer)
g.fit(X_params, y_converted)
a = pd.DataFrame(g.cv_results_["params"])
a["train_score"] = g.cv_results_["mean_train_score"]
a["test_score"] = g.cv_results_["mean_test_score"]
print(a["test_score"])
