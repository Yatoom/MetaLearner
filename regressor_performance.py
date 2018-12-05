import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(flow):
    files = glob.glob(f"results/*{flow}*.json")
    all_data = {}
    for file in files:
        basename, extension = os.path.splitext(os.path.basename(file))
        label = basename.replace(str(flow), "")[1:]
        with open(file, "rb+") as f:
            data = json.load(f)
            all_data[label] = data
    return all_data


def rank_losses(data):
    correlations = {
        name: {metric: np.nan_to_num(info["correlations"]) for metric, info in data.items()}
        for name, data in data.items()
    }
    switched = pd.DataFrame(correlations).T.to_dict()
    ranked_switched = {metric: pd.DataFrame(d).rank(axis=1).mean().to_dict() for metric, d in switched.items()}
    ranked = pd.DataFrame(ranked_switched).T
    return ranked


def get_mean(data, value):
    means = {name: {metric: np.mean(np.nan_to_num(info["correlations"])) for metric, info in data.items()} for name, data in
             data.items()}
    stds = {name: {metric: np.var(np.nan_to_num(info["correlations"])) for metric, info in data.items()} for name, data in
             data.items()}
    frame_means = pd.DataFrame(means)
    frame_std = pd.DataFrame(stds)
    return frame_means, frame_std


def plot(frame_mean, frame_std, title):
    # Sort by mean rank
    # frame = frame_mean.reindex(frame_mean.mean().sort_values(ascending=False).index, axis=1)
    frame = frame_mean.sort_values("predictive_accuracy", axis=1)

    # Plot the bar
    frame_std = None
    frame.plot.barh(width=0.95, cmap="tab20b", figsize=(9, 9), xerr=frame_std)

    # Get the axis
    axis = plt.gca()
    handles, labels = axis.get_legend_handles_labels()

    # Invert the legend to match the
    axis.legend(handles[::-1], labels[::-1])
    axis.set_xlim(0, np.max(np.max(frame)) * 3 / 1.8)
    axis.set_title(title)

    for patch in axis.patches:
        x_pos = patch.get_width()
        # x_pos = 0
        y_pos = patch.get_y() + patch.get_height() / 4
        width = patch.get_width()
        text = f"{width:4.4}"
        axis.text(x_pos, y_pos, text, fontsize=10, color='dimgray')

    plt.show()
    print()


if __name__ == "__main__":
    correlations = []
    for flow in [6970, 8317, 8315, 6969]:
    # for flow in ["6970"]:
        data = get_data(flow)
        # ranked_losses = rank_losses(data)
        mean_correlation, std_correlation = get_mean(data, "loss")
        correlations.append(mean_correlation)
        # mean_fitting = get_mean(data, "fitting_time")
        # mean_prediction = get_mean(data, "prediction_time")

        # plot(ranked_losses, f"Ranked loss over flow {flow}")
        plot(mean_correlation, std_correlation, f"Mean correlation over flow {flow}")
        # mean_fitting = mean_fitting.drop(["Std", "Mean strategy", "Unbiased LightGBM", "Unbiased XGBoost",
        #                                   "Unbiased Extra Trees", "Unbiased CatBoost (lr=0.1)"], axis=1)
        # plot(mean_fitting, f"Mean fitting time over flow {flow}")
        # plot(mean_prediction, f"Mean prediction time over flow {flow}")
    print()
