import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

with open("results/result-8.json", "r") as f:
    dictionary = json.load(f)

for evaluation in ["correlation", "mse"]:
    f, axarr = plt.subplots(4, sharex=True, figsize=(6,11))


    for i, flow in enumerate(dictionary):
        frame = pd.DataFrame({i: {set_name: set_features[evaluation] for set_name, set_features in j.items()} for i, j in dictionary[flow].items()})
        frame.T.plot.barh(ax=axarr[i])
        axarr[i].set_title(f"{evaluation} of flow {flow}\n")

        # Shrink current axis's height by 10% on the bottom
        # box = axarr[i].get_position()
        # axarr[i].set_position([box.x0, box.y0 + box.height * 0.1,
        #                        box.width, box.height * 0.9])

        # Put a legend below current axis
        axarr[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                        fancybox=True, shadow=True, ncol=5)

    plt.show()
        # values = {i: j["correlation"] for i, j in dictionary[flow][metric].items()}
        # pd.DataFrame(values).plot.barh()
        # plt.show()
        # print()

# frame = pd.read_json("results/result.json")
# print()