import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

download = [2, 4, 5, 7, 12, 16, 20, 22, 23, 24, 28, 29, 31, 32, 36, 37, 38, 42, 44, 46, 50, 54, 60, 151, 182, 183, 188,
            300, 307, 312, 333, 334, 335, 375, 377, 451, 458, 469, 470, 1038, 1046, 1049, 1050, 1053, 1063, 1067, 1068,
            1112, 1114, 1116, 1120, 1220, 1459, 1461, 1462, 1464, 1466, 1467, 1471, 1475, 1478, 1480, 1487, 1489, 1491,
            1492, 1493, 1494, 1497, 1501, 1504, 1510, 1570, 1590, 4134, 4135, 4534, 4538, 6332, 23380, 23381, 40496,
            40499, 40536]

BASE = "http://openml.org/api/v1/json"
all_qualities = []
for dataset_id in tqdm(download):
    data = requests.get(BASE + f"/data/qualities/{dataset_id}").json()
    qualities = data['data_qualities']['quality']
    converted_qualities = {i['name']: i['value'] for i in qualities if not (isinstance(i['value'], list) or np.isnan(float(i['value'])))}
    all_qualities.append(converted_qualities)
pd.DataFrame(all_qualities, index=download).to_csv("cache/metafeatures2.csv")
