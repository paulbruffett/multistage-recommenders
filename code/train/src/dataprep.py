import pandas as pd
import os
import pickle
import re
import numpy as np
from tqdm import tqdm
from azureml.fsspec import AzureMachineLearningFileSystem

fs = AzureMachineLearningFileSystem('azureml://subscriptions/3bdbda93-8c3a-472b-bdde-25e3028fc307/resourcegroups/mlworkspace/workspaces/mlworkspace/datastores/mlworkspace3685223142/paths/aliccp_train')

print(fs.ls)

def _convert_common_features(common_path, pickle_path=None):
    common = {}

    with fs.open(common_path, "r") as common_features:
        for csv_line in tqdm(common_features, desc="Reading common features..."):
            line = csv_line.strip().split(",")
            kv = np.array(re.split("[]", line[2]))
            keys = kv[range(0, len(kv), 3)]
            values = kv[range(1, len(kv), 3)]
            common[line[0]] = dict(zip(keys, values))

        if pickle_path:
            with open(pickle_path, "wb") as handle:
                pickle.dump(common, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return common

file_size = 1000000
current = []

common = _convert_common_features("aliccp/common_features_train.csv", None)

with fs.open("aliccp/sample_skeleton_train.csv", "r") as skeleton:
    for i, csv_line in tqdm(enumerate(skeleton), desc="Processing data..."):

        line = csv_line.strip().split(",")
        if line[1] == "0" and line[2] == "1":
            continue
        kv = np.array(re.split("[]", line[5]))
        key = kv[range(0, len(kv), 3)]
        value = kv[range(1, len(kv), 3)]
        feat_dict = dict(zip(key, value))
        feat_dict.update(common[line[3]])
        feat_dict["click"] = int(line[1])
        feat_dict["conversion"] = int(line[2])

        current.append(feat_dict)

        if i > 0 and i % file_size == 0:
          df = pd.DataFrame(current)

          index = int((i / file_size) - 1)
          file_name = f"train_{index}.parquet"
          df.to_parquet(
              os.path.join(file_name)
          )
          c_client.upload_blob("aliccp/processed/"+file_name,file_name)
          current = []