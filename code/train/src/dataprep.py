import pandas as pd
import os
import pickle
import re
import numpy as np
from tqdm import tqdm

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = os.getenv('CONNECTION_STRING')

blob_service_client = BlobServiceClient.from_connection_string(connect_str)

cclient = blob_service_client.get_container_client("azureml-blobstore-64879d37-44cb-403b-b850-a7bc6761c022")


blob_stream = cclient.download_blob("aliccp_train/common_features_train.csv")
with open(file="common_features_train.csv", mode="wb") as download_file:
    download_file.write(blob_stream.readall())

blob_stream = cclient.download_blob("aliccp_train/sample_skeleton_train.csv.csv")
with open(file="sample_skeleton_train.csv.csv", mode="wb") as download_file:
    download_file.write(blob_stream.readall())

def _convert_common_features(common_path, pickle_path=None):
    common = {}

    with open(common_path, "r") as common_features:
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

common = _convert_common_features("common_features_train.csv", None)

with open("sample_skeleton_train.csv", "r") as skeleton:
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