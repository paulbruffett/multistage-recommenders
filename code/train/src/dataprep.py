import pandas as pd
import os
import pickle
import re
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--file-location", type=str)
    parser.add_argument("--output-location", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    path = args.file_location
    output_path = args.output_location
    
    print(path)
    print(os.listdir(path))
    os.makedirs(output_path+"/train_processed/", exist_ok=True)
    os.makedirs(output_path+"/test_processed/", exist_ok=True)


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
    

    for train_type in ['train','test']:

        if not os.listdir(output_path+"/%s_processed/" % train_type):
            print("Directory is empty")
        else:    
            print("Directory is not empty")
            continue

        file_size = 1000000
        current = []

        common = _convert_common_features(path+"/aliccp_%s/common_features_%s.csv" % (train_type, train_type), None)

        with open(path+"/aliccp_%s/sample_skeleton_%s.csv" % (train_type,train_type), "r") as skeleton:
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

                    df = df.rename(columns={'109_14':"user_categories", "110_14":"user_shops","127_14":"user_brands","150_14":"user_intentions",'121':'user_profile','122':'user_group',
                        "124":"user_gender","125":"user_age","126":"user_consumption","128":"user_is_occupied","129":"user_geography",
                        "205":"item_id","206":"item_category","210":"item_intention","216":"item_brand","508":"user_item_categories",
                        "509":"user_item_shops","702":"user_item_brands","853":"user_item_intentions","301":"position","207":"item_shop",
                        "127":"user_consumption_2","101":"user_id"})

                    index = int((i / file_size) - 1)
                    file_name = f"%s_{index}.parquet" % train_type
                    df.to_parquet(
                        os.path.join(output_path,"%s_processed" % train_type, file_name)
                    )
                    current = []