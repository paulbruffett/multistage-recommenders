import os
import pprint
import tempfile
import argparse

from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--file-location", type=str)

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

    df = pd.read_parquet(path+'/train_processed/')

    print(len(df))