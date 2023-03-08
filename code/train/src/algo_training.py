import os
import pprint
import tempfile
import argparse

from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import tensorflow_recommenders as tfrs
import datetime
import gc


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

    mlflow.tensorflow.autolog()

    #get hour and minute as string
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # run main function
    path = args.file_location
    log_path = args.output_location
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path+"/logs/"+now)

    df_train = pd.read_parquet(path+'/train_processed/')
    print(len(df_train))

    #filter out long tail products and reduce size of dataset
    df_train['counter'] = 1
    df_agg = df_train[['item_id','counter']].groupby('item_id').count()
    df_agg = df_agg.sort_values("counter",ascending=False)[:10000]
    df_train = pd.merge(df_agg, df_train, on=['item_id', 'item_id'], how='left')
    print(len(df_train))
    df_train['counter'] = 1
    cust_agg = df_train[['user_id','counter']].groupby('user_id').count()
    cust_agg = cust_agg.sort_values("counter",ascending=False)[:8000]
    df_train = pd.merge(cust_agg, df_train, on=['user_id', 'user_id'], how='left')
    print(len(df_train))

    train = df_train[df_train['click']==1][['item_id','user_id']]
    features = ['item_id','user_id']
    for i in features:
        train[i] = pd.to_numeric(train[i], errors='coerce',downcast="integer")
    train= train.dropna()

    items = train['item_id'].unique()
    items = tf.convert_to_tensor(items, dtype=tf.int64)

    train = tf.convert_to_tensor(train, dtype=tf.int64)


    train = tf.data.Dataset.from_tensor_slices(train)

    interactions = train.map(lambda x: {
        "item_id": x[0],
        "user_id": x[1],
    })

    items_ds = tf.data.Dataset.from_tensor_slices(items)

    tf.random.set_seed(42)
    shuffled = interactions.shuffle(100000, seed=42, reshuffle_each_iteration=False)

    split = round(len(shuffled)*.8)

    train = shuffled.take(split)
    test = shuffled.skip(split).take(len(shuffled)-split)

    item_batch = items_ds.batch(1_000)
    user_ids = interactions.batch(1_000_000).map(lambda x: x["user_id"])

    unique_items = np.unique(np.concatenate(list(item_batch)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 64

    user_model = tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

    item_model = tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_items, mask_token=None),
        tf.keras.layers.Embedding(len(unique_items) + 1, embedding_dimension)
        ])
    
    metrics = tfrs.metrics.FactorizedTopK(candidates=items_ds.batch(128).map(item_model))

    task = tfrs.tasks.Retrieval(metrics=metrics)

    class RecommendationModel(tfrs.Model):

        def __init__(self, user_model, item_model):
            super().__init__()
            self.item_model: tf.keras.Model = item_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features["user_id"])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.item_model(features["item_id"])

            # The task computes the loss and the metrics.
            return self.task(user_embeddings, positive_movie_embeddings)


    model = RecommendationModel(user_model, item_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=15)

    model.evaluate(cached_test, return_dict=True)

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model,k = 50)

    index.index_from_dataset(tf.data.Dataset.zip((items_ds.batch(100), items_ds.batch(100).map(model.item_model))))

    _, titles = index(tf.constant([42]))
    print(titles)
    tf.saved_model.save(index, "index")
    mlflow.log_artifacts("index")