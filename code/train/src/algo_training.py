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

    # run main function
    path = args.file_location
    log_path = args.output_location
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path+"/logs")

    df = pd.read_parquet(path+'/train_processed/')

    print(len(df))

    features = ['item_id','user_id']
    for i in features:
        df[i] = pd.to_numeric(df[i], errors='coerce',downcast="integer")

    df_features = df[df['click']==1][['item_id','user_id']]
    df_features= df_features.dropna()
    df_tensor = tf.convert_to_tensor(df_features, dtype=tf.int64)

    f_dataset = tf.data.Dataset.from_tensor_slices(df_tensor)

    len_train = round(len(f_dataset)*.2)
    len_test = round(len(f_dataset)*.2)

    tf.random.set_seed(42)
    shuffled = f_dataset.shuffle(len(f_dataset), seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(len_train)
    test = shuffled.skip(len_train).take(len_test)

    unique_items = np.unique(df_features['item_id'].values.squeeze())
    unique_users = np.unique(df_features['user_id'].values.squeeze())

    unique_items = tf.convert_to_tensor(unique_items,dtype=tf.int32)
    unique_users = tf.convert_to_tensor(unique_users,dtype=tf.int32)

    item_ids = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(df_features['item_id'].values,dtype=tf.int64))

    embedding_dimension = 32
    mlflow.log_param("embedding_dimension", embedding_dimension)

    user_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=unique_users, mask_token=None),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_users) + 1, embedding_dimension)
    ])

    item_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=unique_items, mask_token=None),
    tf.keras.layers.Embedding(len(unique_items) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(candidates=item_ids.batch(128).map(item_model))

    task = tfrs.tasks.Retrieval(metrics=metrics)

    class RetrievalModel(tfrs.Model):
        def __init__(self, user_model, movie_model):
            super().__init__()
            self.item_model: tf.keras.Model = item_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features, training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features['user_id'])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.item_model(features['item_id'])

            # The task computes the loss and the metrics.
            return self.task(user_embeddings, positive_movie_embeddings)
        
    model = RetrievalModel(user_model, item_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


    train_dataset = train.map(lambda x: {
        "user_id": x[1],
        "item_id": x[0],
    })

    cached_train = train_dataset.batch(8192).cache()

    model.fit(cached_train, epochs=1,callbacks=[tensorboard_callback])