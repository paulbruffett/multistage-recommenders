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

def dataset_to_tensor(df):
    features = ['item_id','user_id']
    for i in features:
        df[i] = pd.to_numeric(df[i], errors='coerce',downcast="integer")

    df_features = df[df['click']==1][['item_id','user_id']]
    df_features= df_features.dropna()
    df_tensor = tf.convert_to_tensor(df_features, dtype=tf.int64)

    f_dataset = tf.data.Dataset.from_tensor_slices(df_tensor)
    return f_dataset, df_features

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

    train = pd.read_parquet(path+'/train_processed/')


    #filter out long tail products and reduce size of dataset
    train['counter'] = 1
    df_agg = train[['item_id','counter']].groupby('item_id').count()
    df_agg = df_agg[df_agg['counter']>500]
    train = pd.merge(df_agg, train, on=['item_id', 'item_id'], how='left')
    print(len(train))

    train, df_features = dataset_to_tensor(train)


    tf.random.set_seed(42)
    train = train.shuffle(len(train), seed=42, reshuffle_each_iteration=False)

    
    #train = train.take(round(len(train)*0.01))

    unique_items = np.unique(df_features['item_id'].values.squeeze())
    unique_users = np.unique(df_features['user_id'].values.squeeze())

    unique_items = tf.convert_to_tensor(unique_items,dtype=tf.int32)
    unique_users = tf.convert_to_tensor(unique_users,dtype=tf.int32)

    item_ids = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(df_features['item_id'].values,dtype=tf.int64))

    embedding_dimension = 64
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
        def __init__(self, user_model, item_model):
            super().__init__()
            self.item_model: tf.keras.Model = item_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features, training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features['user_id'])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_item_embeddings = self.item_model(features['item_id'])

            # The task computes the loss and the metrics.
            return self.task(user_embeddings, positive_item_embeddings)
        
    model = RetrievalModel(user_model, item_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


    train_dataset = train.map(lambda x: {
        "user_id": x[1],
        "item_id": x[0],
    })

    cached_train = train_dataset.batch(512).cache()

    model.fit(cached_train, epochs=15,callbacks=[tensorboard_callback])

    del df_features,train,train_dataset,cached_train
    gc.collect()

    test = pd.read_parquet(path+'/test_processed/')
    test = pd.merge(df_agg, test, on=['item_id', 'item_id'], how='left')
    test, _ = dataset_to_tensor(test)
    #test = test.take(round(len(test)*0.01))

    test_dataset = test.map(lambda x: {
        "user_id": x[1],
        "item_id": x[0],
    })

    cached_test = test_dataset.batch(512).cache()

    model.evaluate(cached_test, return_dict=True, callbacks=[tensorboard_callback])

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model,k = 50)

    index.index_from_dataset(tf.data.Dataset.zip((item_ids.batch(100), item_ids.batch(100).map(model.item_model))))

    _, titles = index(tf.constant([42]))
    print(titles)
    tf.saved_model.save(index, "index")
    mlflow.log_artifacts("index", "index")