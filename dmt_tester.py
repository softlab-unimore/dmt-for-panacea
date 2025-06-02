from skmultiflow.data import FileStream
from DMT.DMT import DynamicModelTree

from util import get_metrics, save_metrics
from data_utils.load_dataset import get_dataset
from data_utils.preprocessing import preprocess
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional
from sklearn.cluster import KMeans

import time
import os
import numpy as np
import pandas as pd
import pickle

num_classes = {
    "CICIDS2017": 2,
    "IDS2018": 2,
    "Kitsune": 2,
    "rKitsune": 2,
    "mKitsune": 2
}

def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='CICIDS2017')
    args.add_argument('--batch_size', type=int, default=None)
    args.add_argument('--delay', type=int, default=10)
    args.add_argument('--sampling', type=float, default=0.1)
    args = args.parse_args()

    return args

# copy of corresponding function in main.py
def find_best_kmeans(data_train):
    list_kmeans = []
    distorsions = []

    for k in range(1, 7):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_train)
        list_kmeans.append(kmeans)
        distorsions.append(kmeans.inertia_)

    first_derivative = np.diff(distorsions)
    optimal_k = np.argmin(first_derivative) + 2

    try:
        kmeans = list_kmeans[optimal_k]
    except IndexError:
        kmeans = list_kmeans[-1]

    return kmeans

# copy of corresponding function in main.py
def find_data_sampled(data_train, labels_train, kmeans, sampling_perc):
    data_train_sampled_list = []
    labels_train_sampled_list = []

    for i in np.unique(kmeans.labels_):
        data_cluster = data_train[kmeans.labels_ == i]
        labels_cluster = labels_train[kmeans.labels_ == i]

        mask = np.random.rand(len(data_cluster)) < sampling_perc

        data_sample = data_cluster[mask]
        labels_sample = labels_cluster[mask]

        if len(data_sample) > 0: #not data_sample.empty:
            data_train_sampled_list.append(data_sample)
        if len(labels_sample) > 0: #not labels_sample.empty:
            labels_train_sampled_list.append(labels_sample)

    # data_train_sampled = pd.concat(data_train_sampled_list, axis=0) if data_train_sampled_list else pd.DataFrame(columns=data_train.columns)
    # labels_train_sampled = pd.concat(labels_train_sampled_list, axis=0) if labels_train_sampled_list else pd.Series(name=labels_train.name)
    if data_train_sampled_list:
        data_train_sampled = np.concatenate(data_train_sampled_list, axis=0)
    else:
        data_train_sampled = np.empty((0, data_train.shape[1]))

    if labels_train_sampled_list:
        labels_train_sampled = np.concatenate(labels_train_sampled_list, axis=0)
    else:
        labels_train_sampled = np.empty((0,), dtype=labels_train.dtype)

    return data_train_sampled, labels_train_sampled

class DMTRunner:
    def __init__(self, params: dict, dataset_name: str, batch_size: Optional[int] = None, sampling: Optional[float] = 0, delay: Optional[int] = 0):
        # sampling 0 -> no online training; sampling 1 -> full test set is used for training
        # delay: number of batches to wait before training on test set
        self.model = DynamicModelTree(**params)
        self.stream = None
        self.batch_size = batch_size
        self.sampling = sampling
        self.delay = delay
        self.dataset_name = dataset_name
        self.dir_path = None

    def calculate_batch_size(self, csv_path: str) -> int:
        # in DMT, the batch size is always 0.1% of the whole dataset
        with open(csv_path, "r") as f:
            num_samples = sum(1 for _ in f) - 1

        return num_samples // 1000

    def load_stream(self, csv_path: str, stream_type: str = "train") -> FileStream:
        if stream_type not in ["train", "test"]:
            raise ValueError(f"Unsupported stream type: {stream_type}")

        with open(csv_path, "r") as f:
            label = f.readline().strip().split(",")[-1].lower()
            if label != "label":
                raise ValueError(f"Unsupported label: {label}. The target data must be stored in a \"Label\" column, positioned at the end of the dataset")

        if stream_type == "train":
            if self.batch_size is None:
                self.batch_size = self.calculate_batch_size(csv_path)
            print(f"Batch size chosen: {self.batch_size}")
            self.dir_path = f'./results/DMT/B{self.batch_size}/{self.sampling}/{self.delay}/{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(self.dir_path, exist_ok=True)

            stream = FileStream(csv_path, target_idx=-1)
            self.train_stream = stream
        else:
            stream = FileStream(csv_path, target_idx=-1)
            self.test_stream = stream

        return stream, self.batch_size

    def train(self, verbose: bool = True):
        print("Start training...")

        iteration = 0
        while self.train_stream.has_more_samples():
            x, y = self.train_stream.next_sample(batch_size=self.batch_size)
            self.model.partial_fit(x, y)
            if verbose:
                if iteration != 0 and iteration % 50 == 0:
                    print(f"Reached iteration {iteration}")
            iteration += 1

        self.train_stream.restart()
        print("Training finished. Saving the model...")
        self.save_model()
        print("Model saved.")

    def test(self, verbose: bool = True) -> pd.DataFrame:
        preds = np.array([], dtype=np.int32)
        targets = np.array([], dtype=np.int32)
        iteration = 0
        print("Start testing...")
        feature_batches, target_batches = [], []
        while self.test_stream.has_more_samples():
            x, y = self.test_stream.next_sample(batch_size=self.batch_size)
            y_pred = self.model.predict(x)

            preds = np.append(preds, y_pred)
            targets = np.append(targets, y.astype(np.int32))

            feature_batches.append(x)
            feature_batches = feature_batches[-self.delay-1:] #code to prevent testing to completely fill the CPU. The -1 is added just to be sure not to remove meaningful batches
            target_batches.append(y)
            target_batches = target_batches[-self.delay-1:]

            num_batch_to_train = iteration - self.delay
            if num_batch_to_train >= 0:
                if self.sampling > 0:
                    # print("Update tree...")
                    kmeans = find_best_kmeans(feature_batches[-self.delay])
                    data_train, labels_train = find_data_sampled(feature_batches[-self.delay], target_batches[-self.delay], kmeans, args.sampling)
                    self.model.partial_fit(data_train, labels_train)

            if verbose:
                if iteration != 0 and iteration % 50 == 0:
                    print(f"Reached iteration {iteration}")
            iteration += 1

        print("Testing finished. Saving the model...")
        self.save_model(name="tree_test.pkl")
        print("Model saved.")

        df = pd.DataFrame({"Label": targets, "Predicted": preds})

        return df

    def save_model(self, name="tree_train.pkl"):
        with open(f'{self.dir_path}/{name}', 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    args = get_args()
    print("Preprocessing dataset...")
    train_path = "DMT/preprocessed_datasets/train.csv"
    test_path = "DMT/preprocessed_datasets/test.csv"
    df_train, df_test = get_dataset(args)
    df_train, df_test, metadata = preprocess(df_train, df_test) #metadata may contain categorical features, which should be added in params
    print("Preprocessing finished. Saving datasets to disk...")
    # we save the processed datasets, later we delete them
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print("Saving finished.")

    params = {
        "n_classes": num_classes[args.dataset],
        "cat_features": None, # TODO: detect categorical features and put them here
    }

    dm = DMTRunner(params, args.dataset, args.batch_size, args.sampling, args.delay)
    dm.load_stream(train_path)
    dm.load_stream(test_path, stream_type="test")

    total_time = time.time()
    dm.train()
    test_time = time.time()
    preds = dm.test()
    test_time = time.time() - test_time

    metrics = get_metrics(preds, test_time, total_time)
    save_metrics(metrics, dm.dir_path)

    # we remove the previously processed datasets
    os.remove(train_path)
    os.remove(test_path)
