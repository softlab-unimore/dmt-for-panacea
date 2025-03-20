from skmultiflow.data import FileStream
from dmt2.DMT import DynamicModelTree
from sklearn.metrics import accuracy_score

from util import get_metrics, save_metrics
from load_dataset import get_dataset
from preprocessing import preprocess
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional

import time
import os
import numpy as np
import pandas as pd
import pickle

num_classes = {
    "CICIDS2017": 2,
    "IDS2018": 2,
    "Kitsune": 2
}

def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='CICIDS2017')
    args.add_argument('--batch_size', type=int, default=None)
    args = args.parse_args()

    return args

class DMTRunner:
    def __init__(self, params: dict, dataset_name: str, batch_size: Optional[int] = None):
        self.model = DynamicModelTree(**params)
        self.stream = None
        self.batch_size = batch_size
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
            self.dir_path = f'./results/DMT/B{self.batch_size}/{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
        while self.test_stream.has_more_samples():
            x, y = self.test_stream.next_sample(batch_size=self.batch_size)
            y_pred = self.model.predict(x)

            preds = np.append(preds, y_pred)
            targets = np.append(targets, y.astype(np.int32))

            self.model.partial_fit(x, y) #DMT trains online

            if verbose:
                if iteration != 0 and iteration % 50 == 0:
                    print(f"Reached iteration {iteration}")
                iteration += 1

        print("Testing finished.")

        df = pd.DataFrame({"Label": targets, "Predicted": preds})

        return df

    def save_model(self):
        with open(f'{self.dir_path}/tree.pkl', 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    args = get_args()
    print("Preprocessing dataset...")
    train_path = "dmt2/preprocessed_datasets/train.csv"
    test_path = "dmt2/preprocessed_datasets/test.csv"
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

    dm = DMTRunner(params, args.dataset, args.batch_size)
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