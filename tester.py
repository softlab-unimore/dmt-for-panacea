from skmultiflow.data import FileStream
from dmt.DMT import DynamicModelTree
from sklearn.metrics import accuracy_score

from main import get_args, get_dataset, get_metrics, save_metrics
from load_dataset import preprocess
from datetime import datetime

import time


class DMTRunner:
    def __init__(self, params: dict, dataset_name: str):
        self.model = DynamicModelTree(**params)
        self.stream = None
        self.batch_size = None
        self.dataset_name = dataset_name
        self.dir_path = None

    def calculate_batch_size(self, csv_path: str) -> int:
        # in DMT, the batch size is always 0.1% of the whole dataset
        with open(filename, "r") as f:
            num_samples = sum(1 for _ in f) - 1

        return num_samples // 1000

    def load_stream(self, csv_path: str, stream_type: str = "train") -> FileStream:
        if stream_type not in ["train", "test"]:
            raise ValueError(f"Unsupported stream type: {stream_type}")

        if stream_type == "train":
            self.batch_size = calculate_batch_size(csv_path)
            self.dir_path = f'../results/DMT/B{self.batch_size}/{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            stream = FileStream(csv_path, target_idx=-1)
            self.train_stream = stream
        else:
            stream = FileStream(csv_path, target_idx=-1)
            self.test_stream = stream

        return stream, self.batch_size

    def train(self):
        while self.train_stream.has_more_samples():
            x, y = self.train_stream.next_sample(batch_size=self.batch_size)
            self.model.partial_fit(x, y)

        self.train_stream.restart()

    def test(self) -> list:
        preds = []
        while self.test_stream.has_more_samples():
            x, y = self.test_stream.next_sample(batch_size=self.batch_size)

            y_pred = self.model.predict(x)
            preds.append(y_pred)
            model.partial_fit(x, y) #DMT trains online

        return preds

if __name__ == "__main__":
    args = get_args()
    df_train, df_test = get_dataset(args)
    df_train, df_test, metadata = preprocess(df_train, df_test) #metadata may contain categorical features, which should be added in params

    # we save the processed datasets, later we delete them
    train_path = "dmt/preprocessed_datasets/train.csv"
    test_path = "dmt/preprocessed_datasets/test.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    params = {
        "n_classes": 2,
        "cat_features": None,
    }
    dm = DMTRunner(params, args.dataset)
    dm.load_stream(train_path)
    dm.load_stream(test_path, stream_type="test")

    total_time = time.time()
    dm.train()
    test_time = time.time()
    preds = dm.test()
    test_time = time.time() - test_time

    metrics = get_metrics(preds, test_time, total_time)
    save_metrics(metrics, dm.dir_path)

