import KitNET.KitNET as kit
import numpy as np
import pandas as pd
import time
import os
import pickle

from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from util import get_metrics, save_metrics
from data_utils.load_dataset import get_dataset
from data_utils.preprocessing import preprocess

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
    args.add_argument('--thr_mode', type=str, default='max')
    args.add_argument('--percentile', type=float, default=0.95)

    args = args.parse_args()

    return args

class KitNETRunner:
    def __init__(self, args: dict):
        self.maxAE = args["maxAE"]
        self.FMgrace = args["FMgrace"]
        self.ADgrace = args["ADgrace"]
        self.dataset_name = args["dataset"]
        self.batch_size = 1
        self.dir_path = f'./results/B{self.batch_size}/Kitsune/{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.beta = args["beta"]

    def build(self, data: np.array):
        self.model = kit.KitNET(data.shape[1]-1, self.maxAE, None if self.FMgrace==0 else self.FMgrace, self.ADgrace)

    def run(self, data: np.array) -> (pd.DataFrame, float):
        if self.model.n != data.shape[1]-1:
            raise ValueError(f"Using a different dataset than the one used to build() KitNETRunner. Please provide the same dataset.")

        RMSEs = np.zeros(data.shape[0])
        testing_phase = False
        test_time = 0

        for i in tqdm(range(data.shape[0])):
            if i < self.FMgrace+self.ADgrace:
                if int(data[i,-1]) != 0:
                    self.model.n_trained += 1
                    continue
            elif not testing_phase:
                testing_phase = True
                test_time = time.time()
                self.save_model()
            RMSEs[i] = self.model.process(data[i,:-1])
            # print(RMSEs[i])

        results = pd.DataFrame([], columns=["Predicted", "Label"])
        results["Predicted"] = [el for el in list(RMSEs)]
        results["Label"] = [int(el) for el in list(data[:,-1])]

        return results, test_time

    def get_predictions(self, rmses: list, thr_mode: str = "max", percentile: int = 0.95) -> (list, float):
        if thr_mode == "max":
            max_train_rmse = max(rmses[:self.FMgrace + self.ADgrace])
        elif thr_mode == "percentile":
            if percentile < 0 or percentile > 1:
                raise ValueError("percentile must be between 0 and 1")
            max_train_rmse = sorted(rmses[:self.FMgrace + self.ADgrace])[int((self.FMgrace + self.ADgrace-1)*percentile)]
        else:
            raise ValueError(f"Unknown thr_mode: {thr_mode}")

        test_rmses = rmses[self.FMgrace+self.ADgrace:]
        predictions = []

        for rmse in test_rmses:
            if rmse >= max_train_rmse*self.beta:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions, max_train_rmse

    def save_model(self, name="model.pkl"):
        os.makedirs(self.dir_path, exist_ok=True)
        with open(f'{self.dir_path}/{name}', 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    args = get_args()
    print("Preprocessing dataset...")
    df_train, df_test = get_dataset(args)
    df_train, df_test, metadata = preprocess(df_train, df_test) #metadata may contain categorical features, which should be added in params
    print("Preprocessing finished.")
    args = vars(args)

    params = {
        "n_classes": num_classes[args["dataset"]],
        "cat_features": None, # TODO: detect categorical features and put them here
    }

    # adding default values of KitNET

    default_values = {
        "maxAE": 10,
        "FMgrace": len(df_train)//10,
        "ADgrace": len(df_train)-(len(df_train)//10),
        "beta": 1
    }

    args |= default_values

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True).values #[:int((len(df_train)//10)*1.2)] #.values

    ktr = KitNETRunner(args)

    print("Running KitNET 🚀")
    start = time.time()
    ktr.build(df)
    rmses, test_time = ktr.run(df)
    print(f"🏁 KitNET run completed.")

    predictions, max_phi = ktr.get_predictions(rmses["Predicted"].tolist(), thr_mode=args["thr_mode"], percentile=args["percentile"])
    final_results = pd.DataFrame({
        "Predicted": predictions,
        "Label": rmses["Label"].tolist()[args["FMgrace"] + args["ADgrace"]:],
        "Raw scores": rmses["Predicted"].tolist()[args["FMgrace"] + args["ADgrace"]:]
    })

    final_results.to_csv(os.path.join(ktr.dir_path, "result.csv"), index=False)
    metrics = get_metrics(final_results, time.time()-test_time, start)
    metrics |= {"threshold": max_phi*ktr.beta}
    os.makedirs(ktr.dir_path, exist_ok=True)
    save_metrics(metrics, ktr.dir_path)