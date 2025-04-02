import os
import pickle
import time
from datetime import datetime

from argparse import ArgumentParser
from UnsupervisedDMT.decision_tree import DecisionTree
import warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import SettingWithCopyWarning
from data_utils.load_dataset import get_dataset

from data_utils.preprocessing import preprocess
from util import get_metrics, save_metrics

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='CICIDS2017')
    args.add_argument('--batch_size', type=int, default=1024)
    args.add_argument('--mode', type=str, default='')
    args.add_argument('--max_depth', type=int, default=20)
    args.add_argument('--dist_threshold', type=float, default=0.25)
    args.add_argument('--min_points_per_leaf', type=int, default=20)
    args.add_argument('--closest_k_points', type=float, default=0.1)
    args.add_argument('--number_thresholds', type=int, default=2)
    args = args.parse_args()

    return args

if __name__=='__main__':
    args = get_args()
    df_train, df_test = get_dataset(args)
    df_train, df_test, cat = preprocess(df_train, df_test)
    dir_path = f'results/B{args.batch_size}/{args.mode}/{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(dir_path, exist_ok=True)

    tree = DecisionTree(max_depth=args.max_depth, min_points_per_leaf=args.min_points_per_leaf, dist_threshold=args.dist_threshold, closest_k_points=0.1, closer_DBSCAN_point=0.1, eps_DBSCAN=0.1, number_thresholds=2, ordinal_categories=cat['ordinal_categories'])

    total_time = time.time()

    print('------------------------------------')
    print('Start Train')
    print('------------------------------------')

    if df_train.shape[0] % args.batch_size != 0:
        df_train = df_train.iloc[:-(df_train.shape[0] % args.batch_size)]

    print(f'Batch: 0/{df_train.shape[0]}')
    root = tree.partial_fit(None, df_train.iloc[:, :-1][:args.batch_size], df_train['Label'][:args.batch_size])

    for i in range(args.batch_size, df_train.shape[0], args.batch_size):
        print(f'Batch: {i}/{df_train.shape[0]}')
        data_train, labels_train = df_train.iloc[:, :-1][i:i+args.batch_size], df_train['Label'][i:i+args.batch_size]
        root = tree.partial_fit(root, data_train, labels_train)

    print('Saving tree...')
    with open(f'{dir_path}/tree.pkl', 'wb') as f:
        pickle.dump(root, f)

    print('------------------------------------')
    print('Start Test')
    print('------------------------------------')

    data_test, labels_test = df_test.iloc[:, :-1].copy(deep=True), df_test.loc[:, 'Label'].copy(deep=True)

    test_time = time.time()
    # get_anomalies = tree.detect_anomalies(root, data_test)
    results = tree.detect_anomalies_with_mad(root, data_test, labels_test)
    results.to_csv(f'{dir_path}/results_{args.dataset}.csv', index=False)
    test_time = time.time() - test_time

    metrics = get_metrics(results, test_time, total_time)
    save_metrics(metrics, dir_path)

    print('Finish')