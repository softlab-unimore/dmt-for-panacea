import json
import os
import pickle
import time
from datetime import datetime

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from UnsupervisedDMT.decision_tree import DecisionTree
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from pandas.errors import SettingWithCopyWarning
from data_utils.load_dataset import get_dataset

from data_utils.preprocessing import preprocess
from util import get_metrics, save_metrics, save_parameters

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

np.random.seed(42)

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


def find_data_sampled(data_train, labels_train, kmeans, sampling_perc):
    data_train_sampled_list = []
    labels_train_sampled_list = []

    for i in np.unique(kmeans.labels_):
        data_cluster = data_train[kmeans.labels_ == i]
        labels_cluster = labels_train[kmeans.labels_ == i]

        mask = np.random.choice(len(data_cluster), int(len(data_cluster) * sampling_perc), replace=False)

        data_sample = data_cluster[mask]
        labels_sample = labels_cluster[mask]

        if not data_sample.empty:
            data_train_sampled_list.append(data_sample)
        if not labels_sample.empty:
            labels_train_sampled_list.append(labels_sample)

    data_train_sampled = pd.concat(data_train_sampled_list, axis=0) if data_train_sampled_list else pd.DataFrame(columns=data_train.columns)
    labels_train_sampled = pd.concat(labels_train_sampled_list, axis=0) if labels_train_sampled_list else pd.Series(name=labels_train.name)

    return data_train_sampled, labels_train_sampled


def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='Kitsune')
    args.add_argument('--batch_size', type=int, default=1024)
    args.add_argument('--mode', type=str, default='with_pca')
    args.add_argument('--max_depth', type=int, default=20)
    args.add_argument('--dist_threshold', type=float, default=0.25)
    args.add_argument('--homogeneity_gain_threshold', type=float, default=0)
    args.add_argument('--tmad', type=float, default=3.5)
    args.add_argument('--min_points_per_leaf', type=int, default=20)
    args.add_argument('--number_thresholds', type=int, default=2)
    args.add_argument('--pca', action='store_true', default=False)
    args.add_argument('--delay', type=int, default=10)
    args.add_argument('--sampling', type=float, default=0.1)
    args = args.parse_args()

    return args

if __name__=='__main__':
    args = get_args()
    df_train, df_test = get_dataset(args)
    df_train, df_test, cat = preprocess(df_train, df_test)

    print(f'Dataset shape: {df_train.shape}')

    if args.pca:
        # Apply PCA with 0.95 variance ratio
        pca = PCA(n_components=0.95)
        df_train_pca = pca.fit_transform(df_train.iloc[:, :-1])
        df_test_pca = pca.transform(df_test.iloc[:, :-1])

        df_train = np.concatenate((df_train_pca, df_train.iloc[:, -1].values.reshape(-1, 1)), axis=1)
        df_test = np.concatenate((df_test_pca, df_test.iloc[:, -1].values.reshape(-1, 1)), axis=1)

        df_train = pd.DataFrame(df_train, columns=[f'PC{i+1}' for i in range(df_train_pca.shape[1])] + ['Label'])
        df_test = pd.DataFrame(df_test, columns=[f'PC{i+1}' for i in range(df_train_pca.shape[1])] + ['Label'])

    print(f"PCA shape: {df_train.shape}")

    dir_path = f'results/B{args.batch_size}/{args.mode}/{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(dir_path, exist_ok=True)
    save_parameters(args, dir_path)

    tree = DecisionTree(
        max_depth=args.max_depth,
        min_points_per_leaf=args.min_points_per_leaf,
        dist_threshold=args.dist_threshold,
        homogeneity_gain_threshold=args.homogeneity_gain_threshold,
        number_thresholds=args.number_thresholds,
        ordinal_categories=cat['ordinal_categories']
    )

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

    test_time = time.time()

    if df_test.shape[0] % args.batch_size != 0:
        df_test = df_test.iloc[:-(df_test.shape[0] % args.batch_size)]

    results = pd.DataFrame(columns=['Label', 'Predicted'])
    for i in range(0, df_test.shape[0], args.batch_size):
        print(f'Batch {i} / {df_test.shape[0]}')
        data_test, labels_test = df_test.iloc[:, :-1][i:i + args.batch_size], df_test['Label'][i:i + args.batch_size]
        # batch_results = tree.detect_anomalies_with_mad(root, data_test, labels_test, T_mad=args.tmad)
        # batch_results = tree.detect_anomalies_with_mean(root, data_test, labels_test, T_mad=args.tmad)
        batch_results = tree.detect_anomalies_with_adaptive_clusters(root, data_test, labels_test)
        results = pd.concat([results, batch_results], axis=0)

        if i % 40 == 0:
            batch_time = time.time() - test_time
            metrics = get_metrics(results, batch_time, total_time)

        i = i - args.batch_size * args.delay

        if i >= 0:
            print('Update tree...')
            data_train, labels_train = df_test.iloc[:, :-1][i: i + args.batch_size], df_test['Label'][i: i + args.batch_size]

            if args.sampling > 0:
                kmeans = find_best_kmeans(data_train)
                data_train, labels_train = find_data_sampled(data_train, labels_train, kmeans, args.sampling)

            root = tree.partial_fit(root, data_train, labels_train)

    results.to_csv(f'{dir_path}/results_{args.dataset}_batch.csv', index=False)
    test_time = time.time() - test_time

    metrics = get_metrics(results, test_time, total_time)
    save_metrics(metrics, dir_path, conf=f'tmad_{args.tmad}')

    print('Finish')