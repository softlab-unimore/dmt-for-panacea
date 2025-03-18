import json
import os
import pickle
import time
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import load_dataset
from decision_tree import DecisionTree
import warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def replace_inf(x):
    x = np.where(np.isinf(x), np.finfo(np.float64).max, x)
    x = np.where(np.isneginf(x), np.finfo(np.float64).min, x)
    return x

if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='CICIDS2017')
    args.add_argument('--batch_size', type=int, default=10000)
    args.add_argument('--max_depth', type=int, default=20)
    args.add_argument('--min_points_per_leaf', type=int, default=20)
    args.add_argument('--closest_k_points', type=float, default=0.1)
    args.add_argument('--closer_DBSCAN_point', type=float, default=0.1)
    args.add_argument('--eps_DBSCAN', type=float, default=0.1)
    args.add_argument('--number_thresholds', type=int, default=2)
    args = args.parse_args()

    if args.dataset == 'CICIDS2017_improved':
        df_train, df_test = load_dataset.load_cicids_2017_improved(args.dataset)
    elif args.dataset == 'CICIDS2017':
        df_train, df_test = load_dataset.load_cicids_2017(args.dataset)
    elif args.dataset == 'IDS2018':
        df_train, df_test = load_dataset.load_csv(args.dataset, test_file='NewTestData.csv')
    elif args.dataset == 'Kitsune':
        df_train, df_test = load_dataset.load_csv(args.dataset)
    elif args.dataset == 'mKitsune':
        df_train, df_test = load_dataset.load_csv(dataset='Kitsune', test_file='NewTestData.csv')
    elif args.dataset == 'rKitsune':
        df_train, df_test = load_dataset.load_csv(dataset='Kitsune', test_file='Recurring.csv')
    elif args.dataset == 'CICIDS2017_prova':
        df = pd.read_csv(f'datasets/{args.dataset}.csv', delimiter=',')
        df = load_dataset.process_timestamps(df)
        train_end = int(len(df) * 0.2)
        df_train, df_test = df.iloc[:train_end], df.iloc[train_end:]
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    ordinal_categories = [i  for i, col in enumerate(df_train.columns) if df_train[col].dtype == 'object' or df_train[col].dtype == 'O']
    print('Categorical Categories: ', len(ordinal_categories))

    numerical_categories = [i  for i, col in enumerate(df_train.columns) if i not in ordinal_categories]
    print('Ordinal Categories: ', len(numerical_categories))

    pipeline = ColumnTransformer([
        ('numerical', Pipeline([
            ('replace_inf', FunctionTransformer(replace_inf)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), numerical_categories),
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int), ordinal_categories)
    ], remainder='passthrough')

    df_train = pd.DataFrame(pipeline.fit_transform(df_train), columns=df_train.columns)
    df_test = pd.DataFrame(pipeline.transform(df_test), columns=df_test.columns)
    dir_path = f'results/B{args.batch_size}/{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(dir_path, exist_ok=True)

    tree = DecisionTree(max_depth=args.max_depth, min_points_per_leaf=args.min_points_per_leaf, closest_k_points=0.1, closer_DBSCAN_point=0.1, eps_DBSCAN=0.1, number_thresholds=2, ordinal_categories=ordinal_categories)

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
    with open(f'{dir_path}/tree.pkl', 'w') as f:
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

    metrics = {
        'f1_score_macro': f1_score(results['Label'], results['Predicted'], average='macro'),
        'f1_score_micro': f1_score(results['Label'], results['Predicted'], average='micro'),
        'acc': accuracy_score(results['Label'], results['Predicted']),
        'precision': precision_score(results['Label'], results['Predicted'], average='macro'),
        'recall': recall_score(results['Label'], results['Predicted'], average='macro'),
        'test_time': test_time,
        'total_time': time.time() - total_time
    }

    print('------------------------------------')
    print('F1 Score Macro: ', metrics['f1_score_macro'])
    print('F1 Score Micro: ', metrics['f1_score_micro'])
    print('Accuracy: ', metrics['acc'])
    print('Precision: ', metrics['precision'])
    print('Recall: ', metrics['recall'])
    print('Test Time: ', test_time)
    print('Total Time: ', time.time() - total_time)
    print('------------------------------------')

    with open(f'{dir_path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print('Finish')