import json
import os
import time
from argparse import ArgumentParser

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from decision_tree import DecisionTree
import warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def process_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')

    date_cols = {
        'year': df['Timestamp'].dt.year,
        'month': df['Timestamp'].dt.month,
        'day': df['Timestamp'].dt.day,
        'hour': df['Timestamp'].dt.hour,
        'minute': df['Timestamp'].dt.minute,
        'weekday': df['Timestamp'].dt.weekday
    }

    df = pd.concat([pd.DataFrame(date_cols), df.drop(columns=['Timestamp'])], axis=1)
    return df

def replace_inf(x):
    x = np.where(np.isinf(x), np.finfo(np.float64).max, x)
    x = np.where(np.isneginf(x), np.finfo(np.float64).min, x)
    return x

if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='cicids')
    args.add_argument('--n_iterations', type=int, default=5)
    args.add_argument('--n_train', type=int, default=200)
    args.add_argument('--max_depth', type=int, default=200)
    args.add_argument('--min_points_per_leaf', type=int, default=20)
    args.add_argument('--closest_k_points', type=float, default=0.1)
    args.add_argument('--closer_DBSCAN_point', type=float, default=0.1)
    args.add_argument('--eps_DBSCAN', type=float, default=0.1)
    args.add_argument('--number_thresholds', type=int, default=2)
    args = args.parse_args()

    df = pd.read_csv(f'datasets/{args.dataset}.csv', delimiter=',')
    df = process_timestamps(df)

    categorical_categories = [i for i, col in enumerate(df.columns) if df[col].dtype == 'object']
    print('Ordinal Categories: ', categorical_categories)

    ordinal_categories = [i for i, col in enumerate(df.columns) if i not in categorical_categories]
    print('Ordinal Categories: ', ordinal_categories)

    pipeline = ColumnTransformer([
        ('numerical', Pipeline([
            ('replace_inf', FunctionTransformer(replace_inf)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), ordinal_categories),
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int), categorical_categories)
    ], remainder='passthrough')

    df_filter = pd.DataFrame(pipeline.fit_transform(df), columns=df.columns)
    dir_path = f'results/N{args.n_iterations}B{args.n_train}'
    os.makedirs(dir_path, exist_ok=True)

    tree = DecisionTree(max_depth=200, min_points_per_leaf=20, closest_k_points=0.1, closer_DBSCAN_point=0.1, eps_DBSCAN=0.1, number_thresholds=2, ordinal_categories=ordinal_categories)

    n_test = len(df_filter) - (args.n_train * args.n_iterations) - 1

    total_time = time.time()

    print('------------------------------------')
    print('Start Train')
    print('------------------------------------')

    for i in range(args.n_iterations):

        print('Iteration: ', i)

        start_train = i * args.n_train
        end_train = start_train + args.n_train

        data_train, labels_train = df_filter.iloc[:-1, :-1][start_train:end_train], df_filter['Label'][start_train:end_train]

        if i == 0:
            root = tree.partial_fit(None, data_train, labels_train)
        else:
            root = tree.partial_fit(root, data_train, labels_train)

        if i == args.n_iterations - 1:

            print('------------------------------------')
            print('Start Test')
            print('------------------------------------')

            start_test = end_train
            end_test = start_test + n_test
            data_test, labels_test = df_filter.iloc[:-1, :-1][start_test:end_test].copy(deep=True), df_filter['Label'][start_test:end_test].copy(deep=True)

            test_time = time.time()
            # get_anomalies = tree.detect_anomalies(root, data_test)
            results = tree.detect_anomalies_with_mad(root, data_test, labels_test)
            results.to_csv(f'{dir_path}/results_cicids.csv', index=False)
            test_time = time.time() - test_time

            metrics = {
                'f1_score_macro': f1_score(results['Label'], results['Predicted'], average='macro'),
                'f1_score_micro': f1_score(results['Label'], results['Predicted'], average='micro'),
                'acc': accuracy_score(results['Label'], results['Predicted']),
                'precision': precision_score(results['Label'], results['Predicted'], average='macro'),
                'recall': recall_score(results['Label'], results['Predicted'], average='macro'),
                'test_time': test_time
            }

            with open(f'{dir_path}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            print('------------------------------------')
            print('F1 Score Macro: ', metrics['f1_score_macro'])
            print('F1 Score Micro: ', metrics['f1_score_micro'])
            print('Accuracy: ', metrics['acc'])
            print('Precision: ', metrics['precision'])
            print('Recall: ', metrics['recall'])
            print('------------------------------------')
            print('Test Time: ', test_time)

    print('Total Time: ', time.time() - total_time)