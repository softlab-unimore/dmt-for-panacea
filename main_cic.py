import pandas as pd
from DecisionTree import DecisionTree
import warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# df = pd.read_csv('CICDATASET/Train_data.csv', delimiter=',')
df = pd.read_csv('CICDATASET/02-14-2018.csv', delimiter=',')
# ordinal_categories = ['protocol_type', 'service', 'flag']
ordinal_categories = [col for col in df.columns if df[col].dtype == 'object']
print('Ordinal Categories: ', ordinal_categories)

# df_filter = df.drop(['protocol_type','service','flag'], axis=1)
# Create and fit encoder
df_filter = df.copy(deep=True)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(df_filter[ordinal_categories])
encoded_values = encoder.transform(df_filter[ordinal_categories])
df_filter[ordinal_categories] = encoded_values
original_values = encoder.inverse_transform(df_filter[ordinal_categories])
# df_filter['class'] = df_filter['class'].map({'normal': 0, 'anomaly': 1})
# df_filter['Label'] = df_filter['Label'].map({'Benign': 0}).fillna(1)
print(len(df_filter))

# TODO: FARE CATEGORICI
# tree = DecisionTree(max_depth=100, min_points_per_leaf=30, closest_k_points=0.2, closer_DBSCAN_point=0.8, eps_DBSCAN=0.5)
tree = DecisionTree(max_depth=200, min_points_per_leaf=20, closest_k_points=0.1, closer_DBSCAN_point=0.1, eps_DBSCAN=0.1, number_thresholds=2, ordinal_categories=ordinal_categories)
n_iterations = 5
n_train = 200
n_test = len(df_filter) - (n_train*n_iterations) - 1

for i in range(n_iterations):
    print('Iteration: ', i)

    start_train = i * n_train
    end_train = start_train + n_train

    # data_train, labels_train = df_filter.iloc[:-1,:-1][start_train:end_train], df_filter['class'][start_train:end_train]
    data_train, labels_train = df_filter.iloc[:-1,:-1][start_train:end_train], df_filter['Label'][start_train:end_train]


    if i == 0:
        root = tree.partial_fit(None, data_train, labels_train)
    else:
        root = tree.partial_fit(root, data_train, labels_train)

    if i == n_iterations - 1:
        start_test = end_train
        end_test = start_test + n_test
        data_test, labels_test = df_filter.iloc[:-1, :-1][start_test:end_test].copy(deep=True), df_filter['Label'][start_test:end_test].copy(deep=True)

        get_anomalies = tree.detect_anomalies(root, data_test)
        print(f1_score(labels_test, get_anomalies.sort_index()['Label'], average='macro'))
        differences = sum(1 if x != y else 0 for x, y in zip(labels_test, get_anomalies.sort_index()['Label']))
        print(differences, len(labels_test))

   # print(get_anomalies)

# Create the figure and first axis
# print(f1_score(labels10, get_anomalies.sort_index()['Label'], average='macro'))