import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, MinMaxScaler

def replace_inf(x):
    x = np.where(np.isinf(x), np.finfo(np.float64).max, x)
    x = np.where(np.isneginf(x), np.finfo(np.float64).min, x)
    return x

def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, dict):
    ordinal_categories = [i for i, col in enumerate(df_train.columns) if
                          df_train[col].dtype == 'object' or df_train[col].dtype == 'O']
    print('Categorical Categories: ', len(ordinal_categories))

    numerical_categories = [i for i, col in enumerate(df_train.columns) if i not in ordinal_categories]
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
    metadata = {"ordinal_categories": ordinal_categories, "numerical_categories": numerical_categories}

    # we return also metadata, as it could be useful for further preprocessing
    return df_train, df_test, metadata