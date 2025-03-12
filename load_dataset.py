import pandas as pd

def process_timestamps(df, format='%d/%m/%Y %H:%M:%S'):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=format)

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

def load_cicids_2017_improved(dataset):
    train_files = ['monday.csv', 'tuesday.csv']
    test_files = ['wednesday.csv', 'thursday.csv', 'friday.csv']
    df_train = pd.concat([pd.read_csv(f'./datasets/{dataset}/{file}', delimiter=',') for file in train_files], axis=0)
    df_test = pd.concat([pd.read_csv(f'./datasets/{dataset}/{file}', delimiter=',') for file in test_files], axis=0)
    df_train = process_timestamps(df_train, format='%Y-%m-%d %H:%M:%S.%f')
    df_test = process_timestamps(df_test, format='%Y-%m-%d %H:%M:%S.%f')
    return df_train.iloc[:, :-1], df_test.iloc[:, :-1]


def load_cicids_2017(dataset):
    pass