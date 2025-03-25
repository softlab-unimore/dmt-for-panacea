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

def load_csv(dataset, test_file='TestData.csv'):
    df_train = pd.read_csv(f'./datasets/{dataset}/TrainData.csv', delimiter=',')
    df_test = pd.read_csv(f'./datasets/{dataset}/{test_file}', delimiter=',')
    return df_train, df_test

def load_cicids_2017(dataset):
    df = pd.read_csv(f'./datasets/{dataset}/clean_data.csv', delimiter=',')
    return df[:693702], df[693702:]

def get_dataset(args) -> (pd.DataFrame, pd.DataFrame):
    if args.dataset == 'CICIDS2017_improved':
        df_train, df_test = load_cicids_2017_improved(args.dataset)
    elif args.dataset == 'CICIDS2017':
        df_train, df_test = load_cicids_2017(args.dataset)
    elif args.dataset == 'IDS2018':
        df_train, df_test = load_csv(args.dataset, test_file='NewTestData.csv')
    elif args.dataset == 'Kitsune':
        df_train, df_test = load_csv(args.dataset)
    elif args.dataset == 'mKitsune':
        df_train, df_test = load_csv(dataset='Kitsune', test_file='NewTestData.csv')
    elif args.dataset == 'rKitsune':
        df_train, df_test = load_csv(dataset='Kitsune', test_file='Recurring.csv')
    elif args.dataset == 'CICIDS2017_prova':
        df = pd.read_csv(f'datasets/{args.dataset}.csv', delimiter=',')
        df = process_timestamps(df)
        train_end = int(len(df) * 0.2)
        df_train, df_test = df.iloc[:train_end], df.iloc[train_end:]
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return df_train, df_test