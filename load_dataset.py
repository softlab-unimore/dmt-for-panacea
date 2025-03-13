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
    df_train = df_train.drop(columns=['id', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp'])
    df_test = df_test.drop(columns=['id', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp'])
    return df_train.iloc[:, :-1], df_test.iloc[:, :-1]

def load_csv(dataset, test_file='TestData.csv'):
    df_train = pd.read_csv(f'./datasets/{dataset}/TrainData.csv', delimiter=',')
    df_test = pd.read_csv(f'./datasets/{dataset}/{test_file}', delimiter=',')
    return df_train, df_test

def load_cicids_2017(dataset):
    df = pd.read_csv(f'./datasets/{dataset}/clean_data.csv', delimiter=',')
    return df[:693702], df[693702:]