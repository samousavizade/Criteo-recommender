from fileinput import filename
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from flask import Flask, request
import requests
import os

MODIFIED_CSV = 'modified_dataset.csv'
DEV = 'dev'
PROD = 'prod'
DATA_CSV = 'data.csv'
QUERY_CSV = 'query.csv'


def get_numerical_categorical_columns(data):
    numerical_columns_df = data.select_dtypes(include=np.number)
    numerical_column_names = numerical_columns_df.columns.tolist()

    column_types_df = pd.DataFrame(data.columns, columns=['column name'])

    column_types_df['numerical or categorical'] = np.where(column_types_df['column name'].isin(numerical_column_names),
                                                           'numerical', 'categorical')

    num_columns = column_types_df[column_types_df['numerical or categorical']
                                  == 'numerical']['column name']
    cat_columns = column_types_df[column_types_df['numerical or categorical']
                                  == 'categorical']['column name']

    return num_columns, cat_columns


def pre_process(data_f, phase: str):
    df = pd.read_csv(data_f)
    query_size = len(df)
    target_column_name = 'Sale'
    target = None
    threshold = 6
    prev_data = None
    to_be_dropped_cols = [
        'product_category(7)',
        'SalesAmountInEuro',
        'time_delay_for_conversion',
        'audience_id',
        'click_timestamp',
        'product_title',
        'user_id',
        'product_id',
    ]
    numerical_columns, categorical_columns = get_numerical_categorical_columns(df)
    df[numerical_columns] = df[numerical_columns].apply(
        lambda col: col.replace({-1: np.nan}))

    df['product_price'] = df['product_price'].replace({0: np.nan})

    df[categorical_columns] = df[categorical_columns].apply(
        lambda col: col.replace({'-1': np.nan}))

    if phase == DEV:
        target = df[target_column_name]
        to_be_dropped_cols.append(target_column_name)
    df.drop(to_be_dropped_cols, axis=1, inplace=True)

    if phase == DEV:
        df = df.dropna(thresh=threshold)
    if phase == PROD:
        prev_data = pd.read_csv(DATA_CSV)
        prev_data.drop([target_column_name, ], axis=1, inplace=True)
        numerical_columns, categorical_columns = get_numerical_categorical_columns(prev_data)
        prev_data[numerical_columns] = prev_data[numerical_columns].apply(
            lambda col: col.replace({-1: np.nan}))

        prev_data['product_price'] = prev_data['product_price'].replace({0: np.nan})

        prev_data[categorical_columns] = prev_data[categorical_columns].apply(lambda col: col.replace({'-1': np.nan}))
        prev_data = prev_data.dropna(thresh=threshold)
        prev_data.drop(to_be_dropped_cols, axis=1, inplace=True)
        prev_data = prev_data.dropna(thresh=threshold)

    if phase == PROD:
        assert prev_data is not None
        final_df = pd.concat([prev_data, df])
    else:
        final_df = df

    numerical_columns, categorical_columns = get_numerical_categorical_columns(final_df)
    # final_df.iloc[:len(final_df) - query_size, :]
    for column in numerical_columns:
        if phase == PROD:
            median_val = prev_data.loc[~prev_data[column].isnull(), column].median()
        else:
            median_val = final_df.loc[~final_df[column].isnull(), column].median()
        final_df.loc[final_df[column].isnull(), column] = median_val

    for column in categorical_columns:
        if phase == PROD:
            mode_val = prev_data.loc[~prev_data[column].isnull(), column].mode().iat[0]
        else:
            mode_val = final_df.loc[~final_df[column].isnull(), column].mode().iat[0]
        final_df.loc[final_df[column].isnull(), column] = mode_val

    # Categorical Columns Encode
    for c_column in categorical_columns:
        le = LabelEncoder()
        encoded: pd.Series = le.fit_transform(final_df.loc[~final_df[c_column].isnull(), c_column])
        final_df.loc[~final_df[c_column].isnull(), c_column] = encoded

    # normalizer = StandardScaler()
    # normalizer = MinMaxScaler()
    # df.loc[:, numerical_columns] = normalizer.fit_transform(df[numerical_columns])

    if phase == DEV:
        assert target_column_name is not None and target is not None
        final_df[target_column_name] = target
    final_df = final_df[-query_size:]
    cols = final_df.columns.tolist()
    if phase == DEV:
        cols = cols[-1:] + cols[:-1]
    data_frame = final_df[cols]
    header_for_df = False
    if phase == PROD:
        data_frame.replace(np.nan, '0', inplace=True)
        header_for_df = True

    # print(f'numerical_columns len: {len(numerical_columns)}')
    # print(f'numerical_columns: {numerical_columns.tolist()}')
    #
    # print(f'categorical_columns len: {len(categorical_columns)}')
    # print(f'categorical_columns: {categorical_columns.tolist()}')
    #
    # print('dataset dimensions: {data_frame.shape[0]}')
    pd.DataFrame.to_csv(data_frame,
                        MODIFIED_CSV,
                        sep='\t',
                        index=False,
                        header=header_for_df
                        )


def send(phase: str):
    host = 'mlflow'
    port = 8080
    url = f'http://{host}:{port}/ml?phase={phase}'
    csv_f = open(MODIFIED_CSV, 'rb')
    r = requests.post(url=url, files={'data_file': csv_f})
    return r.text


api = Flask(__name__)


@api.route('/analyze', methods=['POST'])
def analyze():
    phase = request.args.get('phase')
    if phase == DEV:
        file_name = DATA_CSV
    elif phase == PROD:
        assert os.path.exists(DATA_CSV)
        file_name = QUERY_CSV
    else:
        return "Not a valid phase"
    data_f = request.files.get('data_file')
    data_f.save(file_name)
    pre_process(file_name, phase)
    res = send(phase)
    return res


if __name__ == '__main__':
    api.run('0.0.0.0', 8050)
