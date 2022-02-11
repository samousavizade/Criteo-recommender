

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

path = '../train_dataset.csv'
df = pd.read_csv(path)
df.head()

def get_numerical_categorical_columns(data):
    numerical_columns_df = data.select_dtypes(include=np.number)
    numerical_column_names = numerical_columns_df.columns.tolist()

    column_types_df = pd.DataFrame(data.columns, columns=['column name'])

    column_types_df['numerical or categorical'] = np.where(column_types_df['column name'].isin(numerical_column_names),
                                                           'numerical', 'categorical')

    num_columns = column_types_df[column_types_df['numerical or categorical'] == 'numerical']['column name']
    cat_columns = column_types_df[column_types_df['numerical or categorical'] == 'categorical']['column name']

    return num_columns, cat_columns


numerical_columns, categorical_columns = get_numerical_categorical_columns(df)

df[numerical_columns] = df[numerical_columns].apply(lambda col: col.replace({-1: np.nan}))

df['product_price'] = df['product_price'].replace({0: np.nan})

df[categorical_columns] = df[categorical_columns].apply(lambda col: col.replace({'-1': np.nan}))

target_column_name = 'Sale'
target = df[target_column_name]

df.drop(
    [
        target_column_name,
        'product_category(7)',
        'SalesAmountInEuro',
        'time_delay_for_conversion',
        'audience_id',
        'click_timestamp',
        'product_title',
        'user_id',
        'product_id',
    ],
    axis=1, inplace=True
)

threshold = 6
df = df.dropna(thresh=threshold)

numerical_columns, categorical_columns = get_numerical_categorical_columns(df)

for column in numerical_columns:
    df.loc[df[column].isnull(), column] = df.loc[~df[column].isnull(), column].median()

for column in categorical_columns:
    df.loc[df[column].isnull(), column] = df.loc[~df[column].isnull(), column].mode().iat[0]

# Categorical Columns Encode
for c_column in categorical_columns:
    le = LabelEncoder()
    encoded: pd.Series = le.fit_transform(df.loc[~df[c_column].isnull(), c_column])
    df.loc[~df[c_column].isnull(), c_column] = encoded


normalizer = StandardScaler()
# normalizer = MinMaxScaler()
df.loc[:, numerical_columns] = normalizer.fit_transform(df[numerical_columns])


df[target_column_name] = target
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_frame = df[cols]

print(f'numerical_columns len: {len(numerical_columns)}')
print(f'numerical_columns: {numerical_columns.tolist()}')

print(f'categorical_columns len: {len(categorical_columns)}')
print(f'categorical_columns: {categorical_columns.tolist()}')

print('dataset dimensions: {data_frame.shape[0]}')
pd.DataFrame.to_csv(data_frame,
                    'modified_dataset.csv',
                    sep='\t',
                    index=False,
                    header=False
                    )
