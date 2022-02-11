import pandas as pd
import requests
import argparse

host = '127.0.0.1'

port = 8080

url = f'http://{host}:{port}/invocations'

headers = {
    'Content-Type': 'application/json',
}

# test_data is a Pandas dataframe with data for testing the ML model
test_data = pd.read_csv('query_test.csv', sep='\t')
http_data = test_data.to_json(orient='split')

r = requests.post(url=url, headers=headers, data=http_data)

print(f'Predictions: {r.text}')
