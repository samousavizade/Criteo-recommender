import requests
import sys

host = '127.0.0.1'

port = 8050

file_name, phase = str(sys.argv[1]), str(sys.argv[2])
url = f'http://{host}:{port}/analyze?phase={phase}'
csv_f = open(file_name, 'rb')
r = requests.post(url=url, files={'data_file': csv_f})
print(r.text)
