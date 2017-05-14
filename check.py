import csv

with open('confused_data/Raw/0.csv', 'r') as f:
    reader = csv.reader(f)
    print('0.csv', len(next(reader)))
    print(next(reader))

with open('confused_data/Raw/1.csv', 'r') as f:
    reader = csv.reader(f)
    print('1.csv', len(next(reader)))
