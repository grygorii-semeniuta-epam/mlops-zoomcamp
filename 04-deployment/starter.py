#!/usr/bin/env python
# coding: utf-8


# get_ipython().system('pip freeze | grep scikit-learn')
# get_ipython().system('python -V')



import pickle
import pandas as pd
import os
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    std_duration = y_pred.std()
    print(f'Standard deviation of predictions: {std_duration}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    output_file = 'output.parquet'

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / 1024 / 1024
    print(f'Output file size: {file_size_mb:.2f} MB')

    mean_duration = y_pred.mean()
    print(f'Mean duration of predictions: {mean_duration:.2f} minutes')

if __name__ == '__main__':
    run()
