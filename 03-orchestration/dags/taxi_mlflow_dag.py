from datetime import datetime, timedelta
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ====== CONFIG ======
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Docker service name from docker-compose
MLFLOW_EXPERIMENT_NAME = "yellow-taxi-experiment"

# fixed dataset URL
MARCH_2023_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "yellow_taxi_lr_pipeline",
    default_args=default_args,
    description="Yellow Taxi pipeline: load -> clean -> train/log -> cleanup",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
)

# ====== TASK FUNCTIONS ======

def task_1_load_data(**context):
    df = pd.read_parquet(MARCH_2023_URL)
    print(f"[TASK 1] Loaded {len(df)} records.")
    df.to_parquet("/tmp/raw_data.parquet")


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def task_2_prepare_data(**context):
    df_clean = read_dataframe("/tmp/raw_data.parquet")
    print(f"[TASK 2] After cleaning: {len(df_clean)} records.")
    df_clean.to_parquet("/tmp/clean_data.parquet")


def task_3_train_and_log(**context):
    df = pd.read_parquet("/tmp/clean_data.parquet")

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    y = df["duration"].values

    lr = LinearRegression()
    lr.fit(X, y)

    print(f"[TASK 3] Model intercept: {lr.intercept_}")

    rmse = root_mean_squared_error(y, lr.predict(X))
    print(f"[TASK 3] RMSE on training data: {rmse}")

    # ==== MLflow logging ====
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID,DOLocationID,trip_distance")
        mlflow.log_metric("rmse", rmse)

        Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr, artifact_path="model")


# ====== DAG TASKS ======

load_task = PythonOperator(
    task_id="load_data",
    python_callable=task_1_load_data,
    dag=dag,
)

prepare_task = PythonOperator(
    task_id="prepare_data",
    python_callable=task_2_prepare_data,
    dag=dag,
)

train_log_task = PythonOperator(
    task_id="train_and_log",
    python_callable=task_3_train_and_log,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="rm -f /tmp/raw_data.parquet /tmp/clean_data.parquet && rm -rf models",
    dag=dag,
)

# ====== TASK FLOW ======
cleanup_task >> load_task >> prepare_task >> train_log_task
