#!/bin/bash

sudo mkdir -p ./mlflow_data ./dags ./airflow_home/logs ./airflow_home/dags ./airflow_home/plugins ./airflow_home/tmp
sudo chown -R 50000:0 ./airflow_home
sudo chown -R 50000:0 ./dags

docker compose run --rm airflow-init
docker compose up -d
