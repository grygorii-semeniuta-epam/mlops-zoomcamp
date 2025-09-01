# How to start MLFlow UI

Create and activate Conda environment `exp-tracking-env` with Python version `3.13.5`:

```Bash
conda create -n exp-tracking-env python=3.13.5
conda activate exp-tracking-env
```

Install requirements:

```Bash
pip install -r requirements.txt
```

requirements.txt:
```Text
mlflow
jupyter
scikit-learn
pandas
seaborn
hyperopt
xgboost
fastparquet
boto3
```

Start MLFlow UI:

```Bash
mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db
```
