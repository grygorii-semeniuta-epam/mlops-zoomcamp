# Homework answers

## Question 1. Select the Tool

You can use the same tool you used when completing the module,
or choose a different one for your homework.

What's the name of the orchestrator you chose? 

### Answer: Airflow

## Question 2. Version

What's the version of the orchestrator? 

### Answer: 2.11.0

## Question 3. Creating a pipeline

Let's read the March 2023 Yellow taxi trips data.

How many records did we load? 

### Answer: 3,403,766

## Question 4. Data preparation

Let's continue with pipeline creation.

We will use the same logic for preparing the data we used previously. 

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
```

Let's apply to the data we loaded in question 3. 

What's the size of the result? 

### Answer: 3,316,216

## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1.

* Fit a dict vectorizer.
* Train a linear regression with default parameters.
* Use pick up and drop off locations separately, don't create a combination feature.

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model.

What's the intercept of the model? 

### Answer: 24.77

## Question 6. Register the model 

The model is trained, so let's save it with MLFlow.

Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field):

### Answer: 4,534
