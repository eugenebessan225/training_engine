import os
import re
import configparser
import shutil

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix


import mlflow
import pickle




config = configparser.ConfigParser()
config.read('config.ini')
FRAME_SIZE = config.getint('Default', 'FRAME_SIZE')
PATH_PROCESSED_DATA = config.get('Paths', 'PATH_PROCESSED_DATA')
PATH_PROCESSED_SCHEMA = config.get('Paths', 'PATH_PROCESSED_SCHEMA')
DIR_MLRUNS = config.get('Paths', 'DIR_MLRUNS')
experiment_name = "wear_detection_exp"


## loading processed data
with open(PATH_PROCESSED_DATA, 'rb') as file:
    print("\t \t Loading processed data =========>>>>>>>>")
    X_train, X_test, Y_train, Y_test = pickle.load(file)
print(X_train.head())

# Define a location for tracking experiments different from the default one
mlflow.set_tracking_uri("file:" + os.path.abspath(DIR_MLRUNS))
mlflow.sklearn.autolog(disable=True)
# Set up an experiment
experiment_name = "Wear Detection"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id


#ML models to train and test
knn_model = KNeighborsClassifier()
rdf_model = RandomForestClassifier()
lgr_model = LogisticRegression()
xgb_model = XGBClassifier()
mlp_model = MLPClassifier()


## Definition of grids
knn_grid = {
    'n_neighbors': [i for i in range(3,33,2)],
    'p':[1, 2],
    'weights' : ["uniform", "distance"]
}

rdf_grid = {
    'n_estimators': [i for i in range(50,550,50)],
    'max_features' : ["sqrt", "log2", None]
}

lgr_grid = {
    'penalty':['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'C': [0.1, 1, 10]
}

xgb_grid = {
    'learning_rate' : [0.1, 0.01, 0.001],
    'max_depth' : [i for i in range(3,10,1)],
    'n_estimators': [i for i in range(50,550,50)]
}

mlp_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (100,50,20), (100,100,50), (100,100,100), (200,100,50), (200,100,50, 20)],
    'activation': ['relu', 'tanh', 'identity', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd']
}


gs_knn = GridSearchCV(knn_model, knn_grid, cv=10)
gs_rdf = GridSearchCV(rdf_model, rdf_grid, cv=10)
gs_lgr = GridSearchCV(lgr_model, lgr_grid, cv=10)
gs_xgb = GridSearchCV(xgb_model, xgb_grid, cv=10)
gs_mlp = GridSearchCV(mlp_model, mlp_grid, cv=10)


models_train = {
    "knn": gs_knn,
    "rdf": gs_rdf,
    "lgr": gs_lgr,
    "xgb":gs_xgb,
    "mlp":gs_mlp
}


def model_run(model, X_train, X_test, Y_train, Y_test):
    # Train the model on the train data
    model.fit(X_train, Y_train)
    # Make predictions on the test data
    y_pred = model.best_estimator_.predict(X_test)

    # Calculate standard metrics for classifiers
    metrics = {
        "test_accuracy_score": accuracy_score(Y_test, y_pred),
        "test_f1_score": f1_score(Y_test, y_pred),
        "test_precision_score": precision_score(Y_test, y_pred),
    }
    model_path = "model"
    mlflow.sklearn.log_model(model.best_estimator_, model_path, registered_model_name=f"wear-detection_{run_name}")
    mlflow.log_params(model.best_estimator_.get_params())
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(PATH_PROCESSED_SCHEMA, model_path)



run_id = []
for run_name, mod in models_train.items():
    with mlflow.start_run(run_name=f'{run_name}', experiment_id=experiment_id) as run:
        # Train the model with the current values of hyperparameters, calculate scores, log with mlflow
        model_run(mod, X_train, X_test, Y_train, Y_test)
        run_id.append(run.info.run_id)



runs = []
for r_id in run_id:
    r_dict = {}
    r = mlflow.get_run(run_id=r_id)
    r_dict["id"] = r_id
    r_dict["test_accuracy_score"] = r.data.metrics["test_accuracy_score"]
    runs.append(r_dict)

best_one = sorted(runs, key=lambda x: x['test_accuracy_score'], reverse = True)[0]
print(best_one)


best_run = mlflow.get_run(run_id=best_one["id"])
target_directory = "../model"

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Define the source directory of the model
model_directory = best_run.info.artifact_uri+"/model"
# Copy or move the model files to the target directory
for root, dirs, files in os.walk(model_directory.split("//")[1]):
    for file in files:
        source_path = os.path.join(root, file)
        target_path = os.path.join(target_directory, file)
        shutil.copy(source_path, target_path)
print("Model saved !")