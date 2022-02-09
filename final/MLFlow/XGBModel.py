import argparse
import os
from flask import Flask, request
import requests
import threading

import mlflow

import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from skopt import BayesSearchCV
from skopt.utils import OptimizeResult
from skopt.space import Real, Categorical, Integer

MODIFIED_CSV = 'modified_dataset.csv'
MLFLOW_PORT = 5001

MLFLOW_THREAD = None


def ml():
    global MLFLOW_THREAD
    ret_val = ''
    f = open('tuned_model_run_id.txt', mode='rt')
    run_id = f.read()
    model_name_in_mlflow = "xgb-model"

    # check if optimal model run id exists or not
    if run_id == "":
        experiment_name = f'XGBBoost Model'

        # set experiment id for mlflow experiment
        try:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            experiment_id = mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_id=experiment_id)

        # read modified dataset
        data = pd.read_csv(MODIFIED_CSV, sep='\t')

        # get independent and dependent values
        target_column = data.columns.tolist()[0]
        feature_columns = data.columns[~data.columns.isin([target_column])]

        with mlflow.start_run() as run:
            y = data[target_column]
            X = data[feature_columns]

            # split data to train test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

            # define n-fold for cross validation
            cross_validation_n_fold = 5

            # define bayesian search described in the report
            opt = BayesSearchCV(
                xgb.XGBClassifier(),

                # define hyperparameters search space
                search_spaces={
                    'n_estimator': Integer(low=500, high=1500, prior='uniform'),
                    'max_depth': Integer(low=5, high=10, prior='uniform'),
                    'booster': Categorical(['dart', 'gbtree']),
                    'use_label_encoder': Categorical([False]),
                    'eval_metric': Categorical(['logloss']),
                },
                n_iter=1,
                cv=cross_validation_n_fold,
                scoring=lambda estimator, X, y: f1_score(y, estimator.predict(X)),
                return_train_score=True,
                verbose=0
            )

            # fit model to training data
            opt.fit(X_train, y_train)

            # get optimal model in defined hyperparameters space
            optimal_model: xgb.XGBClassifier = opt.best_estimator_

            # predict test dataset labels
            predicted_y_test = optimal_model.predict(X_test)

            # log optimal parameters to mlflow
            optimal_parameters = dict(opt.best_params_)
            changed_optimal_parameters = dict()
            for key in optimal_parameters.keys():
                changed_optimal_parameters[key + '-optimal'] = optimal_parameters[key]

            mlflow.log_params(changed_optimal_parameters)

            # log metrics
            accuracy = accuracy_score(y_test, predicted_y_test)
            f_score = f1_score(y_test, predicted_y_test)
            mlflow.log_metrics({
                'accuracy-optimal': accuracy,
                'f_score-optimal': f_score
            })

            ret_val = f'accuracy: {accuracy}, f_score: {f_score}'
            # log optimal model
            mlflow.xgboost.log_model(optimal_model, model_name_in_mlflow)

            # log results during hyperparameters tuning
            for i, result in enumerate(opt.cv_results_['params']):
                with mlflow.start_run(nested=True):
                    mlflow.log_params(dict(result))

                    train_score_sum = 0
                    validation_score_sum = 0
                    for k in range(cross_validation_n_fold):
                        train_score_sum += opt.cv_results_[f'split{k}_train_score'][i]
                        validation_score_sum += opt.cv_results_[f'split{k}_test_score'][i]

                    mlflow.log_metrics({
                        'mean_train_score': train_score_sum / cross_validation_n_fold,
                        'mean_validation_score': validation_score_sum / cross_validation_n_fold,
                    })
            MLFLOW_THREAD = threading.Thread(target=deploy, args=(run, model_name_in_mlflow))
            MLFLOW_THREAD.start()
    else:
        MLFLOW_THREAD = threading.Thread(target=deploy, args=(run_id, model_name_in_mlflow))
        MLFLOW_THREAD.start()
    return ret_val


api = Flask(__name__)


def deploy(run, model_name_in_mlflow):
    # deploy model (used in container)
    os.system(
        f'mlflow models serve -m "./mlruns/1/{run.info.run_id}/artifacts/{model_name_in_mlflow}" --no-conda '
        f'-h 0.0.0.0 -p {MLFLOW_PORT}')

    # save optimal model run id
    f = open('tuned_model_run_id.txt', mode='wt')
    f.write(str(run.info.run_id))
    f.close()


def send_to_mlflow():
    host = '127.0.0.1'
    url = f'http://{host}:{MLFLOW_PORT}/invocations'
    headers = {
        'Content-Type': 'application/json',
    }
    test_data = pd.read_csv(MODIFIED_CSV, sep='\t')
    http_data = test_data.to_json(orient='split')
    r = requests.post(url=url, headers=headers, data=http_data)
    return f'Predictions: {r.text}'


@api.route('/ml', methods=['POST'])
def analyze():
    phase = request.args.get('phase')
    if phase in ('dev', 'prod'):
        file_name = MODIFIED_CSV
    else:
        return "Not a valid phase"
    data_f = request.files.get('data_file')
    data_f.save(file_name)
    if phase == 'dev':
        ret_val = ml()
    else:
        ret_val = send_to_mlflow()
    return ret_val


if __name__ == '__main__':
    api.run('127.0.0.1', 8080)
