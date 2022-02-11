from sklearn.metrics import accuracy_score, f1_score

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel

import numpy as np
import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import lmdb
import torch.utils.data
from tqdm import tqdm
import mlflow as mlf
import optuna
import torchviz
import torch


class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, num_feats, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        self.NUM_FEATS = num_feats
        self.NUM_INT_FEATS = 2
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(float(val))
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)


class ModelTrainer:

    def __init__(self,
                 train_data_loader,
                 validation_data_loader,
                 model,
                 criterion,
                 optimizer,
                 epochs,
                 device,
                 ):

        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        model: WideAndDeepModel
        optimizer: torch.optim.Optimizer
        criterion: torch.nn.BCELoss
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = epochs

        self.device = device

        self.test_predicts_proba = None
        self.test_true_labels = None

    def train_loop(self):
        data_loader = self.train_data_loader
        self.model.train()

        targets, predicts_probabilities = list(), list()

        # sum_loss = 0
        # sum_correct = 0
        for i, (fields, target) in enumerate(data_loader):
            fields, target = fields.to(self.device), target.to(self.device)

            predicts_proba: torch.Tensor = self.model(fields)
            loss = self.criterion(predicts_proba, target.float())

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # predicts = self.get_predicts_label(predicts_proba)

            # sum_loss += loss.item()
            # sum_correct += predicts.eq(target).sum().item()
            targets.extend(target.tolist())
            predicts_probabilities.extend(predicts_proba.tolist())

        targets = torch.Tensor(targets, device=self.device)
        predicts_proba = torch.Tensor(predicts_probabilities, device=self.device)

        # n_batches = len(data_loader)
        # train_loss = sum_loss / n_batches
        # train_accuracy = sum_correct / n_batches

        return targets, predicts_proba

    def get_predicts_label(self, predicts_prob):
        predicts_label = torch.zeros_like(predicts_prob, device=self.device, dtype=torch.int8)
        threshold = .5
        predicts_label[predicts_prob >= threshold] = 1
        predicts_label[predicts_prob < threshold] = 0
        return predicts_label

    def test_loop(self):
        data_loader = self.validation_data_loader
        self.model.eval()

        targets, predicts_probabilities = list(), list()

        # sum_loss = 0
        # sum_correct = 0
        with torch.no_grad():
            for fields, target in data_loader:
                fields, target = fields.to(self.device), target.to(self.device)
                predicts_proba = self.model(fields)

                loss = self.criterion(predicts_proba, target.float())

                # sum_loss += loss.item()

                # predicts = self.get_predicts_label(predicts_proba)

                # sum_correct += predicts.eq(target).sum().item()

                targets.extend(target.tolist())
                predicts_probabilities.extend(predicts_proba.tolist())

        targets = torch.Tensor(targets, device=self.device)
        predicts_proba = torch.Tensor(predicts_probabilities, device=self.device)

        # n_batches = len(data_loader)
        # test_loss = sum_loss / n_batches
        # test_accuracy = sum_correct / n_batches

        return targets, predicts_proba

    def calculate_metrics(self, trues, predicts_proba):

        # calculate metrics in each batches
        loss = self.criterion(predicts_proba, trues.float()).item()

        auc = roc_auc_score(trues, predicts_proba)

        predicts = self.get_predicts_label(predicts_proba)

        accuracy = accuracy_score(trues, predicts)
        f_score = f1_score(trues, predicts)
        return auc, accuracy, f_score, loss

    def fit(self):
        print("{:<15}{:<25}{:<25}{:<25}{:<25}".format('Epoch', 'Train Loss', 'Validation Loss', 'Train F-Score', 'Validation F-Score'))
        print('-' * 115)

        # train per epochs
        for epoch_i in range(self.epochs):
            train_true_labels, train_predicts_proba = self.train_loop()

            train_auc, train_accuracy, train_f_score, train_loss = self.calculate_metrics(train_true_labels, train_predicts_proba)

            mlf.log_metrics({
                'train_AUC': train_auc,
                'train_accuracy': train_accuracy,
                'train_f1score': train_f_score,
                'train_loss': train_loss,
            })

            test_true_labels, test_predicts_proba = self.test_loop()

            validation_auc, validation_accuracy, validation_f_score, validation_loss = self.calculate_metrics(test_true_labels, test_predicts_proba)

            mlf.log_metrics({
                'validation_AUC': validation_auc,
                'validation_accuracy': validation_accuracy,
                'validation_f1score': validation_f_score,
                'validation_loss': validation_loss,
            })

            print("{:<15}{:<25}{:<25}{:<25}{:<25}".format(epoch_i + 1, train_loss, validation_loss, train_f_score, validation_f_score))
            print('-' * 115)

        return self

    def predict(self, test_data_loader):

        # predict test dataset labels
        X, y = next(iter(test_data_loader))
        X: torch.Tensor

        self.test_true_labels = y
        self.test_predicts_proba = self.model(X)
        return self

    def compute_test_metrics(self):

        # compute test metrics on test dataset
        self.test_true_labels = self.test_true_labels.detach()
        self.test_predicts_proba = self.test_predicts_proba.detach()

        test_auc, test_accuracy, test_f_score, test_loss = self.calculate_metrics(self.test_true_labels,
                                                                                  self.test_predicts_proba)

        return test_auc, test_accuracy, test_f_score, test_loss


def objective_function(trial: optuna.Trial):
    params = {
        'learning_rate': trial.suggest_discrete_uniform('learning_rate', 1e-4, 6 * 1e-4, q=1 * 1e-4),
        'eps': trial.suggest_categorical('eps', [1e-8, ]),
        'weight_decay': trial.suggest_discrete_uniform('weight_decay', 1 * 1e-6, 5 * 1e-6, q=1 * 1e-6),
        'dropout': trial.suggest_discrete_uniform('dropout', 0.25, 0.35, q=0.05),
        'batch_size': trial.suggest_categorical('batch_size', [512]),
        'amsgrad': trial.suggest_categorical('amsgrad', [False, True]),
        'epochs': trial.suggest_categorical('epochs', [50]),
    }
    experiment_name = DEEP_MODEL_NAME

    try:
        experiment_id = mlf.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlf.create_experiment(experiment_name)

    mlf.set_experiment(experiment_id=experiment_id)

    with mlf.start_run(run_name=f'trial: {trial.number}'):
        device_str = 'cpu'

        device = torch.device(device_str)

        data_path = DATASET_PATH
        data = CriteoDataset(num_feats=14, dataset_path=data_path)

        hyperparameters = {
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'dropout': params['dropout'],
            'eps': params['eps'],
            'amsgrad': params['amsgrad'],
            'epochs': params['epochs'],
        }

        print('*' * 115)
        for k, v in hyperparameters.items():
            print("{:<15}{:<25}".format(k, str(v)))

        print('*' * 115)

        dataset_size = len(data)

        train_set_size = int(dataset_size * 0.8)
        validation_set_size = int(dataset_size * 0.1)
        test_set_size = dataset_size - train_set_size - validation_set_size

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data, (train_set_size,
                                                                                          validation_set_size,
                                                                                          test_set_size))

        train_data_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        validation_data_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=test_set_size, shuffle=True)

        complexity_metric = 16
        if DEEP_MODEL_NAME == XDFM_STR:
            model = ExtremeDeepFactorizationMachineModel(
                data.field_dims,
                embed_dim=complexity_metric,
                cross_layer_sizes=(complexity_metric, complexity_metric),
                split_half=False,
                mlp_dims=(complexity_metric, complexity_metric),
                dropout=hyperparameters['dropout']
            )

        else:
            model = WideAndDeepModel(
                data.field_dims,
                embed_dim=complexity_metric,
                mlp_dims=(complexity_metric, complexity_metric),
                dropout=hyperparameters['dropout']
            )

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            eps=hyperparameters['eps'],
            amsgrad=hyperparameters['amsgrad']
        )

        mlf.log_params(hyperparameters)
        test_auc, test_accuracy, test_f_score, test_loss = ModelTrainer(
            train_data_loader,
            validation_data_loader,
            model,
            criterion,
            optimizer,
            hyperparameters['epochs'],
            device,
        ).fit().predict(test_data_loader).compute_test_metrics()

        mlf.log_metrics({
            'test_AUC': test_auc,
            'test_accuracy': test_accuracy,
            'test_f1score': test_f_score,
            'test_loss': test_loss,
        })

        print('*' * 115)
        print("{:<15}{:<25}{:<25}{:<25}{:<25}".format('Test Result', 'Loss', 'Accuracy', 'AUC', 'F-Score'))
        print('-' * 115)
        print("{:<15}{:<25}{:<25}{:<25}{:<25}".format('', test_loss, test_accuracy, test_auc, test_f_score))
        print('-' * 115)

    return test_f_score


# Choose Deep Model
XDFM_STR = 'Extreme Deep Factorization Model'
WADM_STR = 'Wide & Deep Model'
DEEP_MODEL_NAME = XDFM_STR

# modified dataset path
DATASET_PATH = 'modified_dataset.csv'


def main():
    # define hyperparameters tuning object
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    # run hyperparameters optimization
    study.optimize(objective_function, n_trials=15)

    # get best trial during hps optimization
    best_trial = study.best_trial

    # get best trial hps
    params = best_trial.params

    # print values
    print('/' * 115)
    for k, v in params.items():
        print("{:<15}{:<25}".format(k, str(v)))

    print('/' * 115)

    with mlf.start_run(run_name=f'optimal model'):
        # define pytorch tensors devices
        device_str = 'cpu'
        device = torch.device(device_str)

        # raed dataset
        data_path = DATASET_PATH
        data = CriteoDataset(num_feats=14, dataset_path=data_path)

        # get hps from trial
        hyperparameters = {
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'dropout': params['dropout'],
            'eps': params['eps'],
            'amsgrad': params['amsgrad'],
            'epochs': params['epochs'],
        }

        # print hps
        print('*' * 115)
        for k, v in hyperparameters.items():
            print("{:<15}{:<25}".format(k, str(v)))

        print('*' * 115)

        # train test validation split data
        dataset_size = len(data)

        train_set_size = int(dataset_size * 0.8)
        validation_set_size = int(dataset_size * 0.1)
        test_set_size = dataset_size - train_set_size - validation_set_size

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data, (train_set_size,
                                                                                          validation_set_size,
                                                                                          test_set_size))

        train_data_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        validation_data_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=test_set_size // 2, shuffle=True)

        # experiment_id = 3
        # run_id = '4f95dd52487d4f959b980050d1ea0f98'
        # path = f'./mlruns/{experiment_id}/{run_id}/artifacts/{DEEP_MODEL_NAME}/'
        # # path = f'./mlruns/3/4f95dd52487d4f959b980050d1ea0f98/artifacts/Extreme Deep Factorization Model/'
        # model = mlf.pytorch.load_model(path)
        #
        # import hiddenlayer as hl
        # X, y = next(iter(test_data_loader))
        # yhat = model(X)
        # torchviz.make_dot(yhat, params=dict(list(model.named_parameters()))).render(f'deep_model_archits/{DEEP_MODEL_NAME} tz result', format="png")
        #
        # graph = hl.build_graph(model, X, )
        # graph.theme = hl.graph.THEMES['blue'].copy()
        # graph.save(f'deep_model_archits/{DEEP_MODEL_NAME} hl result', format='png')
        #
        # exit(0)

        # instantiate model object
        complexity_metric = 16
        if DEEP_MODEL_NAME == XDFM_STR:
            model = ExtremeDeepFactorizationMachineModel(
                data.field_dims,
                embed_dim=complexity_metric,
                cross_layer_sizes=(complexity_metric, complexity_metric),
                split_half=False,
                mlp_dims=(complexity_metric, complexity_metric),
                dropout=hyperparameters['dropout']
            )

        else:
            model = WideAndDeepModel(
                data.field_dims,
                embed_dim=complexity_metric,
                mlp_dims=(complexity_metric, complexity_metric),
                dropout=hyperparameters['dropout']
            )

        # define binary cross entropy loss
        criterion = torch.nn.BCELoss()

        # define Adam optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            eps=hyperparameters['eps'],
            amsgrad=hyperparameters['amsgrad']
        )

        # log hps to mlflow
        mlf.log_params(hyperparameters)

        # get results on test data
        test_auc, test_accuracy, test_f_score, test_loss = ModelTrainer(
            train_data_loader,
            validation_data_loader,
            model,
            criterion,
            optimizer,
            hyperparameters['epochs'],
            device,
        ).fit().predict(test_data_loader).compute_test_metrics()

        # log metrics to mlflow
        mlf.log_metrics({
            'optimal-test_AUC': test_auc,
            'optimal-test_accuracy': test_accuracy,
            'optimal-test_f1score': test_f_score,
            'optimal-test_loss': test_loss,
        })

        # log model to mlflow
        mlf.pytorch.log_model(model, DEEP_MODEL_NAME)

        # print metric results
        print('*' * 115)
        print("{:<15}{:<25}{:<25}{:<25}{:<25}".format('Test Result', 'Loss', 'Accuracy', 'AUC', 'F-Score'))
        print('-' * 115)
        print("{:<15}{:<25}{:<25}{:<25}{:<25}".format('', test_loss, test_accuracy, test_auc, test_f_score))
        print('-' * 115)


if __name__ == '__main__':
    main()
