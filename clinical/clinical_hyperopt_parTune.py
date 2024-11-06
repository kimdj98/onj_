import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 시드 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Define the hyperparameter search space
space = {
    'choice': hp.choice('num_layers',
                        [{'layers': 'two'},
                         {'layers': 'three',
                          'units3': hp.uniform('units3', 64, 1024),
                          'dropout3': hp.uniform('dropout3', 0.25, 0.75)}]),

    'units1': hp.uniform('units1', 64, 1024),
    'units2': hp.uniform('units2', 64, 1024),

    'dropout1': hp.uniform('dropout1', 0.25, 0.75),
    'dropout2': hp.uniform('dropout2', 0.25, 0.75),

    'batch_size': hp.uniform('batch_size', 28, 128),

    'nb_epochs': 100,
    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
    'activation': 'ReLU'
}

# Define a PyTorch model
class NNModel(nn.Module):
    def __init__(self, params):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(data_x.shape[1], int(params['units1']))
        self.dropout1 = nn.Dropout(p=params['dropout1'])

        self.fc2 = nn.Linear(int(params['units1']), int(params['units2']))
        self.dropout2 = nn.Dropout(p=params['dropout2'])

        if params['choice']['layers'] == 'three':
            self.fc3 = nn.Linear(int(params['units2']), int(params['choice']['units3']))
            self.dropout3 = nn.Dropout(p=params['choice']['dropout3'])
            self.fc_final = nn.Linear(int(params['choice']['units3']), 1)
        else:
            self.fc_final = nn.Linear(int(params['units2']), 1)

        self.activation = getattr(nn, params['activation'])()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)

        x = self.activation(self.fc2(x))
        x = self.dropout2(x)

        if hasattr(self, 'fc3'):
            x = self.activation(self.fc3(x))
            x = self.dropout3(x)

        x = torch.sigmoid(self.fc_final(x))
        return x

# Function to train and evaluate the model
def f_nn_cv(params):
    print('Params testing: ', params)

    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []

    for train_idx, val_idx in kfold.split(data_x):
        X_train_fold, X_val_fold = data_x.iloc[train_idx], data_x.iloc[val_idx]
        y_train_fold, y_val_fold = data_y.iloc[train_idx], data_y.iloc[val_idx]

        model = NNModel(params)
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters())
        elif params['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters())
        else:
            optimizer = optim.Adadelta(model.parameters())

        criterion = nn.BCELoss()
        X_train_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val_fold.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=int(params['batch_size']), shuffle=True)

        for epoch in range(params['nb_epochs']):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_auc = model(X_val_tensor).cpu().numpy()
            auc = roc_auc_score(y_val_fold, pred_auc)
            aucs.append(auc)
            print('Fold AUC:', auc)
            sys.stdout.flush()
    
    avg_auc = np.mean(aucs)
    print('Average AUC:', avg_auc)
    return {'loss': -avg_auc, 'status': STATUS_OK}



######path = 'D:/노트북/Work/보건복지부과제/ONJ/onj/inAndOut_onj'
data_x = pd.read_csv(path + '/data_X.csv', index_col=0)   
data_y = pd.read_csv(path + '/data_Y.csv', index_col=0)





# Hyperparameter optimization
trials = Trials()
best = fmin(f_nn_cv, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best: ', best)


