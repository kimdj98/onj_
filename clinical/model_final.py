import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from tabulate import tabulate
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.model_selection import GridSearchCV


def set(data, name='str'):
    tail = name.split('_')[-1]
    if tail == 'y':
        data = data.squeeze()
    data = pd.DataFrame(data)
    data = torch.tensor(data.values, dtype=torch.float32)
    return data





def find_par(data_x, data_y):
    # Suppress user warnings for cleaner output
    warnings.filterwarnings('ignore')

    
    class BinaryClassificationModel(nn.Module):
        def __init__(self, input_dim, layers):
            super(BinaryClassificationModel, self).__init__()
            layer_list = []
            previous_dim = input_dim
            for layer_dim in layers:
                layer_list.append(nn.Linear(previous_dim, layer_dim))
                layer_list.append(nn.ReLU())
                previous_dim = layer_dim
            layer_list.append(nn.Linear(previous_dim, 1))
            layer_list.append(nn.Sigmoid())
            self.model = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.model(x).squeeze()  # Ensuring output is of shape (batch_size,)

        def get_feature_coefficients(self):
            first_layer = self.model[0]
            if isinstance(first_layer, nn.Linear):
                return first_layer.weight.data



    class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim, layers, lr=0.001, batch_size=10, epochs=10):
            self.input_dim = input_dim
            self.layers = layers
            self.lr = lr
            self.batch_size = batch_size
            self.epochs = epochs
            self.model = BinaryClassificationModel(input_dim, layers)
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        def fit(self, X, y):
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1))
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(self.epochs):
                for inputs, labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    labels = labels.view_as(outputs)  # Ensure labels have the same shape as outputs
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            return self
        
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(inputs)
                return (outputs.numpy() > 0.5).astype(int)
        
        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(inputs)
                return np.vstack((1 - outputs.numpy(), outputs.numpy())).T

    # Prepare data
    X = set(data_x, 'data_x').numpy()
    y = set(data_y, 'data_y').numpy()

    # Ensure y is 1-dimensional
    y = y.ravel()

    # Check for NaN values in the data
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("The dataset contains NaN values. Please handle them before proceeding.")


    # Define a custom scoring function for debugging
    def custom_auc_score(estimator, X, y):
        probas = estimator.predict_proba(X)
        if np.isnan(probas).any():
            print("NaN values found in predictions.")
        if np.unique(probas[:, 1]).size == 1:
            print("Warning: All predictions are the same.")
        try:
            auc = roc_auc_score(y, probas[:, 1])
        except ValueError as e:
            print(f"Error calculating AUC: {e}")
            return np.nan
        return auc

    # Hyperparameter tuning using GridSearchCV
    # Node combinations
    node_combinations = [[[16,8]] , [[32,16]] , [[64,32]] , [[64,16]] ,  [[128,64]] , [[128,32]] , [[128,16]] , 
                         [[32,16,8]] , [[64,32,16]] , [[128,64,32]] , [[128,64,16]] , [[128,32,16]] , [[256,128,64]] , [[512,256,128]],
                         [[128,64,32,16]] , [[256,128,64,32]] , [[512,256,128,64]] , [[512,256,128,32]] ,
                         [[512,256,128,64,32]] , [[512,256,128,64,32,16]]]
    #node_combinations = [[128, 64, 32, 16], [128, 64, 16], [32, 16]]
    #node_combinations = [[32,16]]

    param_grid = {
        'layers' : node_combinations,     
        'lr': [0.001, 0.01],
        'batch_size': [10, 20],
        'epochs': [70, 100]
        
    }

    # Initialize the wrapper with the input dimension of the data
    model = SklearnPyTorchWrapper(input_dim=X.shape[1], layers=[32, 16])

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=custom_auc_score, error_score='raise')
    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation AUC: ", grid_search.best_score_)

    batch_size = grid_search.best_params_['batch_size']
    epochs = grid_search.best_params_['epochs']
    layers = grid_search.best_params_['layers']
    lr = grid_search.best_params_['lr']

    return batch_size , epochs , layers , lr



def make_model(X_train, train_dataset, val_dataset, test_dataset, batch , epochs , layers , lrate):

    layer_name = f'{layers[0][0]}'
    for i in range(len(layers[0])-1):
        layer_name += f'.{layers[0][i+1]}'
    model_name = path + f'/best_model_{batch}_{epochs}_{layer_name}_{lrate}.pt'





    # Define the model class
    class BinaryClassificationModel(nn.Module):
        def __init__(self, input_dim, layers):
            super(BinaryClassificationModel, self).__init__()
            layer_list = []
            previous_dim = input_dim
            for layer_dim in layers:
                layer_list.append(nn.Linear(previous_dim, layer_dim))
                layer_list.append(nn.ReLU())
                previous_dim = layer_dim
            layer_list.append(nn.Linear(previous_dim, 1))
            layer_list.append(nn.Sigmoid())
            self.model = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.model(x)
        
        '''
        def get_feature_coefficients(self):
            first_layer = self.model[0]
            if isinstance(first_layer, nn.Linear):
                return first_layer.weight.data
        '''

    
    node_combinations = layers

    # Loop to create models
    input_dim = X_train.shape[1]
    models = []
    for nodes in node_combinations:
        model = BinaryClassificationModel(input_dim, nodes)
        models.append(model)
        

    # Training and validation function
    def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs):
        best_val_loss = float('inf')
        best_model_wts = model.state_dict()

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Save the model if the validation loss is the best we've seen so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = model.state_dict()
                #torch.save(model.state_dict(), model_name)

        # Load the best model weights
        model.load_state_dict(best_model_wts)

    # Testing function to calculate AUC and Recall
    def test_model(model, test_loader):
        model.eval()
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                all_labels.extend(labels.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())
        auc_score = roc_auc_score(all_labels, all_outputs)
        recall = recall_score(all_labels, (torch.tensor(all_outputs) > 0.5).float().numpy())
        print(f"Test AUC: {auc_score:.4f}, Test Recall: {recall:.4f}")



    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Train and test each model
    for model in models:
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lrate)
        train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer)
        



    for i in range(len(node_combinations)):
        model = models[i]
        print(f'Node : {node_combinations[i]}')
        test_model(model, test_loader)

        
path = 'F:/노트북/Work/보건복지부과제/ONJ/onj/inAndOut_onj'
data_x = pd.read_csv(path + '/X_EW.csv', index_col=0)   
data_y = pd.read_csv(path + '/Y_EW.csv', index_col=0)



## 하위 5개 feature 뺀 다음 성능 비교하기 ## 
###########################################
######## 모델 저장 꼭 끄고 출력하기 #########
###########################################
bottom_5 = ['SM', 'DR', 'PMH__ANT', 'PMH_DB', 'PMH_t_RISED']
bottom_10 = bottom_5 + ['MH_HYPE', 'MH_RF', 'SEX', 'PMH_CK', 'MH_CVA']
#high_3 = ['PMH_MM' , 'ONJ_DIA_AGE' , 'SBP']
data_x = data_x.drop(columns=bottom_10)




X = set(data_x, 'data_x')
y = set(data_y, 'data_y')

# Split data into training, validation, and test sets
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, val_size, test_size])


# Train dataset을 X와 y로 분리
X_train = []
y_train = []

for data, target in train_dataset:
    X_train.append(data.numpy())
    y_train.append(target.numpy())

# 리스트를 텐서로 변환
X_train = np.stack(X_train)
y_train = np.stack(y_train)



#[batch_size , epochs , layers , lr] = find_par(X_train, y_train)







batch_size = 20 
epochs = 100
layers = [[128,32]] 
lr = 0.001
                   
make_model(X_train, train_dataset, val_dataset, test_dataset, batch_size , epochs , layers , lr)