import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
import shap
import numpy as np


#shap.initjs()

path = 'D:/노트북/Work/보건복지부과제/ONJ/onj/inAndOut_onj'
data_x = pd.read_csv(path + '/X_EW_new2.csv', index_col=0)         
data_y = pd.read_csv(path + '/Y_EW_new2.csv', index_col=0)



# 시드 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

## parameter setting ##
batchSize = 31
d1 = 0.362
d2 = 0.333
#d3 = 0.635
u1 = 410
u2 = 225
#u3 = 703



# Train/test split
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.4, random_state=SEED)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=SEED)

ten_X = torch.tensor(X_test.values, dtype=torch.float32)
ten_y = torch.tensor(y_test.values, dtype=torch.float32)

input_dim = data_x.shape[1]

class BestModel(nn.Module):
    def __init__(self):
        super(BestModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], u1)  # units1
        self.dropout1 = nn.Dropout(p=d1)  # dropout1

        self.fc2 = nn.Linear(u1 , u2)  # units2
        self.dropout2 = nn.Dropout(p=d2)  # dropout2

        #self.fc3 = nn.Linear(u2 , u3)  # units3
        #self.dropout3 = nn.Dropout(p=d3)  # dropout3

        self.fc_final = nn.Linear(u2, 1)  # Final output layer

        self.activation = nn.ReLU()  # ReLU activation as per original specification

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)

        x = self.activation(self.fc2(x))
        x = self.dropout2(x)

        #x = self.activation(self.fc3(x))
        #x = self.dropout3(x)

        x = torch.sigmoid(self.fc_final(x))
        return x
    


# Initialize the model and load the saved state dict
model = BestModel()
model.load_state_dict(torch.load(path + '/best_model_EW_new2.pth'))
model.eval()

#print("Expected input features:", X_train.shape[1])
#print("X_test_tensor shape:", ten_X.shape)




# Use SHAP's DeepExplainer for explaining the PyTorch model
explainer = shap.GradientExplainer(model, ten_X)  # Use a subset of test data for background distribution
shap_values = explainer.shap_values(ten_X)  # Explain predictions for the first 10 samples of X_test

shap_values = np.array(shap_values)
shap_values_0 = shap_values[:,:,0]

# Visualize SHAP values for the first sample
X_test = pd.DataFrame(X_test)
shap.summary_plot(shap_values_0, X_test)
shap.summary_plot(shap_values_0, X_test, max_display=X_test.shape[1])
