import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
from sklearn.metrics import roc_auc_score, recall_score
import lime.lime_tabular
import shap

shap.initjs()

path = "/mnt/aix22301/onj/code/clinical/clinical"
X = pd.read_csv(path + "/X_EW.csv", index_col=0)
y = pd.read_csv(path + "/Y_EW.csv", index_col=0)

ten_X = torch.tensor(X.values, dtype=torch.float32)
ten_y = torch.tensor(y.values, dtype=torch.float32)


# Define the model class
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, layers):
        super(BinaryClassificationModel, self).__init__()
        layer_list = []
        previous_dim = input_dim
        for layer_dim in layers:
            for node in layer_dim:
                layer_list.append(nn.Linear(previous_dim, node))
                layer_list.append(nn.ReLU())
                previous_dim = node
        layer_list.append(nn.Linear(previous_dim, 1))
        layer_list.append(nn.Sigmoid())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


dummy = path + "/best_model_20_100_128.32_0.001.pt"  ################# model name ################
dum = dummy.split("/")[-1].split("_")[4].split(".")
layers = []
for i in range(len(dum)):
    layers.append(int(dum[i]))
layers = [layers]


input_dim = X.shape[1]
model = BinaryClassificationModel(input_dim, layers)

# Load the state dict (weights) into the model
model.load_state_dict(torch.load(dummy))


masker = shap.maskers.Independent(X)

# explainer = shap.Explainer(model, masker)
explainer = shap.DeepExplainer(model, ten_X)
shap_values = explainer.shap_values(ten_X)


shap_values = np.array(shap_values)
# shape (156,53,2) -> (156, 53)
shap_values_0 = shap_values[:, :, 0]  # label == 0
# shap_values_1 = shap_values[:,:,1]                                                                     # label == 1


# print(shap_values.shape)                                                                               # (674, 53, 1)
# print(X.shape)                                                                                         # (674, 53)
shap.summary_plot(shap_values_0, X)
shap.summary_plot(shap_values_0, X, max_display=X.shape[1])

# shap.dependence_plot("BMI", shap_values, X, interaction_index="SBP")                                   # relation btw two features

# shap.plots.force(explainer.expected_value[1], shap_values[0][:], X.iloc[0, :], matplotlib = True)      # personal report

# shap.decision_plot(explainer.expected_value[1], shap_values_1, X.columns)
# It visually depicts the model decisions by mapping the cumulative SHAP values for each prediction.
