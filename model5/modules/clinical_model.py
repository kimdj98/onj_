import torch.nn as nn
import pandas as pd

class ClinicalModel(nn.Module):
    def __init__(self, HPARAMS):
        super(ClinicalModel, self).__init__()
        self.HPARAMS = HPARAMS
        self.fc1 = nn.Linear(HPARAMS["input_dim"], HPARAMS["u1"])  # units1
        self.dropout1 = nn.Dropout(p=HPARAMS["d1"])  # dropout1

        self.fc2 = nn.Linear(HPARAMS["u1"], HPARAMS["u2"])  # units2
        self.dropout2 = nn.Dropout(p=HPARAMS["d2"])  # dropout2

        self.fc_final = nn.Linear(HPARAMS["u2"], 1)  # Final output layer

        self.activation = nn.ReLU()  # ReLU activation as per original specification

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)

        x = self.activation(self.fc2(x))
        x = self.dropout2(x)

        # x = torch.sigmoid(self.fc_final(x))
        return x

path = "/mnt/aix22301/onj/code/clinical"
pt_CODE = pd.read_csv(path + "/pt_CODE.csv", index_col=0)
data_x = pd.read_csv(path + "/data_X.csv", index_col=0)
data_y = pd.read_csv(path + "/data_Y.csv", index_col=0)
load_path = path + "/best_model2.pth"  ## model1 or model2

HPARAMS = {
    "input_dim": data_x.shape[1],
    "u1": 410,
    "u2": 225,
    "d1": 0.362,
    "d2": 0.333,
    "load_path": load_path,
}