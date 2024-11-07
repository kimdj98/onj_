import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
import shap
import numpy as np

# 시드 고정
SEED = 21
torch.manual_seed(SEED)
np.random.seed(SEED)

## parameter setting ## parTune에서 print된 튜닝값 직접 입력
batchSize = 31
d1 = 0.362  ## dropout1
d2 = 0.333  ## dropout2
# d3 = 0.635           ## dropout3
u1 = 410  ## unit1
u2 = 225  ## unit2
# u3 = 703             ## unit3
opt = "adadelta? adam? rmsprop?"  ### optimizer만 line76에서 직접 설정


class BestModel(nn.Module):
    def __init__(self):
        super(BestModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], u1)  # units1
        self.dropout1 = nn.Dropout(p=d1)  # dropout1

        self.fc2 = nn.Linear(u1, u2)  # units2
        self.dropout2 = nn.Dropout(p=d2)  # dropout2

        # self.fc3 = nn.Linear(u2 , u3)  # units3                                                  ### 레이어 2개 or 3개에 따라 이 부분 수정
        # self.dropout3 = nn.Dropout(p=d3)  # dropout3                                             ### 여기

        self.fc_final = nn.Linear(u2, 1)  # Final output layer

        self.activation = nn.ReLU()  # ReLU activation as per original specification

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)

        x = self.activation(self.fc2(x))
        x = self.dropout2(x)

        # x = self.activation(self.fc3(x))                                                        ### 여기
        # x = self.dropout3(x)                                                                    ### 여기

        x = torch.sigmoid(self.fc_final(x))
        return x


path = "/mnt/aix22301/onj/code/clinical"
data_x = pd.read_csv(path + "/data_X2.csv", index_col=0)
data_y = pd.read_csv(path + "/data_Y2.csv", index_col=0)
model_name = path + "/best_model2.pth"  ## model1 or model2

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.4, random_state=SEED)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=SEED)


# Initialize the model with the optimized parameters
model = BestModel()

# Use Adam optimizer as optimizer=1 corresponds to 'adam'
# optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adam(model.parameters())
# optimizer = optim.RMSprop(model.parameters())


# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Convert the training and validation data to PyTorch tensors
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for the training data
train_dataset = TensorDataset(X_tensor, y_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=int(batchSize), shuffle=True)  # batch_size(rounded)

# Train the model (example training loop)
nb_epochs = 100  # As per the original code

for epoch in range(nb_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # After each epoch, calculate validation loss
    model.eval()  # Switch to evaluation mode for validation
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor).item()  # Validation loss

    print(f"Epoch [{epoch+1}/{nb_epochs}], Validation Loss: {val_loss:.4f}")

# Save the model only at the last epoch
torch.save(model.state_dict(), model_name)
print(f"Model saved as final_model.pth")

# Final evaluation on test data after training is complete
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

model.eval()  # Switch to evaluation mode for testing
with torch.no_grad():
    test_preds = model(X_test_tensor).cpu().numpy()
    test_preds_binary = (test_preds >= 0.5).astype(int)  # Binarize the predictions at 0.5 threshold

    # Calculate AUC and Recall on test data
    auc = roc_auc_score(y_test_tensor, test_preds)
    recall = recall_score(y_test_tensor, test_preds_binary)

    print(f"Final Test AUC: {auc:.4f}, Final Test Recall: {recall:.4f}")
