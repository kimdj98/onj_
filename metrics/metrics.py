# import torch
# from sklearn.metrics import roc_auc_score

# # Assuming y_true and y_pred are 1D tensors or lists of true labels and predicted probabilities respectively
# y_true = torch.tensor([0, 1, 0, 1])
# y_pred = torch.tensor([0.1, 0.4, 0.35, 0.8])

# # Convert tensors to numpy arrays if they aren't already
# y_true_np = y_true.numpy()
# y_pred_np = y_pred.numpy()

# # Calculate AUROC
# auroc = roc_auc_score(y_true_np, y_pred_np)
# print(f"Area Under the ROC curve: {auroc}")

# create AUROC metrics class for pytorch
import torch
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import BinaryAccuracy

# auroc = MulticlassAUROC(num_classes=2)
