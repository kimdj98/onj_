# utils/plotting.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_auroc(y_true, y_scores, epoch, title="roc"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - Epoch {epoch}")
    plt.legend(loc="lower right")
    plt.savefig(f"{title}.png")
    plt.close()
