import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def roc_auc(label_pred, label_gt):
    label_pred = torch.argmax(label_pred, dim=1)
    y_true = label_gt.detach().numpy().flatten()
    y_scores = label_pred.detach().numpy().flatten()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_score = auc(fpr, tpr)
    return roc_auc_score
