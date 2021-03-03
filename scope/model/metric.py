import torch
from sklearn.metrics import roc_curve, auc


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=1):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def roc_auc(label_pred, label_gt, positive_class=1):
    """
    Softmax forces to probability of class traditionally used for roc analysis
    """
    with torch.no_grad():
        label_prob = torch.softmax(label_pred, dim=1)[:, positive_class]
        y_true = label_gt.detach().cpu().numpy().flatten()
        y_scores = label_prob.detach().cpu().numpy().flatten()

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc_score = auc(fpr, tpr)
    return roc_auc_score


def binary_roc_auc(label_pred, label_gt, positive_class=1):
    """
    ROC AUC but with hard predictions instead of probabilities
    """
    with torch.no_grad():
        label_pred = torch.argmax(label_pred, dim=1)
        y_true = label_gt.detach().cpu().numpy().flatten()
        y_scores = label_pred.detach().cpu().numpy().flatten()

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=positive_class)
        roc_auc_score = auc(fpr, tpr)
    return roc_auc_score


def recall(label_pred, label_gt) -> float:
    """
    Computes the recall
    """
    with torch.no_grad():
        prediction_bin = torch.argmax(label_pred, dim=1)
        TP = torch.mul(prediction_bin, label_gt).sum()
        FN = torch.mul(1 - prediction_bin, label_gt).sum()

        RC = float(TP) / (float(TP + FN) + 1e-6)

    return RC


def specificity(label_pred, label_gt):
    with torch.no_grad():
        prediction_bin = torch.argmax(label_pred, dim=1)
        TN = torch.mul(1 - prediction_bin, 1 - label_gt).sum()
        FP = torch.mul(prediction_bin, 1 - label_gt).sum()

        SP = float(TN) / (float(TN + FP) + 1e-6)

    return SP


def precision(label_pred, label_gt) -> float:
    """
    Computes the precision
    """
    with torch.no_grad():
        prediction_bin = torch.argmax(label_pred, dim=1)
        TP = torch.mul(prediction_bin, label_gt).sum()
        FP = torch.mul(prediction_bin, 1 - label_gt).sum()
        PC = float(TP) / (float(TP + FP) + 1e-6)

    return PC


def f1(label_pred, label_gt) -> float:
    """
    Computes the f1 score
    """
    with torch.no_grad():
        RC = recall(label_pred, label_gt)
        PC = precision(label_pred, label_gt)
        F1 = 2 * (RC * PC) / (RC + PC + 1e-6)

    return F1


def jaccard(label_pred, label_gt) -> float:
    """
    Computes the jaccard score
    """
    with torch.no_grad():
        prediction_bin = torch.argmax(label_pred, dim=1)
        INTER = torch.mul(prediction_bin, label_gt).sum()
        FP = torch.mul(prediction_bin, 1 - label_gt).sum()
        FN = torch.mul(1 - prediction_bin, label_gt).sum()

        JS = float(INTER) / (float(INTER + FP + FN) + 1e-6)

    return JS
