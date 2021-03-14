import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_roc(soft_prediction=None, label=None, tprs=None, fprs=None, save_path = None, plot_title=None,  line_color = None, std_intensity = .2):
    """
    Plot ROC curves

    Args:
        tprs: list of true positive rates
        fprs: list of false positive rates

    Returns:
        undefined
    """
    if save_path is not None:
        plt.ioff()
        plt.switch_backend('agg')

    if tprs is None:
        fpr, tpr, _ = roc_curve(label, soft_prediction)
        tprs = [tpr]
        fprs = [fpr]
        
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        aucs.append(roc_auc)
        tprs_interp.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        # plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3,
        #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=0.5, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    color = 'b'
    if line_color is not None: color = line_color
    plt.plot(mean_fpr, mean_tpr, color=color,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.2, alpha=.8)

    if len(fprs) > 1:
        std_tpr = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=std_intensity,
                         label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensibility (True Positive Rate)')
    if plot_title is None:
        plot_title = 'Receiver Operating Characteristic'
    plt.title(plot_title)
    plt.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.ion()
        plt.draw()
        plt.show()
