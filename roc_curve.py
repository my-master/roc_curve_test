import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def roc_curve(y_true, y_prob):
    """
    :param y_true: numpy array of true Y values
    :param y_prob: numpy array of predicted Y values or probabilities
    :yield: fpr, tpr, threshold tuple
    """
    all_pos = np.count_nonzero(y_true)
    all_neg = y_true.size - all_pos
    p = np.argsort(-y_prob)
    y_prob_sorted = y_prob[p]
    y_true_sorted = y_true[p]
    threshes = y_prob_sorted
    thresh_max = np.max(y_prob) + 1
    fp = 0
    tp = 0
    for i in range(threshes.size):
        if threshes[i] != thresh_max:
            thresh_max = threshes[i]
            fpr = fp / all_neg
            tpr = tp / all_pos
            yield fpr, tpr, threshes[i]
        if y_true_sorted[i] == 1:
            tp +=1
        else:
            fp +=1
    yield fp / all_neg, tp / all_pos, 0 # this is (1, 1) point

def plot_roc_curve(y_true, y_probas):
    plt.figure(figsize=(16, 10))
    for clf_name, predicts in y_probas.items():
        rates = [(fpr, tpr) for fpr, tpr, _ in roc_curve(y_true, predicts)]
        fpr = [r[0] for r in rates]
        tpr = [r[1] for r in rates]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve for {} classifier (area = {:.2f})'.format(clf_name, roc_auc),
                 color=np.random.rand(3,1))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize='large')
        plt.ylabel('True Positive Rate', fontsize='large')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid()
    plt.show()

def test():
    y_true = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0])
    y_probas = {'probabilistic': np.array([0.3, 0.9, 0.2, 0.2, 0.6, 0.5, 0.1, 0.8, 0.5, 0.5]),
                'discrete': np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0]),
                'constant': np.zeros(10)}
    plot_roc_curve(y_true, y_probas)

test()