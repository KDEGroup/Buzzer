from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np 

    

def plot_roc(args, mem, non, name='metrics'):
    ones = np.ones_like(mem)
    zeros = np.zeros_like(non)

    y_score = np.concatenate([mem, non])
    y_true = np.concatenate([ones, zeros])
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)

    data = []
    for r in (0.0001, 0.001, 0.01):
        index = len(fpr[fpr < r]) - 1
        data.append(f'{tpr[index] * 100:.02f}% TPR at {r * 100:.02f}% FPR')
    data.append(f'AUC: {roc_auc_score(y_true, y_score)}')
    metrics = '\n'.join(data)    
    return metrics
