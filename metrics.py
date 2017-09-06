
"""
    This script defines the functions used for evaluating the quality of the comma placement 
    model. This includes functions for evaluating the training as well as final metrics, such
    as precision and recall.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def plot_learning_curves(history):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.plot(np.arange(len(history.history['loss'])), history.history['loss'], 'o-b', label='Training Loss')
    ax.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'], 'o-r', label='Validation Loss')
    ax.set(xlabel='Epoch', ylabel='Log10 Loss', title='Learning Curves')
    ax.legend(loc="upper right")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    print('Final training loss  ', history.history['loss'][-1])
    print('Final validation loss', history.history['val_loss'][-1])

def ticks_and_gridlines(ax):
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    ax.set(xticks=np.arange(0,1,0.05), yticks=np.arange(0,1,0.05))
    ax.grid(color='gray', linestyle='--', linewidth=.5, which='major');

def collapse_multi_task_to_single_metric(y_hat, y):
    """
        The output of the model is multidimensional. We need to compress this into 
        a single value, so we can do precision and recall analysis.
    """
    return y_hat, y

def make_roc_pr_plot(y_hat, y):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(28,14))

    # Initializing variables that will be updated in the loop
    aggregate_tpr = 0.0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    aggregate_roc_auc = 0
    aggregate_pr_auc = 0

    for idx in range(y_hat.shape[1]):
        # Calculating false and true positive rate
        fpr, tpr, threshold = roc_curve(y[:, idx], y_hat[:, idx])
        roc_auc = auc(fpr, tpr)
        aggregate_tpr += np.interp(mean_fpr, fpr, tpr)
        
        # Calculating precision and recall
        precision, recall, threshold = precision_recall_curve(y[:, idx], y_hat[:, idx]);
        pr_auc = auc(recall, precision)
        aggregate_pr_auc += pr_auc

        # Plotting the metrics
        ax0.plot(fpr, tpr, 'ob-', color='grey', alpha=0.6)
        ax1.plot(recall, precision, 'o-', color='grey', alpha=0.6)
    
    # Plotting the average metrics
    mean_tpr = np.divide(aggregate_tpr, (idx+1))
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    mean_pr_auc = aggregate_pr_auc/(idx+1)

    ax0.plot(mean_fpr, mean_tpr, lw=3, color='r', label='Average recall ({0:.2f})'.format(mean_roc_auc))
    ax1.plot(1,1, lw=3, color='r', label='Mean PR AUC ({0:.2f})'.format(mean_pr_auc))

    # Styling the ticks and gridlines
    ticks_and_gridlines(ax0)
    ticks_and_gridlines(ax1)
    # Adding legends
    ax0.legend()
    ax1.legend()
    plt.show()

    def reverse_embedding(x, y, reverse_vocabulary, y_hat = None, threshold=0.9):
        """
            

            If comma placement predictions are supplied it also evaulates true positives, 
            false positives and false negatives.
        """
        if y_hat is None:
            words = [reverse_vocabulary[int_rep] for int_rep in x]
            idx = np.where(y==1)[0][0]
            words[idx] = words[idx]+',' 
            print(' '.join(words))
        else:
            words = [reverse_vocabulary[int_rep] for int_rep in x]
            idx_true = np.where(y==1)[0]
            idx_pred = np.where(y_hat>threshold)[0]
            for true_idx in idx_true:
                if true_idx in idx_pred:
                    words[true_idx] = words[true_idx]+', [tp]'
                else:
                    words[true_idx] = words[true_idx]+', [fn]'
            for pred_idx in idx_pred:
                if pred_idx not in idx_true:
                    words[pred_idx] = words[pred_idx]+', [fp]'
            print(' '.join(words))

def get_pr_per_comma(y, y_hat, threshold):
    P = np.sum(y)
    tp = 0
    fp = 0
    fn = 0
    for idx in range(y_hat.shape[0]):
        idx_true = np.where(y[idx, :]==1)[0]
        idx_pred = np.where(y_hat[idx, :]>threshold)[0]
        for true_idx in idx_true:
            if true_idx in idx_pred:
                tp=tp+1
            else:
                fn=fn+1
        for pred_idx in idx_pred:
            if pred_idx not in idx_true:
                fp=fp+1

    precision = tp/(tp+fp)
    recall = tp/P
    return precision, recall

def pr_analysis(y_test, y_hat):
    thresholds = np.arange(0,1,0.01)
    pr = np.zeros((1, len(thresholds)))
    re = np.zeros((1, len(thresholds)))
    for idx, threshold in enumerate(thresholds):
        pr[0, idx], re[0, idx] = get_pr_per_comma(y_test, y_hat, threshold=threshold)