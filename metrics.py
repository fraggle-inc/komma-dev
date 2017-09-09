"""
    This script defines the functions used for evaluating the quality of the comma placement 
    model. This includes functions for evaluating the training of the model, the performance, as well as final metrics, such
    as precision and recall.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import keras

class callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.sentence_accuracy = []
        self.pr_scores = []
        self.pr_auc = []
        self.f1_scores = []
        self.precision = []
        self.recall = []
 
    def on_train_end(self, logs={}):
        return
 
    #def on_epoch_begin(self, logs={}):
    #    return
 
    def on_epoch_end(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_hat = self.model.predict_proba(X)
        
        precision, recall, fpr = pr_analysis(y, y_hat)
        thresholds = np.arange(0, 1, 0.01)
        best_threshold, best_f1, precision_at_threshold, recall_at_threshold = get_optimal_threshold(precision, recall, thresholds)
        sentence_accuracy = get_sentence_accuracy(y, y_hat, best_threshold)        
        pr_auc = auc(recall, precision)
        
        self.f1_scores.append(best_f1[0])
        self.precision.append(precision_at_threshold)
        self.recall.append(recall_at_threshold)
        self.pr_auc.append(pr_auc)
        self.sentence_accuracy.append(sentence_accuracy)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

def plot_metrics(metric_callback):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.plot(np.arange(len(metric_callback.f1_scores)), metric_callback.f1_scores,
            'o-b',
            label='Optimal F1 score')
    ax.plot(np.arange(len(metric_callback.sentence_accuracy)), metric_callback.sentence_accuracy,
            'o-r',
            label='Sentence accuracy')
    ax.plot(np.arange(len(metric_callback.recall)), metric_callback.recall,
            'o-m',
            label='Comma Precision at optimal threshold')
    ax.plot(np.arange(len(metric_callback.precision)), metric_callback.precision,
            'o-k',
            label='Comma Recall at optimal threshold')
    ax.set(xlabel='Epoch', ylabel='Metric', title='Learning Metrics')
    ax.legend(loc="upper right")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

def plot_learning_curves(history):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.plot(np.arange(len(history.history['loss'])), history.history['loss'], 'o-b', label='Training Loss')
    ax.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'], 'o-r', label='Validation Loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Learning Curves')
    ax.legend(loc="upper right")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    print('Final training loss  ', history.history['loss'][-1])
    print('Final validation loss', history.history['val_loss'][-1])

def ticks_and_gridlines(ax):
    '''This function sets the limits of the x and y axis to [0, 1] and adds
    gridlines in gray on the major axis. Used for ROC and PR curves.'''
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    ax.set(xticks=np.arange(0,1,0.1), yticks=np.arange(0,1,0.1))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.grid(color='gray', linestyle='--', linewidth=.5, which='major');

def make_roc_pr_plot_per_class(y_hat, y):
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

def make_roc_pr_plot(y, y_hat):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(28,14))
    # Calculating metrics
    precision, recall, fpr = pr_analysis(y, y_hat)
    roc_auc = auc(fpr, recall)
    pr_auc = auc(recall, precision)
    # Plotting the metrics
    ax0.plot(fpr, recall, 'ob-', label='AUC ({0:.2f}'.format(roc_auc))
    ax1.plot(recall, precision, 'o-', label='AUC ({0:.2f}'.format(pr_auc))
    # Showing the optimal f1 score
    thresholds = np.arange(0,1,0.01)
    threshold, best_f1, precision_at_threshold, recall_at_threshold = get_optimal_threshold(precision, recall, thresholds)
    ax1.plot(recall_at_threshold, precision_at_threshold,
        'or',
        markerfacecolor='red', 
        markersize=12,
        label='Optimal f1 score')
    ax1.plot([recall_at_threshold, recall_at_threshold], [0, precision_at_threshold], '-r')
    ax1.text(recall_at_threshold/2, precision_at_threshold-0.025,
            '{:0.2f}'.format(precision_at_threshold[0]),
            fontsize=20)
    ax1.text(recall_at_threshold-0.025, precision_at_threshold/2,
            '{:0.2f}'.format(recall_at_threshold[0]),
            rotation=90,
            fontsize=20)
    ax1.plot([0, recall_at_threshold], [precision_at_threshold, precision_at_threshold], '-r')
    ax1.text(.2, .5,
            'Threshold for best score: {:0.2f}'.format(threshold[0]),
            fontsize=20)
    # Adding labels to the axis
    ax0.set(xlabel='False positive rate', ylabel='True positive rate (Recall)')
    ax1.set(xlabel='Recall', ylabel='Precision')
    # Styling the ticks and gridlines
    ticks_and_gridlines(ax0)
    ticks_and_gridlines(ax1)
    # Adding legends
    ax0.legend()
    ax1.legend()
    plt.show()

def get_sentence_accuracy(y, y_hat, threshold):
    pred = y_hat>threshold
    n_correct = np.sum(np.all(np.equal(y, pred), axis=1))
    sentence_accuracy = n_correct/y.shape[0]
    return sentence_accuracy

def get_pr_per_comma(y, y_hat, threshold):
    """
        Input:
        y: Ground truth comma placement. 
        y_hat: Predicted comma placement
        threshold: The probability threshold that turns y_hat into a binary vector.

        Output:
        precision
        recall
        fpr (false positive rate)

        TODO:
        Currently the number of Negatives (words without commas) is unknown. We estimate it by
        subtracting the number of commas from the total dimension of y. But this includes all the 
        padded sentences, so the number of Negative will be overestimated. This makes the ROC curve
        look better than it is.
    """
    P = np.sum(y)
    N = (y.shape[0]*y.shape[1])-P # Bad approximation of actual number of Negatives (words without commas)
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

    precision = np.divide(tp, (tp+fp))
    recall = np.divide(tp, P)
    fpr = np.divide(fp, N)
    return precision, recall, fpr

def pr_analysis(y, y_hat):
    """
        Input:
        y: Ground truth comma placement. 
        y_hat: Predicted comma placement
        threshold: The probability threshold that turns y_hat into a binary vector.

        Output:
        precision
        recall
        fpr (false positive rate)
    """
    thresholds = np.arange(0,1,0.01)
    precision = np.zeros((len(thresholds)))
    recall = np.zeros((len(thresholds)))
    fpr = np.zeros((len(thresholds)))
    for idx, threshold in enumerate(thresholds):
        precision[idx], recall[idx], fpr[idx] = get_pr_per_comma(y, y_hat, threshold=threshold)
    return precision, recall, fpr

def get_optimal_threshold(precision, recall, thresholds):
    """
        Estimates the best threshold based on a F1 analysis.
        Returns the threshold together with the corresponding precision and recall.
    """
    f1_score = 2*(precision*recall)/(precision+recall)
    max_idx = np.where(f1_score == np.nanmax(f1_score))
    threshold = thresholds[max_idx]
    precision_at_threshold = precision[max_idx]
    recall_at_threshold = recall[max_idx]
    best_f1 = f1_score[max_idx]
    return threshold, best_f1, precision_at_threshold, recall_at_threshold