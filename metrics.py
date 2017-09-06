
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