
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def cmb_eval(y_pred, y_test):

    classes = np.unique(y_test)

    acc_scores = [accuracy_score(y_test[y_test == cls], y_pred[y_test == cls]) for cls in classes]
    prec_scores = precision_score(y_test, y_pred, average=None, labels=classes)
    recal_scores = recall_score(y_test, y_pred, average=None, labels=classes)
    f1_scores = f1_score(y_test, y_pred, average=None, labels=classes)

    # Create a dictionary with class labels as keys and F1 scores as values
    acc_dict = {cls: val for cls, val in zip(classes, acc_scores)}
    prec_dict = {cls: val for cls, val in zip(classes, prec_scores)}
    recal_dict = {cls: val for cls, val in zip(classes, recal_scores)}
    f1_dict = {cls: val for cls, val in zip(classes, f1_scores)}

    return acc_dict, prec_dict, recal_dict, f1_dict