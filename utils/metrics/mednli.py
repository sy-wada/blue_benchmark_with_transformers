# coding=utf-8
import pandas as pd
from sklearn.metrics import accuracy_score

from .common_metrics import classification_report

labels = ['contradiction', 'entailment', 'neutral']

def eval_mednli(y_true, y_pred, label_list):
    df = classification_report(y_true, y_pred, label_list, drop_false=False)
    
    # add micro average to the result
    row = df[df['Class'] == 'micro_include_false'].iloc[0]
    results = {
        "TP": row.TP,
        "FP": row.FP,
        "FN": row.FN,
        "precision": row.Precision,
        "recall": row.Recall,
        "FB1": row.F1score,
        "overall_acc": accuracy_score(y_true, y_pred),
    }

    return results, df
