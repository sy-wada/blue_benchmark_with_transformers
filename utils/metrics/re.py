# coding=utf-8
import pandas as pd

from .common_metrics import classification_report


def calculate_metrics(y_true, y_pred, label_list):
    # delete unused labels to calculate metrics.
    if len(label_list) != len(set(y_true + y_pred)):
        # replace index into label.
        y_true = list(map(lambda x: label_list[x], y_true))
        y_pred = list(map(lambda x: label_list[x], y_pred))
        label_list = [label for label in label_list if label in set(y_true + y_pred)]
        # replace label into new index in y_true and y_pred.
        y_true = [label_list.index(value) for value in y_true]
        y_pred = [label_list.index(value) for value in y_pred]
    
    df = classification_report(y_true, y_pred, label_list, drop_false=True,
                              micro=True, macro=True)
    
    row = df[df['Class'] == 'micro_drop_false'].iloc[0]
    

    results = {
        "TP": row.TP,
        "FP": row.FP,
        "FN": row.FN,
        "precision": row.Precision,
        "recall": row.Recall,
        "FB1": row.F1score,
        "FB1_macro": df[df['Class'] == 'macro_drop_false'].iloc[0].F1score,
    }
    return results, df
