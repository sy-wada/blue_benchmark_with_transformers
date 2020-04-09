# coding=utf-8
import pandas as pd

from .pmetrics import classification_report

labels = ['contradiction', 'entailment', 'neutral']

def eval_mednli(y_true, y_pred, label_list):
    result = classification_report(y_true, y_pred, classes_=label_list, macro=True, micro=True)
    df = result.table
    
    # add micro average to the result
    row = df[df['Class'] == 'micro']
    results = {
        "TP": int(row.TP),
        "FP": int(row.FP),
        "FN": int(row.FN),
        "precision": float(row.Precision),
        "recall": float(row.Recall),
        "FB1": float(row['F-score']),
    }
    
    # add overall accuracy to the result
    results['overall_acc'] = float(df[df['Class'] == 'macro'].iloc[0].at['Accuracy'])
    return results, df