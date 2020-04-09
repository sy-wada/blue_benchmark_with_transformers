# coding=utf-8

import numpy as np
from .pmetrics import classification_report, micro


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
    
    metrics = classification_report(y_true, y_pred, macro=False,
                                    micro=False, classes_=label_list)
    
    df = metrics.table
    
    false_index = df[df['Class'].str.contains('false')].index[0]
    df.drop(false_index, inplace=True)
    
    # compute micro average.
    micro_drop_false = ['micro_drop_false']
    micro_drop_false.extend(micro(df.TP.sum(), df.TN.sum(), df.FP.sum(), df.FN.sum()))
    
    # compute macro average.
    macro_columns = ['Precision', 'Recall', 'F-score', 'Accuracy', 'Sensitivity',
                     'Specificity', 'PPV', 'NPV']
    row = [np.nan] * 4
    row += [df[t].mean() for t in macro_columns]
    row += [np.nan]
    macro_drop_false = ['macro_drop_false']
    macro_drop_false.extend(row)
    
    # append micro average and macro average.
    df.loc[df.index.max() + 1] = micro_drop_false
    df.loc[df.index.max() + 1] = macro_drop_false
    
    # add micro average result
    row = df[df['Class'] == 'micro_drop_false']
    results = {
        "TP": int(row.TP),
        "FP": int(row.FP),
        "FN": int(row.FN),
        "micro_precision": float(row.Precision),
        "micro_recall": float(row.Recall),
        "micro_FB1": float(row['F-score']),
    }
    
    # add macro average result
    row = df[df['Class'] == 'macro_drop_false']
    results.update({
        "macro_precision": float(row.Precision),
        "macro_recall": float(row.Recall),
        "macro_FB1": float(row['F-score']),
    })
    
    return results, df