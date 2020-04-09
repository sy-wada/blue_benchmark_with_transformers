# coding=utf-8

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
    micro_drop_false = ['micro_drop_false']
    micro_drop_false.extend(micro(df.TP.sum(), df.TN.sum(), df.FP.sum(), df.FN.sum()))
    df.loc[df.index.max() + 1] = micro_drop_false
    
    row = df[df['Class'] == 'micro_drop_false']
    results = {
        "TP": int(row.TP),
        "FP": int(row.FP),
        "FN": int(row.FN),
        "precision": float(row.Precision),
        "recall": float(row.Recall),
        "FB1": float(row['F-score']),
    }
    return results, df