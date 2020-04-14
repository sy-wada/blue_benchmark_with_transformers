# coding=utf-8
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

report_columns = ['Class', 'TP', 'TN', 'FP', 'FN', 'Support',
                  'Precision', 'Recall', 'F1score', 'Accuracy']

def accuracy(tp, tn, fp, fn):
    return tp / (tp + fn)

def precision(tp, tn, fp, fn):
    return tp / (tp + fp)

def recall(tp, tn, fp, fn):
    return tp / (tp + fn)

def fb1(precision, recall):
    return 2 * precision * recall / (precision + recall)

def overall_acc(tp, tn, fp, fn):
    return np.sum(tp) / np.sum(tp + fn)

def a_p_r_f(tp, tn, fp, fn):
    acc = accuracy(tp, tn, fp, fn)
    prec =  np.nan_to_num(precision(tp, tn, fp, fn))
    rec = np.nan_to_num(recall(tp, tn, fp, fn))
    fscore =  np.nan_to_num(fb1(prec, rec))
    return acc, prec, rec, fscore

def classification_report(y_true, y_pred, label_list, drop_false=False,
                         micro=True, macro=True):
    #search false label position.
    if drop_false:
        for i, l in enumerate(label_list):
            if 'false' in l:
                false_id = i
    
    supports = Counter(y_true)
    supports = np.array([supports.get(i, 0) for i in range(len(set(y_true) | set(y_pred)))])

    cm = confusion_matrix(y_true, y_pred)
    
    tps = np.diag(cm) # TP
    fps = []
    fns = []
    for i, tp in enumerate(tps):
        fps.append(cm[:, i].sum() - tp) # FP
        fns.append(cm[i].sum() - tp) # FN
    fps = np.array(fps)
    fns = np.array(fns)
    
    df = pd.DataFrame({
                'Class': 0,
                'TP': tps,
                'TN': len(y_true) - tps - fps - fns,
                'FP': fps,
                'FN': fns,
                'Support': supports}).astype(int)
    
    df['Class'] = label_list
    if macro:
        macro_metrics = []
        acc, prec, rec, fscore = a_p_r_f(df.TP, df.TN, df.FP, df.FN)
        row = ['macro_include_false'] + [''] * 5
        row += [np.average(t) for t in [prec, rec, fscore, acc]]
        macro_metrics.append(pd.Series(row, index=report_columns))
        if drop_false:
            df_d = df.drop(false_id, axis=0)
            acc, prec, rec, fscore = a_p_r_f(df_d.TP, df_d.TN, df_d.FP, df_d.FN)
            row = ['macro_drop_false'] + [''] * 5
            row += [np.average(t) for t in [prec, rec, fscore, acc]]
            macro_metrics.append(pd.Series(row, index=report_columns))
    
    if micro:
        micro_metrics = []
        micro_include_false = df.sum()
        micro_include_false['Class'] = 'micro_include_false'
        micro_metrics.append(micro_include_false)
        if drop_false:
            micro_drop_false = df.drop(false_id, axis=0).sum()
            micro_drop_false['Class'] = 'micro_drop_false'
            micro_metrics.append(micro_drop_false)
    
        df = df.append(micro_metrics)
    
    # compute micro_metrics
    acc, prec, rec, fscore = a_p_r_f(df.TP, df.TN, df.FP, df.FN)

    df['Precision'] = prec
    df['Recall'] = rec
    df['F1score'] = fscore
    df['Accuracy'] = acc
        
    if macro:
        df = df.append(macro_metrics)
    
    return df.reset_index(drop=True)
